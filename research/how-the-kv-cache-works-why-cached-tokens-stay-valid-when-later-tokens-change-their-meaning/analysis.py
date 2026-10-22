"""
How the KV cache works: why cached tokens stay valid when later tokens change their meaning.

Five CPU-only experiments on EleutherAI Pythia (decoder-only, causal, rotary positions) and
bert-base-uncased (bidirectional encoder):

  E1 exactness     greedy decoding with the KV cache vs full recomputation without it
  E2 immutability  K, V and hidden states of a prefix before/after appending text that
                   changes its meaning (Pythia), vs the same prefix tokens in BERT
  E3 agility       where the disambiguation happens: sense probe at the old position vs a
                   new position, and attention from the new position to the old one
  E4 cost          decode latency per token with and without the cache (timing only) and
                   the analytical KV-memory formula for public model configurations
  E5 approximation where the cache stops being exact: int8/int4/int2 KV quantisation,
                   sliding-window eviction with and without attention sinks, and reuse of
                   a chunk's cache in a different context

Run:  python3 analysis.py      (writes charts/chart1..5.png and results.json)
Everything except the "timing" block of results.json is deterministic.
"""
import json, os, time, platform, statistics
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, DynamicCache

SEED = 20261022
THREADS = 6
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(THREADS)
torch.use_deterministic_algorithms(True)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CH = os.path.join(HERE, "charts")
os.makedirs(CH, exist_ok=True)

PYTHIA = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]
MAIN = "EleutherAI/pythia-160m"
BERT = "bert-base-uncased"

PAIRS = json.load(open(os.path.join(DATA, "ambiguity_pairs.json")))
PASSAGES = [p["text"] for p in json.load(open(os.path.join(DATA, "passages.json")))["passages"]]
CONFIGS = json.load(open(os.path.join(DATA, "model_configs.json")))["models"]

R = {"meta": {"seed": SEED, "threads": THREADS, "torch": torch.__version__,
              "transformers": __import__("transformers").__version__, "dtype": "float32 unless stated"}}
TIMING = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()}

_cache = {}


def load_lm(name, eager=False, dtype=torch.float32):
    key = (name, eager, dtype)
    if key not in _cache:
        m = AutoModelForCausalLM.from_pretrained(
            name, attn_implementation="eager" if eager else "sdpa", dtype=dtype).eval()
        _cache[key] = m
    return _cache[key]


tok = AutoTokenizer.from_pretrained(MAIN)  # all Pythia sizes share one tokenizer


def enc(text):
    return tok(text, return_tensors="pt").input_ids


def r6(x):
    return float(f"{float(x):.6g}")


# ---------------------------------------------------------------- E1 exactness
def greedy_cached(m, ids, n):
    out = m(ids, use_cache=True)
    cache, logits = out.past_key_values, [out.logits[0, -1]]
    seq = ids.clone()
    for _ in range(n):
        nxt = logits[-1].argmax().view(1, 1)
        seq = torch.cat([seq, nxt], 1)
        if len(logits) == n:
            break
        out = m(nxt, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits.append(out.logits[0, -1])
    return seq, torch.stack(logits)


def greedy_uncached(m, ids, n):
    seq, logits = ids.clone(), []
    for _ in range(n):
        lg = m(seq, use_cache=False).logits[0, -1]
        logits.append(lg)
        seq = torch.cat([seq, lg.argmax().view(1, 1)], 1)
    return seq, torch.stack(logits)


def e1():
    N_PROMPT, N_NEW = 32, 64
    prompts = [enc(p)[:, :N_PROMPT] for p in PASSAGES]
    res = {"n_prompts": len(prompts), "prompt_tokens": N_PROMPT, "new_tokens": N_NEW, "models": {}}
    runs = [(n, torch.float32) for n in PYTHIA] + [(MAIN, torch.float64)]
    for name, dt in runs:
        m = load_lm(name, dtype=dt)
        same, maxdiff, argmax_agree, total, divs = 0, 0.0, 0, 0, []
        scale = []
        with torch.no_grad():
            for ids in prompts:
                sc, lc = greedy_cached(m, ids, N_NEW)
                su, lu = greedy_uncached(m, ids, N_NEW)
                same += int(torch.equal(sc, su))
                if not torch.equal(sc, su):
                    step = int((sc[0] != su[0]).nonzero()[0]) - N_PROMPT
                    top2 = lc[step].topk(2).values
                    divs.append({"prompt": len(scale), "first_divergent_step": step,
                                 "top1_minus_top2_logit_cached": r6(top2[0] - top2[1]),
                                 "abs_logit_diff_at_step": r6((lc[step] - lu[step]).abs().max())})
                # compare logits on identical inputs: recompute the cached trajectory in one pass
                full = m(sc, use_cache=False).logits[0, N_PROMPT - 1:N_PROMPT - 1 + N_NEW]
                d = (full - lc).abs().max().item()
                maxdiff = max(maxdiff, d)
                scale.append(full.std().item())
                argmax_agree += int((full.argmax(-1) == lc.argmax(-1)).sum())
                total += N_NEW
        key = name.split("/")[-1] + ("-fp64" if dt == torch.float64 else "")
        res["models"][key] = {"identical_sequences": same, "max_abs_logit_diff": r6(maxdiff),
                              "median_logit_std": r6(statistics.median(scale)),
                              "argmax_agreement": r6(argmax_agree / total), "divergences": divs}
    R["E1_exactness"] = res


# ---------------------------------------------------------------- E2 immutability
def lm_states(m, ids):
    out = m(ids, use_cache=True, output_hidden_states=True)
    c = out.past_key_values
    K = [c.layers[i].keys[0] for i in range(len(c.layers))]   # [heads, T, d] post-RoPE
    V = [c.layers[i].values[0] for i in range(len(c.layers))]
    H = [h[0] for h in out.hidden_states]                      # [T, hidden]
    return K, V, H


def e2():
    items = [("The " + p["word"], p["cont_a"], p["cont_b"]) for p in PAIRS["word_sense"]]
    items += [(g["prefix"], g["cont_a"], g["cont_b"]) for g in PAIRS["garden_path"]]
    res = {"n_items": len(items), "pythia": {}, "bert": {}}
    with torch.no_grad():
        for name, dt in [(MAIN, torch.float32), ("EleutherAI/pythia-410m", torch.float32), (MAIN, torch.float64)]:
            m = load_lm(name, dtype=dt)
            mk = mv = mh = mab = kmax = 0.0
            bit_identical = bit_ab = 0
            for pre, a, b in items:
                ip, ia, ib = enc(pre), enc(pre + a), enc(pre + b)
                n = ip.shape[1]
                assert torch.equal(ia[:, :n], ip) and torch.equal(ib[:, :n], ip)
                Kp, Vp, Hp = lm_states(m, ip)
                Ka, Va, Ha = lm_states(m, ia)
                Kb, Vb, Hb = lm_states(m, ib)
                ok = True; okab = True
                for l in range(len(Kp)):
                    okab &= torch.equal(Ka[l][:, :n], Kb[l][:, :n]) and torch.equal(Va[l][:, :n], Vb[l][:, :n])
                    kmax = max(kmax, Kp[l].abs().max().item(), Vp[l].abs().max().item())
                    for X, Y in ((Kp[l], Ka[l][:, :n]), (Vp[l], Va[l][:, :n])):
                        ok &= torch.equal(X, Y)
                    mk = max(mk, (Kp[l] - Ka[l][:, :n]).abs().max().item(), (Kp[l] - Kb[l][:, :n]).abs().max().item())
                    mv = max(mv, (Vp[l] - Va[l][:, :n]).abs().max().item(), (Vp[l] - Vb[l][:, :n]).abs().max().item())
                    mab = max(mab, (Ka[l][:, :n] - Kb[l][:, :n]).abs().max().item(), (Va[l][:, :n] - Vb[l][:, :n]).abs().max().item())
                for l in range(len(Hp)):
                    mh = max(mh, (Hp[l] - Ha[l][:n]).abs().max().item(), (Hp[l] - Hb[l][:n]).abs().max().item())
                bit_identical += int(ok); bit_ab += int(okab)
            res["pythia"][name.split("/")[-1] + ("-fp64" if dt == torch.float64 else "")] = {
                "max_abs_KV_value_prefix": r6(kmax),
                "items_with_bit_identical_KV_cont_a_vs_cont_b": bit_ab,
                "max_abs_diff_K_prefix_alone_vs_extended": r6(mk),
                "max_abs_diff_V_prefix_alone_vs_extended": r6(mv),
                "max_abs_diff_hidden_prefix_alone_vs_extended": r6(mh),
                "max_abs_diff_KV_cont_a_vs_cont_b": r6(mab),
                "items_with_bit_identical_KV": bit_identical}
        # Pythia: cosine distance of the ambiguous word state across A/B contexts, per layer
        m = load_lm(MAIN)
        cos_py = []
        for pre, a, b in items[:len(PAIRS["word_sense"])]:
            n = enc(pre).shape[1]
            _, _, Ha = lm_states(m, enc(pre + a))
            _, _, Hb = lm_states(m, enc(pre + b))
            cos_py.append([1 - F.cosine_similarity(Ha[l][n - 1].double(), Hb[l][n - 1].double(), 0).item() for l in range(len(Ha))])
        # BERT: same prefix tokens, bidirectional attention
        bt = AutoTokenizer.from_pretrained(BERT)
        bm = AutoModel.from_pretrained(BERT, attn_implementation="eager").eval()
        cos_b, keyrel_b, cos_b_alone = [], [], []
        for p in PAIRS["word_sense"]:
            pre = "The " + p["word"]
            xa, xb, xp = (bt(pre + c, return_tensors="pt") for c in (p["cont_a"], p["cont_b"], ""))
            pos = bt.convert_ids_to_tokens(xa.input_ids[0]).index(p["word"])
            ha = bm(**xa, output_hidden_states=True).hidden_states
            hb = bm(**xb, output_hidden_states=True).hidden_states
            hp = bm(**xp, output_hidden_states=True).hidden_states
            cos_b.append([1 - F.cosine_similarity(ha[l][0, pos], hb[l][0, pos], 0).item() for l in range(len(ha))])
            cos_b_alone.append([1 - F.cosine_similarity(ha[l][0, pos], hp[l][0, pos], 0).item() for l in range(len(ha))])
            # keys each BERT layer would cache for the word: W_k h_{l-1}
            kr = []
            for l, layer in enumerate(bm.encoder.layer):
                ka = layer.attention.self.key(ha[l][0, pos]); kb = layer.attention.self.key(hb[l][0, pos])
                kr.append(((ka - kb).norm() / ka.norm()).item())
            keyrel_b.append(kr)
        res["bert"] = {
            "cosine_distance_word_a_vs_b_by_layer": [r6(x) for x in np.mean(cos_b, 0)],
            "cosine_distance_word_extended_vs_alone_by_layer": [r6(x) for x in np.mean(cos_b_alone, 0)],
            "relative_key_change_a_vs_b_by_layer": [r6(x) for x in np.mean(keyrel_b, 0)],
            "n_words": len(cos_b)}
        res["pythia_cosine_distance_word_a_vs_b_by_layer"] = [r6(x) for x in np.mean(cos_py, 0)]
    R["E2_immutability"] = res
    return bt, bm


# ---------------------------------------------------------------- E3 agility
def tok_pos(ids, tid, last=False):
    hits = (ids[0] == tid).nonzero().flatten().tolist()
    return hits[-1] if last else hits[0]


def e3(bt, bm):
    m = load_lm(MAIN, eager=True)
    L = m.config.num_hidden_layers + 1
    acc_old, acc_new, acc_bert = np.zeros(L), np.zeros(L), np.zeros(L)
    marg_new, marg_bert = np.zeros(L), np.zeros(L)
    att_old, att_cont, att_sink = np.zeros(L - 1), np.zeros(L - 1), np.zeros(L - 1)
    n = 0
    with torch.no_grad():
        for p in PAIRS["word_sense"]:
            w = p["word"]; tid = tok(" " + w).input_ids[0]
            # sense centroids from unambiguous reference sentences, every layer
            cent = {}
            for s in ("a", "b"):
                hs = []
                for ref in p["refs_" + s]:
                    ids = enc(ref)
                    h = m(ids, output_hidden_states=True).hidden_states
                    hs.append(torch.stack([x[0, tok_pos(ids, tid)] for x in h]))
                cent[s] = torch.stack(hs).mean(0)
            bcent = {}
            for s in ("a", "b"):
                hs = []
                for ref in p["refs_" + s]:
                    x = bt(ref, return_tensors="pt")
                    pos = bt.convert_ids_to_tokens(x.input_ids[0]).index(w)
                    h = bm(**x, output_hidden_states=True).hidden_states
                    hs.append(torch.stack([y[0, pos] for y in h]))
                bcent[s] = torch.stack(hs).mean(0)
            for s, other in (("a", "b"), ("b", "a")):
                text = "The " + w + p["cont_" + s] + " The " + w
                ids = enc(text)
                out = m(ids, output_hidden_states=True, output_attentions=True)
                old, new = tok_pos(ids, tid), tok_pos(ids, tid, last=True)
                n_pre = enc("The " + w + p["cont_" + s]).shape[1]
                for l in range(L):
                    h = out.hidden_states[l][0]
                    for pos, acc in ((old, acc_old), (new, acc_new)):
                        c1 = F.cosine_similarity(h[pos], cent[s][l], 0)
                        c2 = F.cosine_similarity(h[pos], cent[other][l], 0)
                        acc[l] += 1.0 if c1 > c2 else (0.5 if c1 == c2 else 0.0)
                        if pos == new:
                            marg_new[l] += (c1 - c2).item()
                for l in range(L - 1):
                    a = out.attentions[l][0, :, new].mean(0)  # head-averaged attention row of the new token
                    att_old[l] += a[old].item()
                    att_cont[l] += a[old + 1:n_pre].sum().item()
                    att_sink[l] += a[0].item()
                # BERT: the original position itself, in the full sentence
                xb = bt("The " + w + p["cont_" + s], return_tensors="pt")
                pos = bt.convert_ids_to_tokens(xb.input_ids[0]).index(w)
                hb = bm(**xb, output_hidden_states=True).hidden_states
                for l in range(L):
                    c1 = F.cosine_similarity(hb[l][0, pos], bcent[s][l], 0)
                    c2 = F.cosine_similarity(hb[l][0, pos], bcent[other][l], 0)
                    acc_bert[l] += 1.0 if c1 > c2 else (0.5 if c1 == c2 else 0.0); marg_bert[l] += (c1 - c2).item()
                n += 1
    R["E3_agility"] = {
        "n_cases": n, "layers_including_embedding": L,
        "pythia_sense_accuracy_old_position": [r6(x / n) for x in acc_old],
        "pythia_sense_accuracy_new_position": [r6(x / n) for x in acc_new],
        "pythia_margin_new_position": [r6(x / n) for x in marg_new],
        "bert_sense_accuracy_original_position": [r6(x / n) for x in acc_bert],
        "bert_margin_original_position": [r6(x / n) for x in marg_bert],
        "attention_new_to_old_word_by_layer": [r6(x / n) for x in att_old],
        "attention_new_to_disambiguating_span_by_layer": [r6(x / n) for x in att_cont],
        "attention_new_to_first_token_by_layer": [r6(x / n) for x in att_sink]}


# ---------------------------------------------------------------- E4 cost
def flops_per_new_token(n_params_nonembed, layers, d_model, ctx, cached):
    # forward FLOPs ~ 2 * params per token processed + attention score/value terms 4*layers*d*ctx per query
    if cached:
        return 2 * n_params_nonembed + 4 * layers * d_model * ctx
    return ctx * 2 * n_params_nonembed + 4 * layers * d_model * ctx * (ctx + 1) / 2


def e4():
    m = load_lm(MAIN)
    cfg = m.config
    nonembed = sum(p.numel() for n_, p in m.named_parameters() if "embed" not in n_)
    lens = [128, 256, 512, 1024, 2000]
    base = torch.cat([enc(" ".join(PASSAGES))[:, :2048]], 1)
    t_cache, t_nocache = {}, {}
    REP = 9
    with torch.no_grad():
        for Lc in lens:
            ids = base[:, :Lc]
            nxt = base[:, Lc:Lc + 1]
            m(ids[:, :16])  # warm-up
            ts = []
            for _ in range(REP):
                t0 = time.perf_counter(); m(torch.cat([ids, nxt], 1), use_cache=False); ts.append(time.perf_counter() - t0)
            t_nocache[Lc] = statistics.median(ts)
            cache = m(ids, use_cache=True).past_key_values
            ts = []
            for _ in range(REP):
                t0 = time.perf_counter(); m(nxt, past_key_values=cache, use_cache=True); ts.append(time.perf_counter() - t0)
                cache.crop(-1)
            t_cache[Lc] = statistics.median(ts)
    TIMING["E4_decode_ms_per_token"] = {
        "model": MAIN, "repeats": REP, "statistic": "median",
        "context_tokens": lens,
        "with_cache_ms": [round(t_cache[l] * 1000, 3) for l in lens],
        "without_cache_ms": [round(t_nocache[l] * 1000, 3) for l in lens],
        "speedup": [round(t_nocache[l] / t_cache[l], 2) for l in lens]}
    fl = {"context_tokens": lens, "nonembedding_params": nonembed,
          "gflops_with_cache": [r6(flops_per_new_token(nonembed, cfg.num_hidden_layers, cfg.hidden_size, l, True) / 1e9) for l in lens],
          "gflops_without_cache": [r6(flops_per_new_token(nonembed, cfg.num_hidden_layers, cfg.hidden_size, l + 1, False) / 1e9) for l in lens]}
    fl["ratio"] = [r6(b / a) for a, b in zip(fl["gflops_with_cache"], fl["gflops_without_cache"])]
    # bytes moved per decode step (fp16 weights + cache), the memory-bandwidth view
    kvb = 2 * cfg.num_hidden_layers * cfg.num_attention_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2
    fl["kv_bytes_per_token_fp16"] = kvb
    R["E4_cost_analytical"] = fl
    # KV memory formula for public configs
    mem = []
    for c in CONFIGS:
        if c["attention"] == "MLA":
            elems = c["layers"] * (c["kv_lora_rank"] + c["qk_rope_head_dim"])
        else:
            elems = 2 * c["layers"] * c["kv_heads"] * c["head_dim"]
        bpt = elems * 2  # 16-bit cache
        mem.append({"name": c["name"], "attention": c["attention"], "kv_bytes_per_token_bf16": bpt,
                    "kib_per_token": r6(bpt / 1024),
                    "gib_per_32k_sequence": r6(bpt * 32768 / 2 ** 30),
                    "gib_per_128k_sequence": r6(bpt * 131072 / 2 ** 30)})
    R["E4_kv_memory"] = mem


# ---------------------------------------------------------------- E5 approximation
def qdq(x, bits):
    # naive symmetric round-to-nearest, one absmax scale per token and head (over head_dim), applied to
    # keys and values alike; no outlier handling, no full-precision residual window (KIVI-style methods
    # quantise keys per channel and keep recent tokens in full precision, so they do better than this).
    # bits=2 gives levels {-1, 0, +1} x scale in practice.
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    return (x / s).round().clamp(-qmax - 1, qmax) * s


def run_variant(m, ids, prefill, variant):
    """Teacher-forced decoding of ids[prefill:] one token at a time with a modified cache.
    Returns log-probs [T, V] for predictions of tokens prefill+1 .. end."""
    out = m(ids[:, :prefill], use_cache=True)
    cache = out.past_key_values
    logps = [F.log_softmax(out.logits[0, -1], -1)]

    def transform(all_entries):
        for layer in cache.layers:
            if variant in ("fp16", "bf16"):
                dt = torch.float16 if variant == "fp16" else torch.bfloat16
                if all_entries:
                    layer.keys = layer.keys.to(dt).float(); layer.values = layer.values.to(dt).float()
                else:
                    layer.keys[:, :, -1:] = layer.keys[:, :, -1:].to(dt).float()
                    layer.values[:, :, -1:] = layer.values[:, :, -1:].to(dt).float()
            elif variant.startswith("int"):
                b = int(variant[3:])
                if all_entries:
                    layer.keys = qdq(layer.keys, b); layer.values = qdq(layer.values, b)
                else:
                    layer.keys[:, :, -1:] = qdq(layer.keys[:, :, -1:], b)
                    layer.values[:, :, -1:] = qdq(layer.values[:, :, -1:], b)
            elif variant.startswith("window"):
                _, budget, sinks = variant.split("_")
                budget, sinks = int(budget), int(sinks)
                T = layer.keys.shape[2]
                if T > budget:
                    keep = list(range(sinks)) + list(range(T - (budget - sinks), T))
                    idx = torch.tensor(keep)
                    layer.keys = layer.keys[:, :, idx]; layer.values = layer.values[:, :, idx]

    if variant != "exact":
        transform(True)
    T = ids.shape[1]
    for t in range(prefill, T - 1):
        o = m(ids[:, t:t + 1], past_key_values=cache, position_ids=torch.tensor([[t]]), use_cache=True)
        cache = o.past_key_values
        if variant != "exact":
            transform(variant.startswith("window"))
        logps.append(F.log_softmax(o.logits[0, -1], -1))
    return torch.stack(logps)


def e5():
    m = load_lm(MAIN)
    PRE, TOT = 256, 512
    seqs = [enc(p)[:, :TOT] for p in PASSAGES]
    seqs = [s for s in seqs if s.shape[1] == TOT]
    variants = ["exact", "fp16", "bf16", "int8", "int4", "int2", "window_128_4", "window_128_0", "window_64_4", "window_64_0"]
    agg = {v: {"kl": [], "top1": [], "nll": []} for v in variants}
    full_check = 0.0
    with torch.no_grad():
        for ids in seqs:
            ref_full = F.log_softmax(m(ids, use_cache=False).logits[0, PRE - 1:TOT - 1], -1)
            targets = ids[0, PRE:TOT]
            lp_exact = None
            for v in variants:
                lp = run_variant(m, ids, PRE, v)
                if v == "exact":
                    lp_exact = lp
                    full_check = max(full_check, (lp - ref_full).abs().max().item())
                kl = (lp_exact.exp() * (lp_exact - lp)).sum(-1)
                agg[v]["kl"].append(kl.mean().item())
                agg[v]["top1"].append((lp.argmax(-1) == lp_exact.argmax(-1)).float().mean().item())
                agg[v]["nll"].append(-lp.gather(1, targets[:, None]).mean().item())
        # cross-context chunk reuse: S (128) + C (128) then 64 teacher-forced tokens
        reuse = {"exact": [], "chunk_cache_wrong_positions": [], "chunk_cache_reindexed": []}
        reuse_kl = {k: [] for k in reuse}; reuse_top1 = {k: [] for k in reuse}
        for ids in seqs:
            S, C, E = 128, 128, 64
            ids = ids[:, :S + C + E]
            lp_ref = None
            for k in reuse:
                cs = m(ids[:, :S], use_cache=True).past_key_values
                if k == "exact":
                    cache = m(ids[:, S:S + C], past_key_values=cs, position_ids=torch.arange(S, S + C)[None], use_cache=True).past_key_values
                else:
                    pos = torch.arange(0, C) if k == "chunk_cache_wrong_positions" else torch.arange(S, S + C)
                    cc = m(ids[:, S:S + C], position_ids=pos[None], use_cache=True).past_key_values
                    for i, layer in enumerate(cs.layers):
                        layer.keys = torch.cat([layer.keys, cc.layers[i].keys], 2)
                        layer.values = torch.cat([layer.values, cc.layers[i].values], 2)
                    cache = cs
                lps = []
                for t in range(S + C, S + C + E):
                    o = m(ids[:, t:t + 1], past_key_values=cache, position_ids=torch.tensor([[t]]), use_cache=True)
                    cache = o.past_key_values
                    lps.append(F.log_softmax(o.logits[0, -1], -1))
                lp = torch.stack(lps)
                if k == "exact":
                    lp_ref = lp
                reuse_kl[k].append((lp_ref.exp() * (lp_ref - lp)).sum(-1).mean().item())
                reuse_top1[k].append((lp.argmax(-1) == lp_ref.argmax(-1)).float().mean().item())
    R["E5_approximation"] = {
        "n_passages": len(seqs), "prefill_tokens": PRE, "decoded_tokens": TOT - PRE,
        "exact_cache_vs_full_forward_max_abs_logprob_diff": r6(full_check),
        "variants": {v: {"mean_kl_vs_exact": r6(np.mean(agg[v]["kl"])),
                         "top1_agreement_vs_exact": r6(np.mean(agg[v]["top1"])),
                         "perplexity": r6(np.exp(np.mean(agg[v]["nll"])))} for v in variants},
        "chunk_reuse": {k: {"mean_kl_vs_exact": r6(np.mean(reuse_kl[k])),
                            "top1_agreement_vs_exact": r6(np.mean(reuse_top1[k]))} for k in reuse},
        "chunk_reuse_setup": "128-token context S, 128-token chunk C, then 64 teacher-forced tokens; "
                             "C's cache computed alone (positions 0..127, or re-indexed to 128..255) and appended to S's cache"}


# ---------------------------------------------------------------- charts
GREY = {"k": "#111111", "d": "#555555", "m": "#bbbbbb", "l": "#dddddd"}


def style(ax, title, xl, yl):
    ax.set_title(title, fontsize=13, loc="left", color="#111111")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.8); ax.set_axisbelow(True)


def src(fig, text):
    fig.text(0.01, 0.01, text, fontsize=8, color="#555555")


def charts():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    e2r, e3r = R["E2_immutability"], R["E3_agility"]
    layers = list(range(len(e2r["bert"]["cosine_distance_word_a_vs_b_by_layer"])))
    # chart1: immutability
    fig, ax = plt.subplots(figsize=(10.67, 6), dpi=150)
    ax.plot(layers, e2r["bert"]["cosine_distance_word_a_vs_b_by_layer"], "-o", color=GREY["k"], lw=2, ms=6,
            label="BERT-base (bidirectional)")
    ax.plot(layers, e2r["pythia_cosine_distance_word_a_vs_b_by_layer"], "-s", color=GREY["d"], lw=2, ms=6,
            label="Pythia-160M (causal)")
    pmax = max(abs(x) for x in e2r["pythia_cosine_distance_word_a_vs_b_by_layer"])
    ax.annotate("Pythia: exactly 0 at every layer" if pmax == 0 else f"Pythia: |distance| < {pmax:.0e} at every layer (rounding)",
                (layers[-1], 0), xytext=(6.5, 0.05), color="#555555")
    style(ax, "Does the ambiguous word's state change when the continuation changes its meaning?",
          "Layer (0 = embeddings)", "Cosine distance of word state, sense A vs B context")
    ax.legend(frameon=False, loc="upper left")
    src(fig, f"Source: our analysis, {e2r['bert']['n_words']} ambiguous words ('The bank' + river vs finance continuation). stabilarity/hub research code, results.json E2.")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(CH, "chart1.png"), facecolor="white"); plt.close(fig)
    # chart2: agility
    fig, ax = plt.subplots(figsize=(10.67, 6), dpi=150)
    ax.plot(layers, [x * 100 for x in e3r["pythia_sense_accuracy_new_position"]], "-o", color=GREY["k"], lw=2, ms=6,
            label="Pythia-160M, new position (word repeated after context)")
    ax.plot(layers, [x * 100 for x in e3r["bert_sense_accuracy_original_position"]], "--^", color=GREY["d"], lw=2, ms=6,
            label="BERT-base, original position")
    ax.plot(layers, [x * 100 for x in e3r["pythia_sense_accuracy_old_position"]], ":s", color=GREY["m"], lw=2, ms=6,
            label="Pythia-160M, original cached position")
    ax.axhline(50, color="#bbbbbb", lw=1)
    ax.set_ylim(0, 105)
    style(ax, "Where is the word's sense resolved? Nearest-centroid sense accuracy by layer",
          "Layer (0 = embeddings)", f"Sense accuracy, % of {e3r['n_cases']} cases")
    ax.legend(frameon=False, loc="lower right")
    src(fig, f"Source: our analysis, {e3r['n_cases'] // 2} words x 2 senses; centroids from 3 unambiguous reference sentences per sense. results.json E3.")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(CH, "chart2.png"), facecolor="white"); plt.close(fig)
    # chart3: decode latency
    t = TIMING["E4_decode_ms_per_token"]
    fig, ax = plt.subplots(figsize=(10.67, 6), dpi=150)
    ax.plot(t["context_tokens"], t["without_cache_ms"], "-o", color=GREY["k"], lw=2, ms=7, label="Without cache (recompute prefix)")
    ax.plot(t["context_tokens"], t["with_cache_ms"], "-s", color=GREY["d"], lw=2, ms=7, label="With KV cache")
    for x, y, s in zip(t["context_tokens"], t["without_cache_ms"], t["speedup"]):
        ax.annotate(f"{s}x", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", color="#555555", fontsize=9)
    ax.set_yscale("log")
    style(ax, "Time to produce one new token, Pythia-160M on CPU", "Context length, tokens", "Milliseconds per token (log scale)")
    ax.legend(frameon=False, loc="upper left")
    src(fig, f"Source: our measurement, median of {t['repeats']} runs, fp32, {THREADS} threads; labels show the speed-up. results.json timing block.")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(CH, "chart3.png"), facecolor="white"); plt.close(fig)
    # chart4: KV memory per token
    mem = R["E4_kv_memory"]
    fig, ax = plt.subplots(figsize=(10.67, 6), dpi=150)
    names = [x["name"] for x in mem]; vals = [x["kib_per_token"] for x in mem]
    cols = ["#111111" if x["attention"] == "MHA" else "#555555" if x["attention"] in ("GQA", "MQA") else "#bbbbbb" for x in mem]
    ax.barh(names, vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:g} KiB ({mem[i]['attention']})", va="center", fontsize=9, color="#111111")
    ax.invert_yaxis(); ax.set_xlim(0, max(vals) * 1.35)
    style(ax, "KV-cache memory per token at 16-bit precision", "KiB per token", "")
    ax.grid(axis="x", color="#eeeeee"); ax.grid(axis="y", visible=False)
    src(fig, "Source: our calculation from published configs (data/model_configs.json): 2 x layers x kv_heads x head_dim x 2 bytes; MLA: layers x (512 + 64) x 2 bytes.")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(CH, "chart4.png"), facecolor="white"); plt.close(fig)
    # chart5: approximation
    e5r = R["E5_approximation"]
    labels = {"exact": "Exact cache (fp32)", "fp16": "fp16 KV", "bf16": "bf16 KV", "int8": "int8 KV", "int4": "int4 KV", "int2": "int2 KV",
              "window_128_4": "Window 128, 4 sinks", "window_128_0": "Window 128, no sinks",
              "window_64_4": "Window 64, 4 sinks", "window_64_0": "Window 64, no sinks"}
    vs = list(labels)
    reuse_lab = {"chunk_cache_reindexed": "Reused chunk, positions fixed", "chunk_cache_wrong_positions": "Reused chunk, stale positions"}
    names = [labels[v] for v in vs] + list(reuse_lab.values())
    top = [100 * e5r["variants"][v]["top1_agreement_vs_exact"] for v in vs] + \
          [100 * e5r["chunk_reuse"][k]["top1_agreement_vs_exact"] for k in reuse_lab]
    fig, ax = plt.subplots(figsize=(10.67, 6), dpi=150)
    cols = ["#111111"] + ["#555555"] * 5 + ["#bbbbbb"] * 4 + ["#dddddd"] * 2
    ax.barh(names, top, color=cols, height=0.6, edgecolor="#555555", linewidth=0.5)
    for i, v in enumerate(top):
        ax.text(v, i, f" {v:.2f}%", va="center", fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 112)
    style(ax, "When the cache stops being exact: top-1 agreement with the exact cache", "Next-token top-1 agreement with exact cache, %", "")
    ax.grid(axis="x", color="#eeeeee"); ax.grid(axis="y", visible=False)
    src(fig, f"Source: our analysis, Pythia-160M, {e5r['n_passages']} Pile test passages; 256-token prefill + 256 decoded tokens (reuse: 128 + 128 + 64). results.json E5.")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(CH, "chart5.png"), facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    def step(name, fn, *a):
        t = time.time(); r = fn(*a); print(f"{name} done in {time.time() - t:.0f}s", flush=True); return r
    step("E1", e1); bt, bm = step("E2", e2); step("E3", e3, bt, bm); step("E4", e4); step("E5", e5)
    TIMING["total_seconds"] = round(time.time() - t0, 1)
    R["timing"] = TIMING  # the only non-deterministic block
    charts()
    json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "E4_kv_memory"}, indent=1)[:6000])
