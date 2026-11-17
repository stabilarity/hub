"""KV cache quantization and pruning benchmark on two small open models (CPU only).

Question: what does compressing the key-value (KV) cache of a real model cost in long-context
retrieval (needle in a haystack, passkey style) and in perplexity, per byte of KV memory saved?

Step 1 (network for the weights, about 6.5 + 4 min of CPU, run once per model):
    python analysis.py --run qwen2.5-0.5b
    python analysis.py --run qwen3-0.6b
Step 2 (offline, deterministic):
    python analysis.py          # reads raw/*.json, writes results.json and charts/chart1-5.png

Methods applied to the cached context (everything before the question):
  bf16                          keys/values rounded to bfloat16 (the baseline every byte count is relative to)
  tok-int{8,4,2}                simulated asymmetric min-max quantization, keys AND values per token
                                (groups of 32 channels)
  kivi-int{8,4,2}               KIVI layout: keys per channel (groups of 32 tokens), values per token
  window-{50,25}                keep only the most recent 50% / 25% of the context tokens
  sink-{50,25}                  StreamingLLM: 4 attention-sink tokens + most recent tokens
  h2o-{50,25}                   H2O-style one-shot: half recent window, half heavy hitters by attention
                                accumulated over all context queries (question-agnostic), per KV head
  snapkv-{50,25}                SnapKV-style: top tokens by attention from the question tokens
                                (observation window = the question), max-pooled with kernel 7, per KV head

Quantization is SIMULATED (quantize -> dequantize in fp32): it measures the accuracy cost, not kernel speed.
Model weights and activations run in fp32 on CPU (bf16 matmuls are emulated and ~3x slower on AVX2).
"""
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import struct
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "Source: our measurements, stabilarity/hub/research/kv-cache-quantization-benchmark (Nov 2026)"

MODELS = {
    "qwen2.5-0.5b": dict(repo="Qwen/Qwen2.5-0.5B-Instruct", rev="7ae557604adf67be50417f59c2c2f167def9a775",
                         label="Qwen2.5-0.5B-Instruct", license="Apache-2.0"),
    "qwen3-0.6b": dict(repo="Qwen/Qwen3-0.6B", rev="c1899de289a04d12100db370d81485cdf75e47ca",
                       label="Qwen3-0.6B", license="Apache-2.0"),
}
# Llama-3.2-1B was tried first and was killed by the OOM killer: 1.24 B parameters in fp32 (~5 GB) plus
# the streaming buffer did not fit next to the other services on this 12 GB machine. Qwen3-0.6B replaced it.
ORDER = ["qwen2.5-0.5b", "qwen3-0.6b"]
METHODS = ["bf16", "tok-int8", "tok-int4", "tok-int2", "kivi-int8", "kivi-int4", "kivi-int2",
           "window-50", "window-25", "sink-50", "sink-25", "h2o-50", "h2o-25", "snapkv-50", "snapkv-25"]
LENGTHS = [1024, 2048]          # total prompt tokens
DEPTHS = [0.05, 0.25, 0.5, 0.75, 0.95]
TRIALS = 2                      # passkeys per (length, depth) cell (default)
TRIALS_BY_MODEL = {"qwen2.5-0.5b": 2, "qwen3-0.6b": 1}   # compute budget: 1 passkey per cell for Qwen3
GROUP = 32                      # quantization group size (channels for per-token, tokens for per-channel)
SINKS = 4
POOL = 7
PPL_PREFIX, PPL_EVAL = 1024, 256
SEED = 0


def kv_fraction(method):
    """Bytes of the compressed context cache relative to BF16 (scale+zero stored as fp16 per group of 32)."""
    if method == "bf16":
        return 1.0
    if "int" in method:
        b = int(method.split("int")[1])
        return (b + 32 / GROUP) / 16          # b bits + (16+16 bits)/32 values of overhead
    return int(method.split("-")[1]) / 100


# =============================================================== experiment (needs torch)
def run(key):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import random
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.initialization import no_init_weights
    from huggingface_hub import snapshot_download
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

    def rss_mb():
        with open("/proc/self/status") as f:
            return {l.split(":")[0]: int(l.split()[1]) / 1024 for l in f if l.startswith(("VmRSS", "VmHWM"))}

    def hub_meta(repo, rev):
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true",
                                    timeout=60) as r:
            d = json.load(r)
        st = [s for s in d["siblings"] if s["rfilename"].endswith(".safetensors")]
        assert len(st) == 1, st
        return st[0]

    def stream_weights(model, repo, rev, fname, sha256):
        """Stream the safetensors file into the parameters in one pass (nothing written to disk)."""
        params = dict(model.named_parameters())
        h = hashlib.sha256()
        n_loaded = 0
        with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{rev}/{fname}", timeout=120) as r:
            def read_exact(n):
                buf = bytearray(n)
                mv, got = memoryview(buf), 0
                while got < n:
                    k = r.readinto(mv[got:])
                    if not k:
                        raise IOError("stream ended early")
                    got += k
                h.update(buf)
                return buf
            header = json.loads(bytes(read_exact(struct.unpack("<Q", read_exact(8))[0])))
            header.pop("__metadata__", None)
            pos = 0
            for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
                s, e = info["data_offsets"]
                if s > pos:
                    read_exact(s - pos)
                raw = read_exact(e - s)
                pos = e
                if name in params:
                    t = torch.frombuffer(raw, dtype=DT[info["dtype"]]).reshape(info["shape"])
                    with torch.no_grad():
                        params[name].data.copy_(t.to(params[name].dtype))
                    n_loaded += 1
            h.update(r.read())
        assert h.hexdigest() == sha256, "SHA-256 mismatch"
        return n_loaded

    m = MODELS[key]
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    torch.set_num_threads(6)
    t_start = time.time()
    tmp = tempfile.mkdtemp(prefix="hfcfg-")
    try:
        st = hub_meta(m["repo"], m["rev"])
        local = snapshot_download(m["repo"], revision=m["rev"], cache_dir=tmp,
                                  allow_patterns=["*.json", "tokenizer.model"])
        cfg = AutoConfig.from_pretrained(local)
        tok = AutoTokenizer.from_pretrained(local)
    finally:
        pass
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    n_loaded = stream_weights(model, m["repo"], m["rev"], st["rfilename"], st["lfs"]["sha256"])
    if getattr(cfg, "tie_word_embeddings", False):
        model.lm_head.weight = model.model.embed_tokens.weight
    shutil.rmtree(tmp, ignore_errors=True)
    t_load = time.time() - t_start
    L_ = cfg.num_hidden_layers
    Hkv, Hq = cfg.num_key_value_heads, cfg.num_attention_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq
    grp = Hq // Hkv
    attn_mods = [layer.self_attn for layer in model.model.layers]

    # ------------------------------------------------ attention-score hooks (H2O and SnapKV scoring)
    HOOK = {"mode": None, "cache": None, "acc": {}}

    def qk(mod, hs, pe):
        B, T, _ = hs.shape
        q = mod.q_proj(hs).view(B, T, -1, D).transpose(1, 2)
        k = mod.k_proj(hs).view(B, T, -1, D).transpose(1, 2)
        if hasattr(mod, "q_norm"):
            q, k = mod.q_norm(q), mod.k_norm(k)
        cos, sin = pe
        return apply_rotary_pos_emb(q, k, cos, sin)

    def pre_hook(mod, args, kwargs):
        if HOOK["mode"] is None:
            return
        hs = kwargs.get("hidden_states", args[0] if args else None)
        q, k_new = qk(mod, hs, kwargs["position_embeddings"])
        q = q[0]                                         # [Hq, Tq, D]
        if HOOK["mode"] == "h2o":                        # context prefill from position 0: causal self-attention
            k = k_new[0].repeat_interleave(grp, 0)       # [Hq, T, D]
            T = k.shape[1]
            acc = torch.zeros(Hq, T)
            for s0 in range(0, T, 256):
                sc = (q[:, s0:s0 + 256] @ k.transpose(1, 2)) * mod.scaling
                qi = torch.arange(s0, min(s0 + 256, T)).unsqueeze(1)
                sc = sc.masked_fill(torch.arange(T).unsqueeze(0) > qi, float("-inf"))
                acc += sc.softmax(-1).sum(1)
        else:                                            # snapkv: question queries over the cached context
            k = HOOK["cache"].layers[mod.layer_idx].keys[0].repeat_interleave(grp, 0)
            acc = ((q @ k.transpose(1, 2)) * mod.scaling).softmax(-1).sum(1)
        HOOK["acc"][mod.layer_idx] = acc.view(Hkv, grp, -1).sum(1)   # [Hkv, T]

    for mod in attn_mods:
        mod.register_forward_pre_hook(pre_hook, with_kwargs=True)

    # ------------------------------------------------ cache compression
    def fq(x, bits, dim):
        """Simulated asymmetric min-max quantization in groups of GROUP along `dim` (2=tokens, 3=channels)."""
        B, H, T, Dd = x.shape
        if dim == 3:
            g = x.reshape(B, H, T, Dd // GROUP, GROUP)
            red = -1
        else:
            pad = (-T) % GROUP
            xp = torch.cat([x, x[:, :, -1:].expand(B, H, pad, Dd)], 2) if pad else x
            g = xp.reshape(B, H, (T + pad) // GROUP, GROUP, Dd)
            red = 3
        mn, mx = g.amin(red, keepdim=True), g.amax(red, keepdim=True)
        qmax = 2 ** bits - 1
        scale = ((mx - mn) / qmax).clamp_min(1e-8).half().float()
        zero = mn.half().float()
        deq = ((g - zero) / scale).round().clamp(0, qmax) * scale + zero
        if dim == 3:
            return deq.reshape(B, H, T, Dd)
        return deq.reshape(B, H, -1, Dd)[:, :, :T]

    def clone(cache):
        from transformers import DynamicCache
        c = DynamicCache(config=cfg)
        for i, lay in enumerate(cache.layers):
            c.update(lay.keys.clone(), lay.values.clone(), i)
        return c

    def keep_idx(method, T, scores, needle_pos=None):
        """Per-layer [Hkv, B] sorted indices of context tokens to keep."""
        frac = int(method.split("-")[1]) / 100
        Bk = int(round(T * frac))
        out = []
        for li in range(L_):
            if method.startswith("window"):
                idx = torch.arange(T - Bk, T).expand(Hkv, Bk)
            elif method.startswith("sink"):
                idx = torch.cat([torch.arange(SINKS), torch.arange(T - Bk + SINKS, T)]).expand(Hkv, Bk)
            elif method.startswith("h2o"):
                rec = Bk // 2
                sc = scores["h2o"][li][:, :T - rec].clone()
                hh = sc.topk(Bk - rec, -1).indices
                idx = torch.cat([hh, torch.arange(T - rec, T).expand(Hkv, rec)], 1)
            else:  # snapkv
                sc = F.max_pool1d(scores["snapkv"][li].unsqueeze(0), POOL, 1, POOL // 2).squeeze(0)
                idx = sc[:, :T].topk(Bk, -1).indices
            out.append(idx.sort(-1).values)
        return out

    def compress(base, method, scores=None):
        """Return (new cache, needle retention or None)."""
        c = clone(base)
        T = c.layers[0].keys.shape[2]
        if method == "bf16" or "int" in method:
            for lay in c.layers:
                if method == "bf16":
                    lay.keys, lay.values = lay.keys.bfloat16().float(), lay.values.bfloat16().float()
                else:
                    b = int(method.split("int")[1])
                    kd = 2 if method.startswith("kivi") else 3
                    lay.keys, lay.values = fq(lay.keys, b, kd), fq(lay.values, b, 3)
            return c, None
        idxs = keep_idx(method, T, scores)
        for lay, idx in zip(c.layers, idxs):
            g = idx[None, :, :, None].expand(1, Hkv, idx.shape[1], D)
            lay.keys = lay.keys.gather(2, g).bfloat16().float()
            lay.values = lay.values.gather(2, g).bfloat16().float()
        return c, idxs

    @torch.no_grad()
    def fwd_on(cache, ids, start_pos, logits=True):
        n = len(ids)
        clen = cache.layers[0].keys.shape[2]
        out = model(torch.tensor([ids]), past_key_values=cache, use_cache=True,
                    position_ids=torch.arange(start_pos, start_pos + n).unsqueeze(0),
                    cache_position=torch.arange(clen, clen + n))
        return out.logits[0].float()

    # ------------------------------------------------ prompt construction
    hs_rows = json.load(open(os.path.join(HERE, "data", "hellaswag_val_first100.json")))["rows"]
    sentences = []
    for r in hs_rows:
        for s in (r["row"]["ctx"]).replace("  ", " ").split(". "):
            s = s.strip().rstrip(".")
            if len(s) > 15:
                sentences.append(s + ". ")
    sent_ids = [tok(s, add_special_tokens=False)["input_ids"] for s in sentences]
    MARK = "@@HAYSTACK@@"
    intro = "There is a pass key hidden inside a lot of irrelevant text. Find it and memorize it."
    question = "What is the pass key? Answer with the number only."
    chat = tok.apply_chat_template([{"role": "user", "content": f"{intro}\n\n{MARK}\n\n{question}"}],
                                   add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)   # Qwen3: empty think block; ignored elsewhere
    pre_s, post_s = chat.split(MARK)
    post_s = post_s + "The pass key is"
    pre_ids = tok(pre_s, add_special_tokens=False)["input_ids"]
    post_ids = tok(post_s, add_special_tokens=False)["input_ids"]

    def build(total, depth, passkey, offset):
        needle = f"The pass key is {passkey}. Remember it. {passkey} is the pass key. "
        n_ids = tok(needle, add_special_tokens=False)["input_ids"]
        budget = total - len(pre_ids) - len(post_ids) - len(n_ids)
        hay, j = [], offset
        while sum(len(x) for x in hay) < budget:
            hay.append(sent_ids[j % len(sent_ids)])
            j += 1
        flat = [t for x in hay for t in x][:budget]
        # insert the needle at the sentence boundary closest to the requested depth
        bounds, c = [0], 0
        for x in hay:
            c += len(x)
            if c <= budget:
                bounds.append(c)
        ins = min(bounds, key=lambda b: abs(b - depth * budget))
        ctx = pre_ids + flat[:ins] + n_ids + flat[ins:]
        npos = list(range(len(pre_ids) + ins, len(pre_ids) + ins + len(n_ids)))
        whole = tok(post_s + " " + str(passkey), add_special_tokens=False)["input_ids"]
        assert whole[:len(post_ids)] == post_ids
        ans = whole[len(post_ids):]
        return ctx, npos, ans

    @torch.no_grad()
    def prefill(ids, h2o=False):
        HOOK["mode"], HOOK["acc"] = ("h2o" if h2o else None), {}
        out = model(torch.tensor([ids]), use_cache=True, logits_to_keep=1)
        HOOK["mode"] = None
        return out.past_key_values, {li: a.clone() for li, a in HOOK["acc"].items()}

    def answer_score(cache, T, ans):
        lg = fwd_on(cache, post_ids + ans[:-1], T)
        lp = torch.log_softmax(lg[len(post_ids) - 1:], -1)
        tgt = torch.tensor(ans)
        pred = lp.argmax(-1)
        return dict(exact=int(bool((pred == tgt).all())),
                    logprob=round(lp[torch.arange(len(ans)), tgt].sum().item(), 4),
                    pred=tok.decode(pred.tolist()).strip())

    # ------------------------------------------------ correctness checks
    checks = {}
    ctx, npos, ans = build(LENGTHS[0], 0.5, 12345, 0)
    base, sc = prefill(ctx)
    lg_cached = fwd_on(clone(base), post_ids + ans[:-1], len(ctx))
    with torch.no_grad():
        lg_full = model(torch.tensor([ctx + post_ids + ans[:-1]])).logits[0, len(ctx):].float()
    checks["cached_vs_full_forward_max_abs_logit_diff"] = round((lg_cached - lg_full).abs().max().item(), 5)
    # gather path with every token kept must equal the baseline
    c_all = clone(base)
    T = len(ctx)
    for lay in c_all.layers:
        idx = torch.arange(T).expand(Hkv, T)
        g = idx[None, :, :, None].expand(1, Hkv, T, D)
        lay.keys, lay.values = lay.keys.gather(2, g), lay.values.gather(2, g)
    checks["gather_all_vs_baseline_max_abs_logit_diff"] = round(
        (fwd_on(c_all, post_ids + ans[:-1], T) - lg_cached).abs().max().item(), 6)
    del base, c_all
    gc.collect()

    # ------------------------------------------------ NIAH grid
    niah = []
    stats = None
    t0 = time.time()
    for total in LENGTHS:
        for depth in DEPTHS:
            for trial in range(TRIALS_BY_MODEL.get(key, TRIALS)):
                pk = rng.randint(10000, 99999)
                ctx, npos, ans = build(total, depth, pk, rng.randint(0, len(sent_ids) - 1))
                T = len(ctx)
                base, h2o_sc = prefill(ctx, h2o=True)
                HOOK["mode"], HOOK["cache"], HOOK["acc"] = "snapkv", base, {}
                fwd_on(clone(base), post_ids, T)
                HOOK["mode"] = None
                scores = dict(h2o=h2o_sc, snapkv={li: a.clone() for li, a in HOOK["acc"].items()})
                if stats is None and total == LENGTHS[-1]:
                    stats = kv_stats(base, fq, torch)
                row = dict(length=total, ctx_tokens=T, depth=depth, trial=trial, passkey=pk,
                           needle_tokens=[npos[0], npos[-1]], answer_tokens=len(ans), methods={})
                for meth in METHODS:
                    c, idxs = compress(base, meth, scores)
                    r = answer_score(c, T, ans)
                    if idxs is not None:
                        need = torch.tensor(npos)
                        r["needle_kept"] = round(statistics.mean(
                            torch.isin(idx, need).sum(-1).float().mean().item() / len(npos) for idx in idxs), 4)
                    row["methods"][meth] = r
                    del c
                niah.append(row)
                print(key, total, depth, trial, pk, "".join(str(row["methods"][x]["exact"]) for x in METHODS),
                      round(time.time() - t0, 1), flush=True)
                del base, scores
                gc.collect()
    t_niah = time.time() - t0

    # ------------------------------------------------ perplexity with a compressed prefix
    t0 = time.time()
    text = "".join(r["row"]["text"] for r in json.load(open(os.path.join(HERE, "data",
                                                                           "wikitext2_test_first100.json")))["rows"])
    bos = [tok.bos_token_id] if (tok.bos_token_id is not None and "llama" in key) else []
    ids = (bos + tok(text, add_special_tokens=False)["input_ids"])[:PPL_PREFIX + PPL_EVAL]
    pre, ev = ids[:PPL_PREFIX], ids[PPL_PREFIX:]
    base, h2o_sc = prefill(pre, h2o=True)
    # SnapKV on plain text: observation window = the last 32 prefix tokens (as in the SnapKV paper)
    HOOK["mode"], HOOK["acc"] = "snapkv", {}
    obs = 32
    tmpc = clone(base)
    for lay in tmpc.layers:
        lay.keys, lay.values = lay.keys[:, :, :-obs], lay.values[:, :, :-obs]
    HOOK["cache"] = tmpc
    fwd_on(clone(tmpc), pre[-obs:], PPL_PREFIX - obs)
    HOOK["mode"] = None
    snap = {li: torch.cat([a, torch.full((Hkv, obs), 1e9)], 1) for li, a in HOOK["acc"].items()}
    scores = dict(h2o=h2o_sc, snapkv=snap)

    ppl = {}
    for meth in METHODS:
        c, _ = compress(base, meth, scores)
        # predict ev[0] from the last prefix token: drop it from the cache and re-feed it
        for lay in c.layers:
            lay.keys, lay.values = lay.keys[:, :, :-1], lay.values[:, :, :-1]
        lg = fwd_on(c, [pre[-1]] + ev[:-1], PPL_PREFIX - 1)
        nll = F.cross_entropy(lg, torch.tensor(ev)).item()
        ppl[meth] = dict(nll=round(nll, 5), ppl=round(math.exp(nll), 4))
    with torch.no_grad():  # no context at all: the eval chunk alone
        inp0, tgt0 = (bos + ev[:-1], ev) if bos else (ev[:-1], ev[1:])
        nll0 = F.cross_entropy(model(torch.tensor([inp0])).logits[0].float(), torch.tensor(tgt0)).item()
    ppl_info = dict(prefix_tokens=PPL_PREFIX, eval_tokens=PPL_EVAL, no_context_nll=round(nll0, 5),
                    no_context_ppl=round(math.exp(nll0), 4), methods=ppl)
    del base
    t_ppl = time.time() - t0

    # ------------------------------------------------ decode latency vs cache length (window eviction)
    t0 = time.time()
    ctx, _, _ = build(LENGTHS[-1], 0.5, 54321, 7)
    base, _ = prefill(ctx)
    lat = {}
    for meth in ["bf16", "window-50", "window-25"]:
        reps = []
        for rep in range(3):
            c, _ = compress(base, meth)
            nxt, pos = post_ids[-1], len(ctx)
            fwd_on(c, [nxt], pos)  # warm-up step
            ts = []
            for i in range(16):
                t1 = time.perf_counter()
                lg = fwd_on(c, [nxt], pos + 1 + i)
                ts.append(time.perf_counter() - t1)
                nxt = int(lg[-1].argmax())
            reps.append(statistics.median(ts))
        lat[meth] = dict(cache_tokens=int(round(len(ctx) * kv_fraction(meth))),
                         median_step_ms=round(1000 * statistics.median(reps), 2),
                         reps_ms=[round(1000 * x, 2) for x in reps])
    del base
    t_lat = time.time() - t0

    res = dict(key=key, **m, file=st["rfilename"], file_bytes=st["size"], sha256=st["lfs"]["sha256"],
               tensors_loaded=n_loaded, compute_dtype="float32", attn="sdpa",
               layers=L_, kv_heads=Hkv, q_heads=Hq, head_dim=D,
               kv_bytes_per_token_bf16=L_ * Hkv * D * 2 * 2,
               chat_prefix_tokens=len(pre_ids), question_tokens=len(post_ids),
               checks=checks, niah=niah, kv_stats=stats, ppl=ppl_info, decode_latency=lat,
               timing_s=dict(load=round(t_load, 1), niah=round(t_niah, 1), ppl=round(t_ppl, 1),
                             latency=round(t_lat, 1), wall=round(time.time() - t_start, 1)),
               peak_rss_mb=round(rss_mb()["VmHWM"], 1), torch=torch.__version__)
    import transformers
    res["transformers"] = transformers.__version__
    out_dir = os.environ.get("KV_OUT", os.path.join(HERE, "raw"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{key}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(dict(checks=checks, ppl=ppl_info, lat=lat, timing=res["timing_s"])))


def kv_stats(cache, fq, torch):
    """Channel-outlier statistics and quantization error of the full-precision context cache."""
    out = dict(per_layer=[])
    for li, lay in enumerate(cache.layers):
        d = dict(layer=li)
        for nm, x in (("key", lay.keys), ("value", lay.values)):
            ch = x[0].abs().amax(1)                       # [H, D] absmax per channel over tokens
            d[f"{nm}_channel_max_over_median"] = round((ch.amax(-1) / ch.median(-1).values).mean().item(), 3)
            for b in (8, 4, 2):
                for lay_name, dim in (("tok", 3), ("chan", 2)):
                    e = (fq(x, b, dim) - x).norm() / x.norm()
                    d[f"{nm}_relerr_{lay_name}_int{b}"] = round(e.item(), 5)
        out["per_layer"].append(d)
    keys = [k for k in out["per_layer"][0] if k != "layer"]
    out["mean"] = {k: round(sum(p[k] for p in out["per_layer"]) / len(out["per_layer"]), 5) for k in keys}
    # channel profile of keys and values for the middle layer, head 0
    mid = len(cache.layers) // 2
    out["profile_layer"] = mid
    out["key_channel_absmax_mid"] = [round(v, 3) for v in cache.layers[mid].keys[0, 0].abs().amax(0).tolist()]
    out["value_channel_absmax_mid"] = [round(v, 3) for v in cache.layers[mid].values[0, 0].abs().amax(0).tolist()]
    return out


# =============================================================== aggregation and charts
def aggregate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    raw = {k: json.load(open(os.path.join(HERE, "raw", f"{k}.json"))) for k in ORDER
           if os.path.exists(os.path.join(HERE, "raw", f"{k}.json"))}
    models = {}
    for k, r in raw.items():
        per = {}
        for meth in METHODS:
            rows = [(row, row["methods"][meth]) for row in r["niah"]]
            n = len(rows)
            ex = sum(x["exact"] for _, x in rows)
            by_len = {}
            for L in LENGTHS:
                sub = [x for row, x in rows if row["length"] == L]
                by_len[str(L)] = dict(n=len(sub), correct=sum(x["exact"] for x in sub),
                                      acc=round(sum(x["exact"] for x in sub) / len(sub), 4))
            by_cell = {}
            for L in LENGTHS:
                for d in DEPTHS:
                    sub = [x for row, x in rows if row["length"] == L and row["depth"] == d]
                    by_cell[f"{L}@{d}"] = round(sum(x["exact"] for x in sub) / len(sub), 4)
            by_depth = {}
            for d in DEPTHS:
                sub = [x for row, x in rows if row["depth"] == d]
                by_depth[str(d)] = round(sum(x["exact"] for x in sub) / len(sub), 4)
            kept = [x["needle_kept"] for _, x in rows if "needle_kept" in x]
            p = r["ppl"]["methods"][meth]
            b0 = r["ppl"]["methods"]["bf16"]["nll"]
            per[meth] = dict(
                kv_fraction=round(kv_fraction(meth), 4),
                kv_mib_at_2k=round(kv_fraction(meth) * r["kv_bytes_per_token_bf16"] * 2048 / 2**20, 3),
                niah=dict(n=n, correct=ex, acc=round(ex / n, 4), ci95=wilson(ex, n), by_length=by_len,
                          by_depth=by_depth, by_cell=by_cell,
                          mean_answer_logprob=round(statistics.mean(x["logprob"] for _, x in rows), 4),
                          needle_kept=round(statistics.mean(kept), 4) if kept else None),
                ppl=dict(nll=p["nll"], ppl=p["ppl"], delta_nll=round(p["nll"] - b0, 5),
                         ppl_ratio=round(p["ppl"] / r["ppl"]["methods"]["bf16"]["ppl"], 4),
                         share_of_context_gain_lost=round((p["nll"] - b0) / (r["ppl"]["no_context_nll"] - b0), 4)))
        models[k] = dict(label=r["label"], repo=r["repo"], revision=r["rev"], license=r["license"],
                         layers=r["layers"], kv_heads=r["kv_heads"], q_heads=r["q_heads"], head_dim=r["head_dim"],
                         kv_bytes_per_token_bf16=r["kv_bytes_per_token_bf16"],
                         kv_mib_bf16_at_2k=round(r["kv_bytes_per_token_bf16"] * 2048 / 2**20, 3),
                         checks=r["checks"], ppl_setup=dict(prefix_tokens=r["ppl"]["prefix_tokens"],
                                                             eval_tokens=r["ppl"]["eval_tokens"],
                                                             no_context_nll=r["ppl"]["no_context_nll"],
                                                             no_context_ppl=r["ppl"]["no_context_ppl"]),
                         kv_stats=dict(mean=r["kv_stats"]["mean"], profile_layer=r["kv_stats"]["profile_layer"]),
                         decode_latency=r["decode_latency"], timing_s=r["timing_s"], peak_rss_mb=r["peak_rss_mb"],
                         methods=per)
    total_wall = round(sum(r["timing_s"]["wall"] for r in raw.values()), 1)
    res = dict(title="KV cache quantization and pruning benchmark (CPU, simulated quantization)",
               setup=dict(lengths=LENGTHS, depths=DEPTHS, trials_per_cell=TRIALS_BY_MODEL, group_size=GROUP,
                          sinks=SINKS, snapkv_pool=POOL, scale_zero_dtype="float16",
                          compute_dtype="float32", threads=6, seed=SEED,
                          torch=next(iter(raw.values()))["torch"],
                          transformers=next(iter(raw.values()))["transformers"],
                          note="Quantization is simulated (quantize-dequantize); KV bytes are computed, "
                               "not allocated. Only the context before the question is compressed; the "
                               "question and answer tokens stay full precision."),
               total_compute_wall_s=total_wall, models=models)

    # pooled over both models
    pooled = {}
    for meth in METHODS:
        c = sum(models[k]["methods"][meth]["niah"]["correct"] for k in models)
        n = sum(models[k]["methods"][meth]["niah"]["n"] for k in models)
        pooled[meth] = dict(correct=c, n=n, acc=round(c / n, 4), ci95=wilson(c, n))
    res["pooled_niah"] = pooled

    os.makedirs(os.path.join(HERE, "charts"), exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#555", "axes.labelcolor": "#111",
                         "xtick.color": "#111", "ytick.color": "#111", "figure.facecolor": "white"})
    cmap = LinearSegmentedColormap.from_list("g", ["#ffffff", "#bbbbbb", "#111111"])
    labels = {k: models[k]["label"] for k in models}
    ks = list(models)

    # chart 1: NIAH heatmap per model (method x length@depth)
    fig, axes = plt.subplots(1, len(ks), figsize=(16, 9), dpi=150, sharey=True)
    axes = axes if len(ks) > 1 else [axes]
    cols = [f"{L}@{d}" for L in LENGTHS for d in DEPTHS]
    for ax, k in zip(axes, ks):
        M = [[models[k]["methods"][meth]["niah"]["by_cell"][c] for c in cols] for meth in METHODS]
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i, row in enumerate(M):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:.1f}".rstrip("0").rstrip(".") if v not in (0, 1) else str(int(v)),
                        ha="center", va="center", fontsize=8, color="#fff" if v > 0.6 else "#111")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{int(float(c.split('@')[1]) * 100)}%\n{c.split('@')[0]}" for c in cols], fontsize=8)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels(METHODS)
        ax.set_title(labels[k])
        ax.set_xlabel("needle depth (% of context) / prompt length (tokens)")
    fig.suptitle("Needle-in-a-haystack passkey accuracy by KV cache method (" + ", ".join(f"{labels[k]}: {TRIALS_BY_MODEL[k]} passkey{'s' if TRIALS_BY_MODEL[k] > 1 else ''} per cell" for k in ks) + ")",
                 fontsize=13)
    fig.text(0.01, 0.01, SRC, fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(os.path.join(HERE, "charts", "chart1.png"))
    plt.close(fig)

    # chart 2: accuracy vs KV bytes, one panel per model
    fam_col = {"bf16": "#000", "tok": "#ffffff", "kivi": "#111", "window": "#ffffff", "sink": "#bbb",
               "h2o": "#555", "snapkv": "#111"}
    fam_mk = {"bf16": "D", "tok": "o", "kivi": "o", "window": "s", "sink": "s", "h2o": "^", "snapkv": "^"}
    fig, axes = plt.subplots(1, len(ks), figsize=(16, 7), dpi=150, sharey=True)
    for ax, k in zip(axes, ks):
        seen = {}
        for meth in METHODS:
            v = models[k]["methods"][meth]
            x, y = 100 * v["kv_fraction"], 100 * v["niah"]["acc"]
            n_same = seen.get((x, y), 0)
            seen[(x, y)] = n_same + 1
            fam = meth.split("-")[0]
            ax.scatter(x, y, marker=fam_mk[fam], s=80, facecolor=fam_col[fam], edgecolor="#111", zorder=3)
            off = (-62, 4) if fam == "snapkv" else (6, 4 - 11 * n_same)
            ax.annotate(meth, (x, y), fontsize=8, xytext=off, textcoords="offset points", color="#111")
        ax.set_xlabel("KV cache size of the context (% of BF16 bytes)")
        ax.set_title(f"{labels[k]} (n = {models[k]['methods']['bf16']['niah']['n']} prompts per method)")
        ax.set_ylim(-5, 108)
        ax.set_xlim(10, 108)
        ax.grid(color="#eee")
    axes[0].set_ylabel("passkey retrieval accuracy (%)")
    fig.suptitle("Retrieval accuracy vs KV cache bytes (1k and 2k prompts, 5 depths pooled)", fontsize=13)
    fig.text(0.01, 0.01, SRC + ". Circles: quantization; squares: window/sink; triangles: H2O/SnapKV.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(HERE, "charts", "chart2.png"))
    plt.close(fig)

    # chart 3: perplexity ratio per method
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150)
    w = 0.38
    for i, k in enumerate(ks):
        vals = [models[k]["methods"][m_]["ppl"]["ppl_ratio"] for m_ in METHODS]
        xs = [j + (i - 0.5) * w for j in range(len(METHODS))]
        ax.bar(xs, vals, w, color=["#111", "#bbb"][i], edgecolor="#111", label=labels[k])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.3f}" if v < 10 else f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=90)
    ax.axhline(1, color="#555", lw=0.8)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=45, ha="right")
    ax.set_ylabel("perplexity / BF16-cache perplexity")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 1e4)
    ax.set_title(f"WikiText-2 perplexity on {PPL_EVAL} tokens after a compressed {PPL_PREFIX}-token prefix")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#eee")
    fig.text(0.01, 0.01, SRC, fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(HERE, "charts", "chart3.png"))
    plt.close(fig)

    # chart 4: why keys need per-channel quantization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=150)
    k0 = ks[0]
    rk = raw[k0]["kv_stats"]
    ax = axes[0]
    ax.bar(range(len(rk["key_channel_absmax_mid"])), rk["key_channel_absmax_mid"], color="#111", label="keys")
    ax.bar(range(len(rk["value_channel_absmax_mid"])), rk["value_channel_absmax_mid"], color="#bbb",
           label="values", alpha=0.9)
    ax.set_xlabel("channel index within one KV head (head 0)")
    ax.set_ylabel("max |activation| over the 2k-token context")
    ax.set_title(f"{labels[k0]}, layer {rk['profile_layer']}: per-channel magnitude")
    ax.legend(frameon=False)
    ax = axes[1]
    cats = [("key", "tok"), ("key", "chan"), ("value", "tok"), ("value", "chan")]
    names = ["keys per-token", "keys per-channel", "values per-token", "values per-channel"]
    w = 0.2
    for ci, (nm, lay) in enumerate(cats):
        for bi, b in enumerate((4, 2)):
            for i, k in enumerate(ks):
                v = models[k]["kv_stats"]["mean"][f"{nm}_relerr_{lay}_int{b}"]
                x = ci + (bi * 2 + i - 1.5) * w
                ax.bar(x, v, w, color=["#111", "#555", "#bbb", "#eee"][bi * 2 + i], edgecolor="#111")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=["#111", "#555", "#bbb", "#eee"][bi * 2 + i], edgecolor="#111",
                     label=f"INT{b}, {labels[k]}") for bi, b in enumerate((4, 2)) for i, k in enumerate(ks)]
    ax.set_xticks(range(4))
    ax.set_xticklabels(names)
    ax.set_ylabel("relative reconstruction error ||x - Q(x)|| / ||x||")
    ax.set_title("Quantization error, mean over layers (group size 32)")
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", color="#eee")
    fig.text(0.01, 0.01, SRC, fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(HERE, "charts", "chart4.png"))
    plt.close(fig)

    # chart 5: eviction - needle retention vs accuracy, one panel per model
    fig, axes = plt.subplots(1, len(ks), figsize=(16, 7), dpi=150, sharey=True)
    for ax, k in zip(axes, ks):
        for meth in METHODS:
            v = models[k]["methods"][meth]["niah"]
            if v["needle_kept"] is None:
                continue
            fam = meth.split("-")[0]
            ax.scatter(100 * v["needle_kept"], 100 * v["acc"], marker=fam_mk[fam], s=80,
                       facecolor=fam_col[fam], edgecolor="#111", zorder=3)
            ax.annotate(meth, (100 * v["needle_kept"], 100 * v["acc"]), fontsize=8,
                        xytext=(-50, 4) if meth.startswith("h2o") else (6, -12 if meth.startswith("sink") else 4),
                        textcoords="offset points", color="#111")
        ax.plot([0, 100], [0, 100], color="#bbb", lw=0.8, ls="--")
        ax.set_xlabel("needle tokens still in the cache (%, mean over layers and KV heads)")
        ax.set_title(labels[k])
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 108)
        ax.grid(color="#eee")
    axes[0].set_ylabel("passkey retrieval accuracy (%)")
    fig.suptitle("Eviction at 50% and 25% budgets: retrieval follows needle retention", fontsize=13)
    fig.text(0.01, 0.01, SRC + ". Dashed line: accuracy = retention.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(HERE, "charts", "chart5.png"))
    plt.close(fig)

    res["charts"] = {"chart1.png": "NIAH passkey accuracy heatmap, method x depth x length, per model",
                     "chart2.png": "NIAH accuracy vs KV bytes (% of BF16)",
                     "chart3.png": "perplexity ratio vs BF16 cache per method",
                     "chart4.png": "key vs value channel magnitudes; quantization error per layout",
                     "chart5.png": "eviction: needle retention vs accuracy"}
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: {m_: (v["methods"][m_]["niah"]["acc"], v["methods"][m_]["ppl"]["ppl_ratio"])
                          for m_ in METHODS} for k, v in models.items()}, indent=0))


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


if __name__ == "__main__":
    if "--pilot" in sys.argv:          # tiny smoke test (writes to $KV_OUT, never to raw/)
        LENGTHS, DEPTHS, TRIALS, TRIALS_BY_MODEL = [1024], [0.5], 1, {}
    if "--run" in sys.argv:
        run(sys.argv[sys.argv.index("--run") + 1])
    else:
        aggregate()
