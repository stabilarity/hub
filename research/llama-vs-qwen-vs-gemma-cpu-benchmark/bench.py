"""Measure one small open-weight LLM on CPU: accuracy, speed, RAM and KV cache.

Usage: python bench.py <model_key> [--items N]

Weights are streamed straight from the Hugging Face Hub into the model's
parameter tensors (one HTTP stream, SHA-256 checked against the Hub's LFS
hash), so no weight file is written to disk. Only config and tokenizer files
are downloaded, into a temporary directory that is deleted at the end.
Writes raw/<model_key>.json.
"""
import gc
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import time
import urllib.request

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.initialization import no_init_weights  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "llama-3.2-1b": dict(repo="unsloth/Llama-3.2-1B-Instruct", rev="5a8abab4a5d6f164389b1079fb721cfab8d7126c",
                         official="meta-llama/Llama-3.2-1B-Instruct", family="Llama", label="Llama-3.2-1B-Instruct",
                         license="Llama 3.2 Community License"),
    "gemma-3-1b": dict(repo="unsloth/gemma-3-1b-it", rev="5b11413a10db4e486ef16a20101fd028f8f2499c",
                       official="google/gemma-3-1b-it", family="Gemma", label="Gemma-3-1B-it",
                       license="Gemma Terms of Use"),
    "qwen3-0.6b": dict(repo="Qwen/Qwen3-0.6B", rev="c1899de289a04d12100db370d81485cdf75e47ca",
                       official="Qwen/Qwen3-0.6B", family="Qwen", label="Qwen3-0.6B", license="Apache-2.0"),
    "qwen2.5-0.5b": dict(repo="Qwen/Qwen2.5-0.5B-Instruct", rev="7ae557604adf67be50417f59c2c2f167def9a775",
                         official="Qwen/Qwen2.5-0.5B-Instruct", family="Qwen", label="Qwen2.5-0.5B-Instruct",
                         license="Apache-2.0"),
}

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def rss_mb():
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS", "VmHWM")):
                k, v = line.split(":")
                out[k] = int(v.split()[0]) / 1024
    return out


def hub_meta(repo, rev):
    url = f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.load(r)
    st = [s for s in d["siblings"] if s["rfilename"].endswith(".safetensors")]
    assert len(st) == 1, st
    return st[0], d.get("gated")


def stream_weights(model, repo, rev, fname, sha256):
    """Read the safetensors file in one pass and copy each tensor into its parameter."""
    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{fname}"
    h = hashlib.sha256()
    loaded, skipped = [], []
    with urllib.request.urlopen(url, timeout=120) as r:
        def read_exact(n):
            buf = bytearray(n)
            mv = memoryview(buf)
            got = 0
            while got < n:
                k = r.readinto(mv[got:])
                if not k:
                    raise IOError("stream ended early")
                got += k
            h.update(buf)
            return buf
        n_hdr = struct.unpack("<Q", read_exact(8))[0]
        header = json.loads(bytes(read_exact(n_hdr)))
        header.pop("__metadata__", None)
        pos = 0
        for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            start, end = info["data_offsets"]
            if start > pos:
                read_exact(start - pos)
            raw = read_exact(end - start)
            pos = end
            t = torch.frombuffer(raw, dtype=DTYPES[info["dtype"]]).reshape(info["shape"])
            if name not in params:
                skipped.append(name)
                continue
            p = params[name]
            assert tuple(p.shape) == tuple(t.shape), (name, p.shape, t.shape)
            with torch.no_grad():
                p.data.copy_(t.to(p.dtype))
            loaded.append(name)
            del raw, t
        rest = r.read()
        h.update(rest)
    assert h.hexdigest() == sha256, "SHA-256 mismatch"
    return loaded, skipped


# ---------------------------------------------------------------- tasks
def load_rows(fname, n):
    with open(os.path.join(HERE, "data", fname)) as f:
        return [r["row"] for r in json.load(f)["rows"]][:n]


def hs_pre(text):
    import re
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def build_items(n):
    items = []
    for task, fname in [("arc_easy", "arc_easy_test_first100.json"),
                        ("arc_challenge", "arc_challenge_test_first100.json")]:
        for r in load_rows(fname, n):
            ctx = f"Question: {r['question']}\nAnswer:"
            conts = [" " + c for c in r["choices"]["text"]]
            gold = r["choices"]["label"].index(r["answerKey"])
            items.append((task, ctx, conts, gold))
    for r in load_rows("hellaswag_val_first100.json", n):
        ctx = hs_pre(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize())
        conts = [" " + hs_pre(e) for e in r["endings"]]
        items.append(("hellaswag", ctx, conts, int(r["label"])))
    return items


@torch.no_grad()
def score_item(model, tok, ctx, conts):
    """Log-likelihood of each continuation given ctx (lm-eval-harness style tokenisation).

    The context is run once; its KV cache is repeated for the choices.
    Returns (loglik per choice, n context tokens, n continuation tokens).
    """
    with_sp = tok("a")["input_ids"]
    bos = with_sp[:len(with_sp) - len(tok("a", add_special_tokens=False)["input_ids"])]
    ctx_ids = bos + tok(ctx, add_special_tokens=False)["input_ids"]
    cont_ids = []
    for c in conts:
        # lm-eval-harness: continuation = tokens of (ctx + cont) after len(context tokens)
        whole = bos + tok(ctx + c, add_special_tokens=False)["input_ids"]
        cont_ids.append(whole[len(ctx_ids):])
    out = model(torch.tensor([ctx_ids]), use_cache=True, logits_to_keep=1)
    first_logp = torch.log_softmax(out.logits[0, -1].float(), -1)
    past = out.past_key_values
    k = len(conts)
    past.batch_repeat_interleave(k)
    L = max(len(c) for c in cont_ids)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    inp = torch.full((k, L), pad, dtype=torch.long)
    mask = torch.zeros((k, len(ctx_ids) + L), dtype=torch.long)
    mask[:, :len(ctx_ids)] = 1
    for i, c in enumerate(cont_ids):
        inp[i, :len(c)] = torch.tensor(c)
        mask[i, len(ctx_ids):len(ctx_ids) + len(c)] = 1
    pos = torch.arange(len(ctx_ids), len(ctx_ids) + L).unsqueeze(0).expand(k, L)
    hid = model.model(input_ids=inp, attention_mask=mask, position_ids=pos, past_key_values=past,
                      use_cache=True).last_hidden_state
    lls = []
    for i, c in enumerate(cont_ids):
        ll = first_logp[c[0]].item()
        if len(c) > 1:
            lg = model.lm_head(hid[i, :len(c) - 1])
            if getattr(model.config, "final_logit_softcapping", None):
                cap = model.config.final_logit_softcapping
                lg = torch.tanh(lg / cap) * cap
            lp = torch.log_softmax(lg.float(), -1)
            ll += lp[torch.arange(len(c) - 1), torch.tensor(c[1:])].sum().item()
        lls.append(ll)
    return lls, len(ctx_ids), sum(len(c) for c in cont_ids)


@torch.no_grad()
def naive_ll(model, tok, ctx, conts):
    """Reference: one full forward pass per choice (used to check score_item)."""
    with_sp = tok("a")["input_ids"]
    bos = with_sp[:len(with_sp) - len(tok("a", add_special_tokens=False)["input_ids"])]
    ctx_ids = bos + tok(ctx, add_special_tokens=False)["input_ids"]
    lls = []
    for c in conts:
        whole = bos + tok(ctx + c, add_special_tokens=False)["input_ids"]
        lg = model(torch.tensor([whole])).logits[0].float()
        lp = torch.log_softmax(lg, -1)
        tgt = whole[len(ctx_ids):]
        lls.append(sum(lp[len(ctx_ids) - 1 + j, t].item() for j, t in enumerate(tgt)))
    return lls


# ---------------------------------------------------------------- speed
@torch.no_grad()
def speed(model, tok, threads, n_prompt=512, n_new=128):
    torch.set_num_threads(threads)
    text = " ".join(r["ctx"] for r in load_rows("hellaswag_val_first100.json", 100))
    ids = tok(text, add_special_tokens=False)["input_ids"][:n_prompt]
    assert len(ids) == n_prompt
    x = torch.tensor([ids])
    model(x[:, :32], use_cache=True, logits_to_keep=1)  # warm-up
    t0 = time.perf_counter()
    out = model(x, use_cache=True, logits_to_keep=1)
    t_pre = time.perf_counter() - t0
    past = out.past_key_values
    kv_bytes = 0
    for layer in past.layers:
        for t in (layer.keys, layer.values):
            kv_bytes += t.numel() * t.element_size()
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    t0 = time.perf_counter()
    for i in range(n_new):
        o = model(nxt, past_key_values=past, use_cache=True,
                  position_ids=torch.tensor([[n_prompt + i]]))
        past = o.past_key_values
        nxt = o.logits[:, -1].argmax(-1, keepdim=True)
    t_dec = time.perf_counter() - t0
    return dict(threads=threads, prompt_tokens=n_prompt, new_tokens=n_new,
                prefill_s=round(t_pre, 3), prefill_tok_s=round(n_prompt / t_pre, 2),
                decode_s=round(t_dec, 3), decode_tok_s=round(n_new / t_dec, 3),
                kv_bytes_after_prefill=kv_bytes)


def kv_from_config(cfg):
    c = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    layers = c.num_hidden_layers
    kvh = c.num_key_value_heads
    hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    per_layer = kvh * hd * 2 * 2  # K and V, bf16
    lt = getattr(c, "layer_types", None) or ["full_attention"] * layers
    win = getattr(c, "sliding_window", None)
    n_slide = sum(1 for t in lt if t == "sliding_attention") if win else 0
    def at(ctx):
        return (layers - n_slide) * per_layer * ctx + n_slide * per_layer * min(ctx, win or ctx)
    return dict(layers=layers, kv_heads=kvh, q_heads=c.num_attention_heads, head_dim=hd,
                hidden=c.hidden_size, vocab=c.vocab_size, sliding_layers=n_slide, sliding_window=win,
                max_position_embeddings=c.max_position_embeddings,
                bytes_per_token_full=layers * per_layer,
                bytes_at_512=at(512), bytes_at_4k=at(4096), bytes_at_32k=at(32768))


def main():
    key = sys.argv[1]
    n = int(sys.argv[sys.argv.index("--items") + 1]) if "--items" in sys.argv else 40
    m = MODELS[key]
    torch.manual_seed(0)
    t_start = time.time()
    tmp = tempfile.mkdtemp(prefix="hfcfg-", dir=os.environ.get("BENCH_TMP"))
    try:
        st, gated = hub_meta(m["repo"], m["rev"])
        local = snapshot_download(m["repo"], revision=m["rev"], cache_dir=tmp,
                                  allow_patterns=["*.json", "tokenizer.model"])
        cfg = AutoConfig.from_pretrained(local)
        tok = AutoTokenizer.from_pretrained(local)
        rss0 = rss_mb()
        t0 = time.time()
        with no_init_weights():
            model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
        model.eval()
        loaded, skipped = stream_weights(model, m["repo"], m["rev"], st["rfilename"], st["lfs"]["sha256"])
        if getattr(cfg.get_text_config(), "tie_word_embeddings", False):
            model.lm_head.weight = model.model.embed_tokens.weight
        n_params = sum(p.numel() for p in model.parameters())
        n_emb = model.model.embed_tokens.weight.numel()
        t_load = time.time() - t0
        rss_loaded = rss_mb()
        gc.collect()

        torch.set_num_threads(6)
        items = build_items(n)
        chk = [(score_item(model, tok, it[1], it[2])[0], naive_ll(model, tok, it[1], it[2]))
               for it in (items[0], items[-1])]
        check_maxdiff = max(abs(a - b) for x, y in chk for a, b in zip(x, y))
        check_same_argmax = all(max(range(len(x)), key=x.__getitem__) == max(range(len(y)), key=y.__getitem__)
                                for x, y in chk)
        per_task = {}
        tok_ctx = tok_cont = 0
        t0 = time.time()
        for task, ctx, conts, gold in items:
            lls, nc, nt = score_item(model, tok, ctx, conts)
            tok_ctx += nc
            tok_cont += nt
            blen = [len(c.encode()) for c in conts]
            pred = max(range(len(lls)), key=lambda i: lls[i])
            pred_n = max(range(len(lls)), key=lambda i: lls[i] / blen[i])
            d = per_task.setdefault(task, dict(n=0, correct=0, correct_norm=0))
            d["n"] += 1
            d["correct"] += int(pred == gold)
            d["correct_norm"] += int(pred_n == gold)
        t_acc = time.time() - t0

        sp = [speed(model, tok, 6), speed(model, tok, 2)]
        rss_end = rss_mb()
        res = dict(key=key, **m, gated_mirror=gated, file=st["rfilename"], file_bytes=st["size"],
                   sha256=st["lfs"]["sha256"], dtype="bfloat16", params=n_params, embedding_params=n_emb,
                   tensors_loaded=len(loaded), tensors_skipped=skipped,
                   load_s=round(t_load, 1), rss_before_load_mb=round(rss0["VmRSS"], 1),
                   rss_loaded_mb=round(rss_loaded["VmRSS"], 1), peak_rss_mb=round(rss_end["VmHWM"], 1),
                   check_shared_kv_vs_naive=dict(max_abs_loglik_diff=round(check_maxdiff, 4),
                                                 same_argmax=check_same_argmax),
                   items_per_task=n, accuracy=per_task, scoring_s=round(t_acc, 1),
                   scored_tokens=dict(context=tok_ctx, continuation=tok_cont),
                   speed=sp, kv=kv_from_config(cfg), torch=torch.__version__,
                   wall_s=round(time.time() - t_start, 1))
        os.makedirs(os.path.join(HERE, "raw"), exist_ok=True)
        with open(os.path.join(HERE, "raw", f"{key}.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps({k: res[k] for k in ("key", "accuracy", "speed", "peak_rss_mb", "wall_s", "scoring_s",
                                              "check_shared_kv_vs_naive", "scored_tokens", "tensors_skipped")}))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
