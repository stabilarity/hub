"""
Step 1 of 2: score every text in data/ with Pythia checkpoints (CPU).

Writes data/scores.csv with one row per (model, set, doc) and four membership
signals per row. Requires the model weights from the Hugging Face Hub on first
run (EleutherAI/pythia-70m, -160m, -410m); everything after that is offline.

    python3 score.py            # ~35 min on 6 CPU cores
    python3 analysis.py         # statistics, charts, results.json (offline)
"""
import csv, json, os, re, time, zlib, platform
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(0)
torch.set_num_threads(os.cpu_count())
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
MODELS = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]
MAX_TOK = 256
K = 0.2


def clean_fresh(t):
    # Pile Wikipedia text has no "== Heading ==" lines; strip them so the
    # fresh set differs from the Pile set in date, not in markup.
    lines = [l for l in t.split("\n") if not re.match(r"^\s*=+.*=+\s*$", l)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def load_sets():
    tr = json.load(open(os.path.join(D, "pile_train_sample.json")))["domains"]
    te = json.load(open(os.path.join(D, "pile_test_sample.json")))["domains"]
    fr = json.load(open(os.path.join(D, "wikipedia_fresh_2026.json")))["articles"]
    sets = []
    for dom, key in [("Wikipedia (en)", "wiki"), ("ArXiv", "arxiv"), ("Pile-CC", "cc")]:
        sets += [(f"pile_train_{key}", 1, t) for t in tr[dom]]
        sets += [(f"pile_test_{key}", 0, t) for t in te[dom]]
    sets += [("fresh_wiki_2026", 0, clean_fresh(a["text"])) for a in fr]
    for L in (64, 128):
        for r in json.load(open(os.path.join(D, f"wikimia_{L}.json")))["rows"]:
            sets.append((f"wikimia_{L}", r["label"], r["text"]))
    return sets


@torch.no_grad()
def score_batch(model, tok, texts):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOK)
    logits = model(**enc).logits[:, :-1].float()
    ids = enc.input_ids[:, 1:]
    mask = enc.attention_mask[:, 1:].bool()
    logp = torch.log_softmax(logits, -1)
    tok_lp = logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    p = logp.exp()
    mu = (p * logp).sum(-1)                      # E_z[log p(z)]
    sigma = ((p * logp.pow(2)).sum(-1) - mu.pow(2)).clamp_min(1e-8).sqrt()
    zpp = (tok_lp - mu) / sigma                  # Min-K%++ token score
    out = []
    for i, t in enumerate(texts):
        m = mask[i]
        lp = tok_lp[i][m].numpy()
        zz = zpp[i][m].numpy()
        n = len(lp)
        k = max(1, int(round(K * n)))
        nll = float(-lp.mean())
        used = tok.decode(enc.input_ids[i][enc.attention_mask[i].bool()])
        zl = len(zlib.compress(used.encode("utf-8")))
        out.append({
            "n_tokens": n + 1,
            "loss": nll,
            "zlib_ratio": nll * n / zl,
            "mink": float(np.sort(lp)[:k].mean()),
            "minkpp": float(np.sort(zz)[:k].mean()),
            "zlib_bytes_per_char": zl / max(1, len(used)),
        })
    return out


def main():
    sets = load_sets()
    rows = []
    meta = {"python": platform.python_version(), "torch": torch.__version__,
            "cpu": platform.processor() or platform.machine(), "threads": torch.get_num_threads(),
            "models": {}}
    import transformers
    meta["transformers"] = transformers.__version__
    for name in MODELS:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(name)
        tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
        order = sorted(range(len(sets)), key=lambda i: len(sets[i][2]))
        res = [None] * len(sets)
        B = 8
        for s in range(0, len(order), B):
            idx = order[s:s + B]
            for i, r in zip(idx, score_batch(model, tok, [sets[j][2] for j in idx])):
                res[i] = r
        for i, (sname, label, _) in enumerate(sets):
            rows.append({"model": name.split("/")[1], "set": sname, "doc": i, "member": label, **res[i]})
        meta["models"][name] = {"seconds": round(time.time() - t0, 1),
                                "params": sum(p.numel() for p in model.parameters())}
        print(name, meta["models"][name], flush=True)
        del model
    with open(os.path.join(D, "scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(meta, open(os.path.join(D, "scoring_env.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
