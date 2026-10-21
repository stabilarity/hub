"""Build the committed data files from the public SPROUT routing dataset (needs network once).

SPROUT (CARROT-LLM-Routing/SPROUT on Hugging Face, Somerstep et al., arXiv:2502.03261) holds 44,241 prompts with
the response, judged correctness score (0-1) and input/output token counts of 13 models. This script

  1. reads the five parquet files (train/validation/test) from a local directory,
  2. writes data/sprout_scores.csv.gz: one row per prompt with split, task family, and for each model the score
     and the input/output token counts (no prompt or response text),
  3. trains a simple text router on the TRAIN split (TF-IDF over the prompt + one ridge regressor per model that
     predicts the judged score) and writes its predicted scores for every TEST prompt to
     data/router_predictions.csv.gz (validation prompts are used to set routing thresholds, test prompts to measure
     realised cost and quality). Two variants: 'in_dist' (trained on all train families) and 'lofo'
     (leave-one-family-out: for each test prompt, the regressor was trained without that prompt's family).

Usage:  python3 build_data.py /path/to/sprout_parquet_dir
Requires pyarrow, scikit-learn, numpy. Deterministic (no random state is used by TF-IDF or ridge).
"""
import csv
import gzip
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SRC = sys.argv[1]

MODELS = [
    "aws-claude-3-5-sonnet-v1", "openai-gpt-4o", "openai-gpt-4o-mini", "aws-titan-text-premier-v1",
    "wxai-llama-3-405b-instruct", "wxai-llama-3-3-70b-instruct", "wxai-llama-3-1-70b-instruct",
    "wxai-mixtral-8x7b-instruct-v01", "wxai-llama-3-1-8b-instruct", "wxai-granite-3-8b-instruct-8k-max-tokens",
    "wxai-llama-3-2-3b-instruct", "wxai-granite-3-2b-instruct-8k-max-tokens", "wxai-llama-3-2-1b-instruct",
]


def family(ds):
    if ds.startswith("openhermes"):
        return "chat_instructions"
    if ds.startswith("TIGER-Lab/MMLU-Pro"):
        return "mmlu_pro"
    if ds.startswith("lighteval/MATH"):
        return "math"
    if ds.startswith("Idavidrein/gpqa"):
        return "gpqa"
    if ds.startswith("TAUR-Lab/MuSR"):
        return "musr"
    if ds.startswith("rungalileo/ragbench"):
        return "ragbench"
    return "other"


rows, prompts = [], []
for split, files in [("train", ["train-00000-of-00003", "train-00001-of-00003", "train-00002-of-00003"]),
                     ("validation", ["validation-00000-of-00001"]), ("test", ["test-00000-of-00001"])]:
    for f in files:
        t = pq.read_table(os.path.join(SRC, f + ".parquet"))
        for r in t.to_pylist():
            row = {"key": r["key"], "split": split, "dataset": r["dataset"], "family": family(r["dataset"])}
            ok = True
            for m in MODELS:
                v = r[m]
                if v is None or v.get("score") is None:
                    ok = False
                    break
                row[m + "|score"] = float(v["score"])
                row[m + "|in"] = int(v["num_input_tokens"] or 0)
                row[m + "|out"] = int(v["num_output_tokens"] or 0)
            if ok:
                rows.append(row)
                prompts.append(r["prompt"] or "")
print("rows with all models scored:", len(rows))

cols = ["key", "split", "dataset", "family"] + [f"{m}|{k}" for m in MODELS for k in ("score", "in", "out")]
with gzip.open(os.path.join(DATA, "sprout_scores.csv.gz"), "wt", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

idx_tr = [i for i, r in enumerate(rows) if r["split"] == "train"]
idx_te = [i for i, r in enumerate(rows) if r["split"] in ("validation", "test")]
vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=3)
X_tr = vec.fit_transform([prompts[i][:4000] for i in idx_tr])
X_te = vec.transform([prompts[i][:4000] for i in idx_te])
fam_tr = np.array([rows[i]["family"] for i in idx_tr])
fam_te = np.array([rows[i]["family"] for i in idx_te])

pred = {("in_dist", m): np.zeros(len(idx_te)) for m in MODELS}
pred.update({("lofo", m): np.zeros(len(idx_te)) for m in MODELS})
for m in MODELS:
    y = np.array([rows[i][m + "|score"] for i in idx_tr])
    reg = Ridge(alpha=1.0).fit(X_tr, y)
    pred[("in_dist", m)] = np.clip(reg.predict(X_te), 0, 1)
    for fam in sorted(set(fam_te)):
        keep = fam_tr != fam
        reg = Ridge(alpha=1.0).fit(X_tr[keep], y[keep])
        sel = fam_te == fam  # held-out family: validation and test prompts of this family
        pred[("lofo", m)][sel] = np.clip(reg.predict(X_te[sel]), 0, 1)
    print("trained", m)

with gzip.open(os.path.join(DATA, "router_predictions.csv.gz"), "wt", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["key", "split"] + [f"{v}|{m}" for v in ("in_dist", "lofo") for m in MODELS])
    for j, i in enumerate(idx_te):
        w.writerow([rows[i]["key"], rows[i]["split"]] + [f"{pred[(v, m)][j]:.4f}" for v in ("in_dist", "lofo") for m in MODELS])
print("done")
