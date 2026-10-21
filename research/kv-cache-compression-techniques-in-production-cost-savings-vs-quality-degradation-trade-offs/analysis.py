"""KV-cache compression in production: cost savings vs quality degradation.

Deterministic, offline analysis. Reads data/ and writes charts/chart1.png ... chart4.png and results.json
next to this file. No network access, no random numbers.

Data
  data/models.csv                    attention configurations and parameter counts (model reports, Hugging Face)
  data/gpus.csv                      HBM capacity and bandwidth (NVIDIA spec pages), on-demand price (RunPod, 2026-10-21)
  data/kv_formats.csv                compression factor of each KV format vs BF16 (nominal or realised)
  data/table2_longbench.csv          LongBench scores per method, Agrawal and Mayer, arXiv:2607.05399, Table 2
  data/table4_compression_llama.csv  realised compression per context bucket, arXiv:2607.05399, Table 4

Model (our estimate, an idealised upper bound)
  KV bytes per token  GQA: 2 * layers * kv_heads * head_dim * bytes;  MLA: layers * (latent + rope) * bytes
  KV budget           0.92 * HBM * n_gpu - weight bytes (vLLM default gpu_memory_utilization = 0.92)
  Batch               min(floor(budget / (context * kv_bytes_per_token / c)), 256)
  Decode step time    (weight bytes + batch * context * kv_bytes_per_token / c) / (bandwidth * n_gpu)
  Cost                USD per GPU-hour * n_gpu / (3600 * batch / step time) * 1e6  -> USD per 1M output tokens
  Dequantisation compute, scale metadata, activations and prefill are ignored, so savings are upper bounds.
"""
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CHARTS = os.path.join(HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)

INK, INK2, MUTED, LINE, FAINT = "#000000", "#111111", "#555555", "#bbbbbb", "#dddddd"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
})
FIGSIZE, DPI = (10.67, 6.0), 150

GPU_MEM_UTIL = 0.92      # vLLM default --gpu-memory-utilization
BATCH_CAP = 256          # scheduler cap on concurrent sequences (our assumption)
BF16_BYTES = 2
CONTEXTS = [8192, 32768, 131072]
GB = 1e9


def load(name):
    with open(os.path.join(DATA, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


models = {m["model"]: m for m in load("models.csv")}
gpus = {g["gpu"]: g for g in load("gpus.csv")}
formats = {f["format"]: float(f["compression_vs_bf16"]) for f in load("kv_formats.csv")}


def kv_bytes_per_token(m, bytes_per_el=BF16_BYTES):
    L = int(m["layers"])
    if m["attention"] == "MLA":
        return L * (int(m["mla_latent_dim"]) + int(m["mla_rope_dim"])) * bytes_per_el
    return 2 * L * int(m["kv_heads"]) * int(m["head_dim"]) * bytes_per_el


# Serving units: one model, one GPU type, n GPUs with tensor parallelism, BF16 weights
UNITS = [
    {"unit": "Llama-3.1-8B on 1x H100", "model": "Llama-3.1-8B-Instruct", "gpu": "H100 SXM", "n": 1},
    {"unit": "Qwen3-32B on 1x H200", "model": "Qwen3-32B", "gpu": "H200 SXM", "n": 1},
    {"unit": "Llama-3.3-70B on 2x H200", "model": "Llama-3.3-70B-Instruct", "gpu": "H200 SXM", "n": 2},
]


def serve(unit, ctx, c):
    m, g, n = models[unit["model"]], gpus[unit["gpu"]], unit["n"]
    w = int(m["params"]) * BF16_BYTES
    budget = GPU_MEM_UTIL * float(g["hbm_gb"]) * GB * n - w
    per_seq = ctx * kv_bytes_per_token(m) / c
    b_mem = int(budget // per_seq)
    b = min(b_mem, BATCH_CAP)
    if b < 1:
        return {"batch": 0, "memory_bound": True, "usd_per_mtok": None, "tok_s": 0.0, "kv_share": None}
    bw = float(g["bandwidth_tb_s"]) * 1e12 * n
    kv_read = b * per_seq
    t = (w + kv_read) / bw
    tok_s = b / t
    usd = float(g["usd_per_gpu_hour"]) * n / (3600 * tok_s) * 1e6
    return {"batch": b, "memory_bound": b_mem < BATCH_CAP, "usd_per_mtok": usd, "tok_s": tok_s,
            "kv_share": kv_read / (w + kv_read), "budget_gb": budget / GB, "per_seq_gb": per_seq / GB}


results = {"meta": {
    "gpu_mem_util": GPU_MEM_UTIL, "batch_cap": BATCH_CAP, "contexts": CONTEXTS,
    "prices_accessed": "2026-10-21", "note": "idealised bandwidth-bound decode model; upper bound on savings"}}

# ------------------------------------------------------------------------------------------------
# RQ1: KV bytes per token and concurrency
# ------------------------------------------------------------------------------------------------
rq1_kv = []
for name, m in models.items():
    kvb = kv_bytes_per_token(m)
    row = {"model": name, "attention": m["attention"], "kv_kib_per_token_bf16": round(kvb / 1024, 2),
           "kv_gb_per_128k_seq_bf16": round(kvb * 131072 / GB, 2)}
    for f, c in formats.items():
        row[f"kv_kib_per_token_{f}"] = round(kvb / c / 1024, 2)
    rq1_kv.append(row)
results["rq1_kv_per_token"] = rq1_kv
ll70 = kv_bytes_per_token(models["Llama-3.3-70B-Instruct"])
ds = kv_bytes_per_token(models["DeepSeek-V3"])
results["rq1_mla_vs_gqa70b_ratio"] = round(ll70 / ds, 2)

FMT_ORDER = ["BF16", "FP8", "KIVI-4", "SnapKV-0.75", "NVFP4", "KIVI-2"]
rq1_conc = []
for u in UNITS:
    for ctx in CONTEXTS:
        row = {"unit": u["unit"], "context": ctx}
        for f in FMT_ORDER:
            s = serve(u, ctx, formats[f])
            row[f"batch_{f}"] = s["batch"]
        rq1_conc.append(row)
results["rq1_concurrency"] = rq1_conc

# ------------------------------------------------------------------------------------------------
# RQ2: cost per 1M output tokens and savings vs BF16
# ------------------------------------------------------------------------------------------------
rq2 = []
for u in UNITS:
    for ctx in CONTEXTS:
        base = serve(u, ctx, 1.0)
        for f in FMT_ORDER:
            s = serve(u, ctx, formats[f])
            sav = None if (s["usd_per_mtok"] is None or base["usd_per_mtok"] is None) else \
                1 - s["usd_per_mtok"] / base["usd_per_mtok"]
            rq2.append({"unit": u["unit"], "context": ctx, "format": f, "compression": formats[f],
                        "batch": s["batch"], "memory_bound": s["memory_bound"],
                        "usd_per_mtok_output": round(s["usd_per_mtok"], 4),
                        "tok_s": round(s["tok_s"], 1), "kv_share_of_bytes": round(s["kv_share"], 3),
                        "saving_vs_bf16": round(sav, 4), "ideal_saving_1_minus_1_over_c": round(1 - 1 / formats[f], 4)})
results["rq2_cost"] = rq2


def crossover_context(u, c):
    """Smallest context (tokens) at which the batch becomes memory-limited below the cap."""
    m, g, n = models[u["model"]], gpus[u["gpu"]], u["n"]
    budget = GPU_MEM_UTIL * float(g["hbm_gb"]) * GB * n - int(m["params"]) * BF16_BYTES
    return int(math.ceil(budget * c / (BATCH_CAP * kv_bytes_per_token(m))))


results["rq2_crossover_context_tokens"] = [
    {"unit": u["unit"], **{f: crossover_context(u, formats[f]) for f in FMT_ORDER}} for u in UNITS]

# A dense sweep for chart 3 and the savings curve
sweep_ctx = [int(x) for x in np.unique(np.round(np.logspace(np.log10(1024), np.log10(131072), 60)))]
sweep = {u["unit"]: {f: [serve(u, x, formats[f])["usd_per_mtok"] for x in sweep_ctx] for f in FMT_ORDER} for u in UNITS}

# ------------------------------------------------------------------------------------------------
# RQ3: quality retention per method (LongBench, arXiv:2607.05399 Table 2) vs realised compression
# ------------------------------------------------------------------------------------------------
t2 = load("table2_longbench.csv")
METHODS = ["KIVI-2", "KIVI-4", "TurboQuant-3", "TurboQuant-4", "SnapKV-0.75", "CaM"]
t4 = load("table4_compression_llama.csv")
t4map = {"KIVI-2": "KIVI2", "KIVI-4": "KIVI4", "TurboQuant-3": "TurboQuant3", "TurboQuant-4": "TurboQuant4",
         "SnapKV-0.75": "SnapKV-0.75", "CaM": "CaM"}
realised = {}
for meth, col in t4map.items():
    vals = [float(r[col]) for r in t4 if r[col] != ""]
    realised[meth] = {"mean": round(float(np.mean(vals)), 2), "min": min(vals), "max": max(vals)}

u8 = UNITS[0]
base32 = serve(u8, 32768, 1.0)["usd_per_mtok"]
rq3 = []
for meth in METHODS:
    per_model = {}
    for mod in ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3"]:
        rows = [r for r in t2 if r["model"] == mod]
        ret = {r["dataset"]: float(r[meth]) / float(r["All-KV"]) for r in rows}
        per_model[mod] = {"mean_retention": round(float(np.mean(list(ret.values()))), 4),
                          "min_retention": round(min(ret.values()), 4),
                          "worst_dataset": min(ret, key=ret.get),
                          "per_dataset": {k: round(v, 4) for k, v in ret.items()}}
    c = realised[meth]["mean"]
    cost = serve(u8, 32768, c)["usd_per_mtok"]
    saving = 1 - cost / base32
    pooled_mean = float(np.mean([per_model[m]["mean_retention"] for m in per_model]))
    pooled_min = min(per_model[m]["min_retention"] for m in per_model)
    rq3.append({"method": meth, "realised_compression_mean": c,
                "realised_compression_range": [realised[meth]["min"], realised[meth]["max"]],
                "usd_per_mtok_llama8b_h100_32k": round(cost, 4), "saving_vs_bf16_32k": round(saving, 4),
                "mean_retention_pooled": round(pooled_mean, 4), "min_retention_pooled": round(pooled_min, 4),
                "cost_saved_pct_per_worst_case_point_lost":
                    round(100 * saving / max(1e-9, 100 * (1 - pooled_min)), 2) if pooled_min < 1 else 0.0,
                "per_model": per_model})
results["rq3_quality"] = rq3
results["rq3_baseline_usd_per_mtok_llama8b_h100_32k"] = round(base32, 4)

# Pareto frontier on (cost, worst-case retention): a method is dominated if another is cheaper-or-equal and
# at least as good on worst-case retention, strictly better on one.
pts = [("BF16 (All-KV)", base32, 1.0)] + [(r["method"], r["usd_per_mtok_llama8b_h100_32k"], r["min_retention_pooled"]) for r in rq3]
front = []
for n_, c_, q_ in pts:
    dom = any((c2 <= c_ and q2 >= q_) and (c2 < c_ or q2 > q_) for n2, c2, q2 in pts if n2 != n_)
    if not dom:
        front.append(n_)
results["rq3_pareto_worst_case"] = front
ptsm = [("BF16 (All-KV)", base32, 1.0)] + [(r["method"], r["usd_per_mtok_llama8b_h100_32k"], r["mean_retention_pooled"]) for r in rq3]
frontm = []
for n_, c_, q_ in ptsm:
    dom = any((c2 <= c_ and q2 >= q_) and (c2 < c_ or q2 > q_) for n2, c2, q2 in ptsm if n2 != n_)
    if not dom:
        frontm.append(n_)
results["rq3_pareto_mean"] = frontm

# Summarisation-only retention (the most sensitive workload)
results["rq3_summarisation_retention"] = {
    r["method"]: {m: r["per_model"][m]["per_dataset"]["MultiNews"] for m in r["per_model"]} for r in rq3}

# ------------------------------------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------------------------------------


def finish(fig, ax, title, xlabel, ylabel, source, name):
    ax.set_title(title, loc="left", fontsize=14, color=INK, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=FAINT, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.01, source, fontsize=8.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(CHARTS, name), dpi=DPI)
    plt.close(fig)


GREYS = [INK, MUTED, LINE, FAINT, "#f3f3f3", "#f9f9f9"]

# Chart 1: KV memory per 128K-token sequence by model and format
order_models = ["Llama-3.3-70B-Instruct", "Qwen3-32B", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "DeepSeek-V3"]
fmts1 = ["BF16", "FP8", "KIVI-4", "NVFP4", "KIVI-2"]
fig, ax = plt.subplots(figsize=FIGSIZE)
y = np.arange(len(order_models))
h = 0.16
for i, f in enumerate(fmts1):
    vals = [kv_bytes_per_token(models[mn]) * 131072 / formats[f] / GB for mn in order_models]
    ax.barh(y + (i - 2) * h, vals, height=h - 0.01, color=GREYS[i], edgecolor=MUTED, linewidth=0.6, label=f"{f} ({formats[f]:g}x)")
ax.set_yticks(y)
ax.set_yticklabels([f"{mn} ({models[mn]['attention']})" for mn in order_models])
ax.invert_yaxis()
ax.legend(frameon=False, loc="lower right", fontsize=10)
finish(fig, ax, "Chart 1. KV-cache memory for one 128K-token sequence",
       "GB per sequence (decimal GB)", "",
       "Source: model configs (Llama 3, Qwen3, DeepSeek-V3 reports; Hugging Face); realised KIVI ratios from arXiv:2607.05399; our calculation",
       "chart1.png")

# Chart 2: concurrent sequences vs context length, Llama-3.1-8B on 1x H100
fig, ax = plt.subplots(figsize=FIGSIZE)
styles = ["-", "--", "-.", ":", "-", "--"]
c2 = [("BF16", "-", INK), ("FP8", "--", INK), ("KIVI-4", "-.", MUTED), ("NVFP4", "-", LINE), ("KIVI-2", ":", INK)]
for f, ls_, col_ in c2:
    bs = [serve(u8, x, formats[f])["batch"] for x in sweep_ctx]
    lab = "NVFP4 or SnapKV-0.75 (4x)" if f == "NVFP4" else f"{f} ({formats[f]:g}x)"
    ax.plot([x / 1024 for x in sweep_ctx], bs, ls=ls_, color=col_, lw=2.4, label=lab)
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
ax.set_xticklabels(["1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K"])
ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
ax.set_yticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256"])
ax.axhline(BATCH_CAP, color=LINE, lw=1)
ax.text(1.05, BATCH_CAP * 1.08, "scheduler cap = 256 (assumption)", color=MUTED, fontsize=9)
ax.legend(frameon=False, loc="lower left", fontsize=10)
finish(fig, ax, "Chart 2. Concurrent sequences that fit: Llama-3.1-8B on one H100 80GB",
       "Tokens held per sequence (log scale)", "Max concurrent sequences (log scale)",
       "Source: NVIDIA H100 spec; vLLM default memory utilisation 0.92; BF16 weights; our calculation", "chart2.png")

# Chart 3: cost per 1M output tokens vs context for the three serving units, BF16 vs FP8 vs KIVI-2
fig, ax = plt.subplots(figsize=FIGSIZE)
lss = {"BF16": "-", "FP8": "--", "KIVI-2": ":"}
cols = [INK, MUTED, LINE]
for ui, u in enumerate(UNITS):
    for f in ["BF16", "FP8", "KIVI-2"]:
        ys = sweep[u["unit"]][f]
        xs = [x / 1024 for x, v in zip(sweep_ctx, ys) if v is not None]
        ys = [v for v in ys if v is not None]
        ax.plot(xs, ys, ls=lss[f], color=cols[ui], lw=2.2, label=f"{u['unit']}, {f}")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
ax.set_xticklabels(["1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K"])
ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=1)
finish(fig, ax, "Chart 3. Decode cost per 1M output tokens vs context length",
       "Tokens held per sequence (log scale)", "USD per 1M output tokens (log scale)",
       "Source: NVIDIA H100/H200 specs; RunPod on-demand prices accessed 2026-10-21; bandwidth-bound decode model (upper bound); our calculation",
       "chart3.png")

# Chart 4: cost vs quality retention (pooled over Llama-3.1-8B and Mistral-7B), worst-case and mean
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.scatter([base32], [100], s=90, color=INK, marker="s", zorder=3)
ax.annotate("BF16 (All-KV)", (base32, 100), textcoords="offset points", xytext=(6, 6), fontsize=10)
for r in rq3:
    x = r["usd_per_mtok_llama8b_h100_32k"]
    ax.plot([x, x], [100 * r["min_retention_pooled"], 100 * r["mean_retention_pooled"]], color=LINE, lw=2, zorder=1)
    ax.scatter([x], [100 * r["mean_retention_pooled"]], s=70, color=INK, zorder=3)
    ax.scatter([x], [100 * r["min_retention_pooled"]], s=70, facecolors="#ffffff", edgecolors=INK, zorder=3)
    ax.annotate(f"{r['method']} ({r['realised_compression_mean']:g}x)", (x, 100 * r["mean_retention_pooled"]),
                textcoords="offset points", xytext=(6, 4), fontsize=9.5)
ax.scatter([], [], s=70, color=INK, label="Mean retention over 12 model-task pairs")
ax.scatter([], [], s=70, facecolors="#ffffff", edgecolors=INK, label="Worst model-task pair")
ax.axhline(95, color=LINE, lw=1, ls="--")
ax.text(1.0, 95.6, "95% retention line (reference)", color=MUTED, fontsize=9)
ax.legend(frameon=False, loc="lower right", fontsize=10)
finish(fig, ax, "Chart 4. Cost saved vs quality kept, LongBench tasks",
       "USD per 1M output tokens, Llama-3.1-8B on one H100 at 32K context (realised compression)",
       "Score retained vs uncompressed cache (%)",
       "Source: LongBench scores and realised compression from Agrawal and Mayer, arXiv:2607.05399 (Tables 2, 4); cost model ours",
       "chart4.png")

with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1)


def has_none(o):
    if o is None:
        return True
    if isinstance(o, dict):
        return any(has_none(v) for v in o.values())
    if isinstance(o, list):
        return any(has_none(v) for v in o)
    return False


assert not has_none(results), "results.json contains nulls"
print(json.dumps({k: results[k] for k in results if k not in ("rq2_cost",)}, indent=1)[:12000])
