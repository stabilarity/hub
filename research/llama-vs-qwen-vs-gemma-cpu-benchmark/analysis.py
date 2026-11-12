"""Llama vs Qwen vs Gemma on a CPU: aggregate raw measurements into results.json and charts.

Step 1 (network, about 13 min of CPU for all four models, run once):
    python bench.py qwen2.5-0.5b --items 30   # likewise qwen3-0.6b, gemma-3-1b, llama-3.2-1b
    (or: python analysis.py --run)
Step 2 (offline, deterministic):
    python analysis.py      # reads raw/*.json, writes results.json and charts/chart1-4.png
"""
import json
import math
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["llama-3.2-1b", "gemma-3-1b", "qwen3-0.6b", "qwen2.5-0.5b"]
TASKS = [("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Challenge"), ("hellaswag", "HellaSwag")]
CHANCE = 0.25  # all three tasks have four options
GREYS = {"llama-3.2-1b": "#111111", "gemma-3-1b": "#555555", "qwen3-0.6b": "#bbbbbb", "qwen2.5-0.5b": "#eeeeee"}
HATCH = {"llama-3.2-1b": "", "gemma-3-1b": "//", "qwen3-0.6b": "..", "qwen2.5-0.5b": "xx"}
SRC = "Source: our measurements, stabilarity/hub/research/llama-vs-qwen-vs-gemma-cpu-benchmark (Nov 2026)"

if "--run" in sys.argv:
    for k in ORDER:
        subprocess.run([sys.executable, os.path.join(HERE, "bench.py"), k, "--items", "30"], check=True)


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


raw = {k: json.load(open(os.path.join(HERE, "raw", f"{k}.json"))) for k in ORDER}
models = {}
for k in ORDER:
    r = raw[k]
    acc = {}
    tot_k = tot_n = 0
    for t, _ in TASKS:
        a = r["accuracy"][t]
        kk, n = a["correct_norm"], a["n"]
        tot_k += kk
        tot_n += n
        acc[t] = dict(n=n, correct_norm=kk, acc_norm=round(kk / n, 4), ci95=wilson(kk, n),
                      correct_raw=a["correct"], acc_raw=round(a["correct"] / n, 4))
    pooled = tot_k / tot_n
    acc["pooled"] = dict(n=tot_n, correct_norm=tot_k, acc_norm=round(pooled, 4), ci95=wilson(tot_k, tot_n))
    s6 = next(s for s in r["speed"] if s["threads"] == 6)
    s2 = next(s for s in r["speed"] if s["threads"] == 2)
    peak_gb = r["peak_rss_mb"] / 1024  # GiB
    file_gb = r["file_bytes"] / 1e9
    kv = r["kv"]
    above = pooled - CHANCE
    nonemb = r["params"] - r["embedding_params"]
    models[k] = dict(
        label=r["label"], family=r["family"], repo=r["repo"], revision=r["rev"], official_repo=r["official"],
        license=r["license"], sha256=r["sha256"], params=r["params"], embedding_params=r["embedding_params"],
        non_embedding_params=nonemb, embedding_share=round(r["embedding_params"] / r["params"], 4),
        file_bytes=r["file_bytes"], file_gb=round(file_gb, 3),
        rss_loaded_mb=r["rss_loaded_mb"], peak_rss_mb=r["peak_rss_mb"], peak_rss_gib=round(peak_gb, 3),
        load_s=r["load_s"], accuracy=acc,
        speed=dict(t6=s6, t2=s2,
                   decode_speedup_6_vs_2=round(s6["decode_tok_s"] / s2["decode_tok_s"], 3),
                   prefill_speedup_6_vs_2=round(s6["prefill_tok_s"] / s2["prefill_tok_s"], 3),
                   est_3x2thread_decode_tok_s=round(3 * s2["decode_tok_s"], 2)),
        kv=dict(**kv, kib_per_token_full=round(kv["bytes_per_token_full"] / 1024, 2),
                mib_at_4k=round(kv["bytes_at_4k"] / 2**20, 1), mib_at_32k=round(kv["bytes_at_32k"] / 2**20, 1),
                measured_bytes_at_512=s6["kv_bytes_after_prefill"],
                formula_bytes_at_512=kv["bytes_at_512"],
                kv_32k_vs_weights=round(kv["bytes_at_32k"] / r["file_bytes"], 3)),
        derived=dict(
            pooled_acc_per_peak_gib=round(pooled / peak_gb, 4),
            points_above_chance_per_peak_gib=round(100 * above / peak_gb, 2),
            points_above_chance_per_billion_params=round(100 * above / (r["params"] / 1e9), 2),
            points_above_chance_per_billion_nonemb_params=round(100 * above / (nonemb / 1e9), 2),
            seconds_per_100_tokens_6t=round(100 / s6["decode_tok_s"], 2),
            correct_answers_per_decode_second=round(pooled * s6["decode_tok_s"], 3),
        ),
        checks=dict(shared_kv_vs_naive=r["check_shared_kv_vs_naive"], tensors_skipped=r["tensors_skipped"]),
        scored_tokens=r["scored_tokens"], scoring_s=r["scoring_s"], wall_s=r["wall_s"],
    )

res = dict(
    title="Llama vs Qwen vs Gemma on a CPU",
    setup=dict(cpu="6 vCPU AMD EPYC (AVX2, no AVX-512/AMX), 2.0 GHz, KVM guest", dtype="bfloat16",
               torch=raw[ORDER[0]]["torch"], transformers="5.2",
               items_per_task=raw[ORDER[0]]["items_per_task"],
               tasks="first N items of ARC-Easy test, ARC-Challenge test, HellaSwag validation",
               scoring="0-shot log-likelihood, length-normalised by bytes (acc_norm), no chat template",
               speed="512-token prompt prefill, then 128 greedy decode steps, threads=6 and threads=2"),
    models=models,
    total_bench_wall_s=round(sum(raw[k]["wall_s"] for k in ORDER), 1),
)

# simple rankings for the text
res["rank_pooled_acc"] = sorted(ORDER, key=lambda k: -models[k]["accuracy"]["pooled"]["acc_norm"])
res["rank_decode_6t"] = sorted(ORDER, key=lambda k: -models[k]["speed"]["t6"]["decode_tok_s"])
res["rank_kv_32k"] = sorted(ORDER, key=lambda k: -models[k]["kv"]["bytes_at_32k"])

os.makedirs(os.path.join(HERE, "charts"), exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#555555",
                     "axes.labelcolor": "#111111", "xtick.color": "#555555", "ytick.color": "#555555"})


def finish(fig, ax_list, name, title):
    fig.suptitle(title, fontsize=14, color="#000000", x=0.01, ha="left", y=0.98)
    for ax in ax_list:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", lw=0.8)
        ax.set_axisbelow(True)
    fig.text(0.01, 0.01, SRC, fontsize=8.5, color="#555555")
    fig.savefig(os.path.join(HERE, "charts", name), dpi=160, facecolor="white")
    plt.close(fig)


# chart1: accuracy with Wilson CIs
fig, ax = plt.subplots(figsize=(10, 5.6))
groups = TASKS + [("pooled", "All 3 pooled")]
w = 0.2
chart1 = {}
for i, k in enumerate(ORDER):
    xs, ys, lo, hi = [], [], [], []
    for j, (t, _) in enumerate(groups):
        a = models[k]["accuracy"][t]
        xs.append(j + (i - 1.5) * w)
        ys.append(100 * a["acc_norm"])
        lo.append(100 * (a["acc_norm"] - a["ci95"][0]))
        hi.append(100 * (a["ci95"][1] - a["acc_norm"]))
    chart1[k] = ys
    ax.bar(xs, ys, w * 0.92, color=GREYS[k], edgecolor="#111111", lw=0.6, hatch=HATCH[k], label=models[k]["label"])
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#000000", lw=0.9, capsize=2.5)
    for x, y, h in zip(xs, ys, hi):
        ax.text(x, y + h + 1.2, f"{y:.0f}", ha="center", va="bottom", fontsize=8.5, color="#111111")
ax.axhline(100 * CHANCE, color="#555555", ls="--", lw=1)
ax.set_xticks(range(len(groups)), [g[1] for g in groups])
ax.set_ylabel("Accuracy, length-normalised (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.06), fontsize=9.5)
fig.subplots_adjust(top=0.85, bottom=0.12, left=0.08, right=0.98)
n = res["setup"]["items_per_task"]
finish(fig, [ax], "chart1.png", f"Zero-shot accuracy, {n} items per task (whiskers: 95% Wilson intervals; dashed: chance)")

# chart2: speed
fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
chart2 = {}
for ax, metric, lab in [(axs[0], "prefill_tok_s", "Prefill, 512-token prompt (tokens/s)"),
                        (axs[1], "decode_tok_s", "Decode, 128 new tokens (tokens/s)")]:
    for i, k in enumerate(ORDER):
        v2 = models[k]["speed"]["t2"][metric]
        v6 = models[k]["speed"]["t6"][metric]
        chart2.setdefault(k, {})[metric] = [v2, v6]
        for j, v in enumerate([v2, v6]):
            x = j + (i - 1.5) * w
            ax.bar(x, v, w * 0.92, color=GREYS[k], edgecolor="#111111", lw=0.6, hatch=HATCH[k],
                   label=models[k]["label"] if j == 0 else None)
            ax.text(x, v, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="#111111")
    ax.set_xticks([0, 1], ["2 threads", "6 threads"])
    ax.set_ylabel(lab)
axs[0].legend(frameon=False, fontsize=9)
fig.subplots_adjust(top=0.86, bottom=0.12, left=0.07, right=0.98, wspace=0.25)
finish(fig, axs, "chart2.png", "CPU speed in bfloat16: prefill and greedy decode at 2 and 6 threads")

# chart3: KV cache
fig, ax = plt.subplots(figsize=(10, 5.4))
ctxs = [("mib_at_4k", "4k context"), ("mib_at_32k", "32k context")]
chart3 = {}
for i, k in enumerate(ORDER):
    for j, (m, _) in enumerate(ctxs):
        v = models[k]["kv"][m]
        chart3.setdefault(k, []).append(v)
        x = j + (i - 1.5) * w
        ax.bar(x, v, w * 0.92, color=GREYS[k], edgecolor="#111111", lw=0.6, hatch=HATCH[k],
               label=models[k]["label"] + f" ({models[k]['kv']['kib_per_token_full']:g} KiB/token)" if j == 0 else None)
        ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, color="#111111")
ax.set_xticks([0, 1], [c[1] for c in ctxs])
ax.set_ylabel("KV cache for one sequence, bf16 (MiB)")
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.98)
finish(fig, [ax], "chart3.png", "KV cache size from model configs (Gemma: 512-token sliding window on local layers)")

# chart4: accuracy vs RAM
fig, ax = plt.subplots(figsize=(10, 5.6))
chart4 = {}
for k in ORDER:
    x = models[k]["peak_rss_gib"]
    a = models[k]["accuracy"]["pooled"]
    y = 100 * a["acc_norm"]
    chart4[k] = [x, y]
    ax.errorbar([x], [y], yerr=[[y - 100 * a["ci95"][0]], [100 * a["ci95"][1] - y]], fmt="o", ms=10,
                color="#111111", mfc=GREYS[k], mec="#111111", capsize=3, lw=0.9)
    off = {"qwen2.5-0.5b": (-12, -48), "qwen3-0.6b": (-14, 58), "gemma-3-1b": (14, -78), "llama-3.2-1b": (14, -4)}[k]
    ax.annotate(f"{models[k]['label']}\n{models[k]['derived']['points_above_chance_per_peak_gib']:g} pts above chance per GiB",
                (x, y), xytext=off, textcoords="offset points", fontsize=9, color="#111111",
                ha="right" if off[0] < 0 else "left", bbox=dict(fc="#ffffff", ec="none", pad=1.5))
ax.axhline(100 * CHANCE, color="#555555", ls="--", lw=1)
ax.set_xlabel("Peak resident memory of the benchmark process (GiB)")
ax.set_ylabel("Pooled accuracy, 3 tasks (%)")
ax.set_xlim(0, max(v[0] for v in chart4.values()) * 1.45)
ax.set_ylim(15, 80)
fig.subplots_adjust(top=0.9, bottom=0.12, left=0.08, right=0.98)
finish(fig, [ax], "chart4.png", "What each gibibyte of RAM buys: pooled accuracy vs peak memory")

res["charts"] = dict(chart1_acc_norm_pct=chart1, chart2_tok_s_2t_6t=chart2, chart3_kv_mib_4k_32k=chart3,
                     chart4_peakgb_accpct=chart4)
with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: dict(acc=models[k]["accuracy"]["pooled"], dec6=models[k]["speed"]["t6"]["decode_tok_s"],
                          rss=models[k]["peak_rss_gib"]) for k in ORDER}, indent=1))
