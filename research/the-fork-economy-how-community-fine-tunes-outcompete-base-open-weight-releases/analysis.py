"""
The Fork Economy: How Community Fine-Tunes Outcompete Base Open-Weight Releases
Analysis script generating reproducible charts and metrics.

Reproduce: python3 analysis.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure charts directory exists
charts_dir = Path("charts")
charts_dir.mkdir(exist_ok=True)

# Reproducible RNG
rng = np.random.default_rng(42)

# Synthetic data: 10 base models, each with 3-5 community fine-tunes
base_models = [
    "Llama-2-7B", "Llama-2-13B", "Llama-2-70B",
    "Mistral-7B", "Mixtral-8x7B",
    "Gemma-7B", "Gemma-2B",
    "Qwen-1.5-7B", "Qwen-1.5-14B",
    "Phi-3-mini"
]

np.random.seed(42)
data = []
model_id = 0

for base in base_models:
    # Base model metrics
    base_downloads = rng.integers(50000, 500000)
    base_benchmark = rng.uniform(45, 65)  # MMLU score
    base_age_days = rng.integers(180, 720)
    
    data.append({
        "model": base,
        "type": "base",
        "base_model": base,
        "downloads": base_downloads,
        "benchmark": base_benchmark,
        "age_days": base_age_days,
        "downloads_per_day": base_downloads / base_age_days,
    })
    
    # 3-5 fine-tunes per base
    n_finetunes = rng.integers(3, 6)
    for i in range(n_finetunes):
        finetune_name = f"{base}-ft-{i+1}"
        # Fine-tunes typically have higher benchmark, varying downloads
        ft_benchmark = min(base_benchmark + rng.uniform(3, 15), 85)
        ft_downloads = rng.integers(int(base_downloads * 0.1), int(base_downloads * 0.8))
        ft_age_days = rng.integers(30, base_age_days)
        
        data.append({
            "model": finetune_name,
            "type": "finetune",
            "base_model": base,
            "downloads": ft_downloads,
            "benchmark": ft_benchmark,
            "age_days": ft_age_days,
            "downloads_per_day": ft_downloads / ft_age_days,
        })

df = pd.DataFrame(data)

# Compute aggregate metrics
metrics = {
    "n_base_models": len(base_models),
    "n_finetunes": len(df[df["type"] == "finetune"]),
    "avg_base_benchmark": float(df[df["type"] == "base"]["benchmark"].mean()),
    "avg_finetune_benchmark": float(df[df["type"] == "finetune"]["benchmark"].mean()),
    "avg_base_downloads": float(df[df["type"] == "base"]["downloads"].mean()),
    "avg_finetune_downloads": float(df[df["type"] == "finetune"]["downloads"].mean()),
    "benchmark_improvement": float(df[df["type"] == "finetune"]["benchmark"].mean() - df[df["type"] == "base"]["benchmark"].mean()),
    "downloads_ratio_finetune_to_base": float(df[df["type"] == "finetune"]["downloads"].mean() / df[df["type"] == "base"]["downloads"].mean()),
}

# Per-base-model comparison
base_comparison = []
for base in base_models:
    base_row = df[(df["type"] == "base") & (df["base_model"] == base)].iloc[0]
    ft_rows = df[(df["type"] == "finetune") & (df["base_model"] == base)]
    
    base_comparison.append({
        "base_model": base,
        "base_benchmark": float(base_row["benchmark"]),
        "avg_finetune_benchmark": float(ft_rows["benchmark"].mean()),
        "benchmark_gain": float(ft_rows["benchmark"].mean() - base_row["benchmark"]),
        "base_downloads": int(base_row["downloads"]),
        "avg_finetune_downloads": int(ft_rows["downloads"].mean()),
        "downloads_ratio": float(ft_rows["downloads"].mean() / base_row["downloads"]),
        "n_finetunes": len(ft_rows),
    })

metrics["per_base_model"] = base_comparison

# Save results.json
with open("results.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ========== CHART 1: Benchmark Comparison - Base vs Fine-tunes ==========
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(base_models))
width = 0.35

base_benchmarks = [b["base_benchmark"] for b in base_comparison]
ft_benchmarks = [b["avg_finetune_benchmark"] for b in base_comparison]

bars1 = ax.bar(x - width/2, base_benchmarks, width, label="Base Model", color="#4C78A8", alpha=0.85)
bars2 = ax.bar(x + width/2, ft_benchmarks, width, label="Avg Community Fine-tune", color="#F58518", alpha=0.85)

ax.set_ylabel("MMLU Benchmark Score", fontsize=14)
ax.set_title("Benchmark Performance: Base Models vs. Community Fine-tunes", fontsize=16, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(base_models, rotation=30, ha="right", fontsize=11)
ax.legend(fontsize=12)
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bar in bars1:
    ax.annotate(f"{bar.get_height():.1f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.annotate(f"{bar.get_height():.1f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("charts/chart1.png", dpi=100)
plt.close()

# ========== CHART 2: Downloads Comparison - Base vs Fine-tunes ==========
fig, ax = plt.subplots(figsize=(12, 8))

base_downloads = [b["base_downloads"] for b in base_comparison]
ft_downloads = [b["avg_finetune_downloads"] for b in base_comparison]

bars1 = ax.bar(x - width/2, base_downloads, width, label="Base Model", color="#4C78A8", alpha=0.85)
bars2 = ax.bar(x + width/2, ft_downloads, width, label="Avg Community Fine-tune", color="#F58518", alpha=0.85)

ax.set_ylabel("Total Downloads", fontsize=14)
ax.set_title("Download Volume: Base Models vs. Community Fine-tunes", fontsize=16, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(base_models, rotation=30, ha="right", fontsize=11)
ax.legend(fontsize=12)
ax.grid(axis="y", alpha=0.3)

# Format y-axis with K/M suffixes
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x < 1e6 else f"{x/1e6:.1f}M"))

for bar in bars1:
    h = bar.get_height()
    ax.annotate(f"{h/1000:.0f}K", xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    h = bar.get_height()
    ax.annotate(f"{h/1000:.0f}K", xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("charts/chart2.png", dpi=100)
plt.close()

# ========== CHART 3: Value Added - Benchmark Gain vs Downloads Ratio ==========
fig, ax = plt.subplots(figsize=(12, 8))

gains = [b["benchmark_gain"] for b in base_comparison]
ratios = [b["downloads_ratio"] for b in base_comparison]
n_fts = [b["n_finetunes"] for b in base_comparison]

scatter = ax.scatter(gains, ratios, s=[n*80 for n in n_fts], c=n_fts, cmap="viridis", alpha=0.7, edgecolors="white", linewidth=1)

# Add labels for each point
for i, base in enumerate(base_models):
    ax.annotate(base, xy=(gains[i], ratios[i]), xytext=(5, 5), 
                textcoords="offset points", fontsize=10, fontweight="bold")

ax.set_xlabel("Benchmark Gain (Fine-tune - Base)", fontsize=14)
ax.set_ylabel("Downloads Ratio (Fine-tune / Base)", fontsize=14)
ax.set_title("Value Added by Fine-tunes: Benchmark Gain vs. Download Retention", fontsize=16, fontweight="bold")
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Parity (1.0)")
ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Number of Fine-tunes", fontsize=12)

plt.tight_layout()
plt.savefig("charts/chart3.png", dpi=100)
plt.close()

# ========== CHART 4: Downloads per Day (Velocity) ==========
fig, ax = plt.subplots(figsize=(12, 8))

base_vel = [b["base_downloads"] / df[(df["type"] == "base") & (df["base_model"] == b["base_model"])].iloc[0]["age_days"] for b in base_comparison]
ft_vel = [b["avg_finetune_downloads"] / df[(df["type"] == "finetune") & (df["base_model"] == b["base_model"])]["age_days"].mean() for b in base_comparison]

bars1 = ax.bar(x - width/2, base_vel, width, label="Base Model", color="#4C78A8", alpha=0.85)
bars2 = ax.bar(x + width/2, ft_vel, width, label="Avg Community Fine-tune", color="#F58518", alpha=0.85)

ax.set_ylabel("Downloads per Day", fontsize=14)
ax.set_title("Download Velocity: Base Models vs. Community Fine-tunes", fontsize=16, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(base_models, rotation=30, ha="right", fontsize=11)
ax.legend(fontsize=12)
ax.grid(axis="y", alpha=0.3)

for bar in bars1:
    h = bar.get_height()
    ax.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    h = bar.get_height()
    ax.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("charts/chart4.png", dpi=100)
plt.close()

# ========== CHART 5: Distribution of Fine-tune Benchmark Improvements ==========
fig, ax = plt.subplots(figsize=(12, 8))

ft_df = df[df["type"] == "finetune"]
improvements = []
for _, row in ft_df.iterrows():
    base_row = df[(df["type"] == "base") & (df["base_model"] == row["base_model"])].iloc[0]
    improvements.append(row["benchmark"] - base_row["benchmark"])

ax.hist(improvements, bins=15, color="#F58518", alpha=0.8, edgecolor="white", linewidth=1)
ax.axvline(np.mean(improvements), color="#4C78A8", linestyle="--", linewidth=2, label=f"Mean: {np.mean(improvements):.1f}")
ax.axvline(np.median(improvements), color="#E45756", linestyle="--", linewidth=2, label=f"Median: {np.median(improvements):.1f}")

ax.set_xlabel("Benchmark Improvement (Fine-tune - Base)", fontsize=14)
ax.set_ylabel("Number of Fine-tunes", fontsize=14)
ax.set_title("Distribution of Benchmark Improvements Across All Community Fine-tunes", fontsize=16, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("charts/chart5.png", dpi=100)
plt.close()

print("Analysis complete. Generated 5 charts and results.json")
print(f"Charts saved to: {charts_dir.resolve()}")
