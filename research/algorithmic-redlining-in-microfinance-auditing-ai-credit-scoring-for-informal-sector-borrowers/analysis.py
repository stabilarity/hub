"""
Algorithmic Redlining in Microfinance: Auditing AI Credit Scoring for Informal-Sector Borrowers
Bias audit of AI-driven microfinance credit models against borrowers lacking formal credit histories.

Reproduce: python3 analysis.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
rng = np.random.default_rng(42)

# Output directories
charts_dir = Path("./charts")
charts_dir.mkdir(exist_ok=True)
results_path = Path("./results.json")

# ------------------------------------------------------------
# Synthetic data generation based on public research findings
# ------------------------------------------------------------
# Regions studied in microfinance bias literature
regions = ["Kenya", "India", "Pakistan", "Cambodia", "Nigeria", "Bangladesh"]

# Simulated borrower characteristics
n_borrowers = 5000
borrower_data = {
    "region": rng.choice(regions, n_borrowers, p=[0.2, 0.2, 0.15, 0.1, 0.2, 0.15]),
    "has_formal_history": rng.choice([0, 1], n_borrowers, p=[0.65, 0.35]),  # 65% lack formal credit history
    "income_usd": rng.lognormal(mean=6.5, sigma=0.8, size=n_borrowers),  # Monthly income ~$670 median
    "loan_amount_usd": rng.lognormal(mean=5.5, sigma=0.6, size=n_borrowers),  # Loan amount ~$245 median
    "repayment_rate": rng.beta(8, 2, n_borrowers),  # High repayment generally
    "digital_footprint_score": rng.beta(2, 5, n_borrowers),  # Low digital footprint for informal sector
    "gender": rng.choice(["M", "F"], n_borrowers, p=[0.55, 0.45]),
    "age": rng.integers(18, 65, n_borrowers),
}

df = pd.DataFrame(borrower_data)

# Simulate AI credit scoring model predictions
# Model uses: formal_history (high weight), digital_footprint, income, loan_amount
# Bias: underweights informal borrowers (no formal history) despite similar repayment

def compute_ai_score(row):
    """Simulated AI credit score (0-100)"""
    base = 50
    if row["has_formal_history"]:
        base += 25 * row["digital_footprint_score"]
    else:
        base += 10 * row["digital_footprint_score"]  # Penalized for lack of formal history
    base += 15 * (row["income_usd"] / 2000)
    base -= 10 * (row["loan_amount_usd"] / 500)
    base += rng.normal(0, 5)  # Model noise
    return np.clip(base, 0, 100)

df["ai_score"] = df.apply(compute_ai_score, axis=1)
df["approved"] = (df["ai_score"] >= 60).astype(int)
df["defaulted"] = (df["repayment_rate"] < 0.5).astype(int)  # Simplified default definition

# ------------------------------------------------------------
# Compute metrics for results.json
# ------------------------------------------------------------
results = {}

# 1. Approval rates by formal credit history
approval_by_history = df.groupby("has_formal_history")["approved"].mean()
results["approval_rate_no_formal_history"] = float(approval_by_history[0])
results["approval_rate_with_formal_history"] = float(approval_by_history[1])
results["approval_gap"] = float(approval_by_history[1] - approval_by_history[0])

# 2. Default rates by formal credit history (among approved)
approved_df = df[df["approved"] == 1]
default_by_history = approved_df.groupby("has_formal_history")["defaulted"].mean()
results["default_rate_no_formal_history"] = float(default_by_history[0])
results["default_rate_with_formal_history"] = float(default_by_history[1])

# 3. Regional disparity in approval rates
regional_approval = df.groupby("region")["approved"].mean().to_dict()
results["regional_approval_rates"] = {k: float(v) for k, v in regional_approval.items()}

# 4. Gender disparity in approval rates (among those without formal history)
no_hist = df[df["has_formal_history"] == 0]
gender_approval = no_hist.groupby("gender")["approved"].mean().to_dict()
results["gender_approval_no_history"] = {k: float(v) for k, v in gender_approval.items()}

# 5. AI score distribution statistics
results["ai_score_stats"] = {
    "mean": float(df["ai_score"].mean()),
    "std": float(df["ai_score"].std()),
    "median": float(df["ai_score"].median()),
    "p25": float(df["ai_score"].quantile(0.25)),
    "p75": float(df["ai_score"].quantile(0.75)),
}

# 6. Calibration: predicted vs actual default by score decile
df["score_decile"] = pd.qcut(df["ai_score"], 10, labels=False)
calibration = df.groupby("score_decile").agg(
    mean_score=("ai_score", "mean"),
    default_rate=("defaulted", "mean"),
    count=("ai_score", "count")
).reset_index()
results["calibration"] = calibration.to_dict(orient="records")

# Save results
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {results_path}")

# ------------------------------------------------------------
# Chart 1: Approval Rate by Formal Credit History & Region
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

# Grouped bar chart
x = np.arange(len(regions))
width = 0.35

no_hist_rates = []
with_hist_rates = []

for region in regions:
    subset = df[df["region"] == region]
    no_hist_rates.append(subset[subset["has_formal_history"] == 0]["approved"].mean())
    with_hist_rates.append(subset[subset["has_formal_history"] == 1]["approved"].mean())

bars1 = ax.bar(x - width/2, no_hist_rates, width, label="No Formal Credit History", color="#e74c3c", alpha=0.85)
bars2 = ax.bar(x + width/2, with_hist_rates, width, label="Has Formal Credit History", color="#2ecc71", alpha=0.85)

ax.set_xlabel("Region", fontsize=14)
ax.set_ylabel("Loan Approval Rate", fontsize=14)
ax.set_title("AI Credit Approval Rates by Region and Formal Credit History Status", fontsize=16, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(regions, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1%}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(charts_dir / "chart1.png", dpi=100, bbox_inches="tight")
plt.close()
print("Chart 1 saved")

# ------------------------------------------------------------
# Chart 2: AI Score Distribution by Formal Credit History
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

no_hist_scores = df[df["has_formal_history"] == 0]["ai_score"]
with_hist_scores = df[df["has_formal_history"] == 1]["ai_score"]

ax.hist(no_hist_scores, bins=40, alpha=0.6, label="No Formal Credit History (n={:,})".format(len(no_hist_scores)),
        color="#e74c3c", density=True, edgecolor="white", linewidth=0.5)
ax.hist(with_hist_scores, bins=40, alpha=0.6, label="Has Formal Credit History (n={:,})".format(len(with_hist_scores)),
        color="#2ecc71", density=True, edgecolor="white", linewidth=0.5)

# Add approval threshold line
ax.axvline(x=60, color="#34495e", linestyle="--", linewidth=2, label="Approval Threshold (Score ≥ 60)")

ax.set_xlabel("AI Credit Score", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.set_title("Distribution of AI Credit Scores by Formal Credit History Status", fontsize=16, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(axis="y", alpha=0.3)

# Add mean lines
ax.axvline(no_hist_scores.mean(), color="#e74c3c", linestyle=":", linewidth=2, alpha=0.8)
ax.axvline(with_hist_scores.mean(), color="#2ecc71", linestyle=":", linewidth=2, alpha=0.8)

plt.tight_layout()
plt.savefig(charts_dir / "chart2.png", dpi=100, bbox_inches="tight")
plt.close()
print("Chart 2 saved")

# ------------------------------------------------------------
# Chart 3: Calibration Curve (Predicted vs Actual Default Rate)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

# Perfect calibration line
ax.plot([0, 100], [0, 1], "k--", alpha=0.4, label="Perfect Calibration")

# Model calibration
cal_scores = [c["mean_score"] for c in results["calibration"]]
cal_defaults = [c["default_rate"] for c in results["calibration"]]
cal_counts = [c["count"] for c in results["calibration"]]

# Size bubbles by count
sizes = np.array(cal_counts) / max(cal_counts) * 500 + 50
scatter = ax.scatter(cal_scores, cal_defaults, s=sizes, alpha=0.7, c=cal_scores, cmap="RdYlGn_r",
                     edgecolors="white", linewidth=1, label="Score Deciles (bubble size = sample count)")

# Add decile labels
for i, (score, default, count) in enumerate(zip(cal_scores, cal_defaults, cal_counts)):
    ax.annotate(f"D{i+1}\n(n={count})", (score, default), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

ax.set_xlabel("Mean AI Credit Score (by Decile)", fontsize=14)
ax.set_ylabel("Actual Default Rate", fontsize=14)
ax.set_title("Calibration Curve: Predicted Risk vs. Actual Default by Score Decile", fontsize=16, fontweight="bold")
ax.legend(fontsize=11, loc="upper left")
ax.grid(alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, max(cal_defaults) * 1.3 if max(cal_defaults) > 0 else 0.3)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Mean Score", fontsize=12)

plt.tight_layout()
plt.savefig(charts_dir / "chart3.png", dpi=100, bbox_inches="tight")
plt.close()
print("Chart 3 saved")

# ------------------------------------------------------------
# Chart 4: Gender Disparity in Approval (No Formal History)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

gender_data = no_hist.groupby("gender")["approved"].agg(["mean", "count"]).reset_index()
colors = ["#3498db", "#e91e63"]

bars = ax.bar(gender_data["gender"], gender_data["mean"], color=colors, alpha=0.85,
              edgecolor="white", linewidth=1.5, width=0.5)

for i, (bar, row) in enumerate(zip(bars, gender_data.itertuples())):
    ax.annotate(f"{row.mean:.1%}\n(n={row.count:,})",
                xy=(bar.get_x() + bar.get_width() / 2, row.mean),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_xlabel("Gender", fontsize=14)
ax.set_ylabel("Approval Rate (No Formal Credit History)", fontsize=14)
ax.set_title("Gender Disparity in AI Loan Approval Among Informal-Sector Borrowers", fontsize=16, fontweight="bold")
ax.set_ylim(0, max(gender_data["mean"]) * 1.4)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(charts_dir / "chart4.png", dpi=100, bbox_inches="tight")
plt.close()
print("Chart 4 saved")

# ------------------------------------------------------------
# Chart 5: Regional Approval Gap (With vs Without Formal History)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

gaps = []
for region in regions:
    subset = df[df["region"] == region]
    gap = (subset[subset["has_formal_history"] == 1]["approved"].mean() -
           subset[subset["has_formal_history"] == 0]["approved"].mean())
    gaps.append(gap)

colors = ["#e74c3c" if g > np.mean(gaps) else "#f39c12" if g > 0 else "#2ecc71" for g in gaps]
bars = ax.barh(regions, gaps, color=colors, alpha=0.85, edgecolor="white", height=0.6)

for bar, gap in zip(bars, gaps):
    ax.annotate(f"{gap:.1%}", xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                xytext=(5 if gap >= 0 else -5, 0), textcoords="offset points",
                ha="left" if gap >= 0 else "right", va="center", fontsize=11, fontweight="bold")

ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Approval Rate Gap (With History - Without History)", fontsize=14)
ax.set_title("Regional Disparity: Approval Gap Between Borrowers With vs. Without Formal Credit History", fontsize=16, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

plt.tight_layout()
plt.savefig(charts_dir / "chart5.png", dpi=100, bbox_inches="tight")
plt.close()
print("Chart 5 saved")

print("\nAll charts generated successfully!")
print(f"Results JSON: {results_path}")
