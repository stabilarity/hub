"""
Informal Labor Platforms and AI Wage Suppression: Evidence from Gig Marketplaces in 2025-2026
Reproduce: python3 analysis.py

Empirical study of algorithmic pricing on gig platforms and its measured effect on informal-sector wage floors.
Uses synthetic data with fixed seed for reproducibility.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Set seed for reproducibility
rng = np.random.default_rng(42)

# Ensure charts directory exists
charts_dir = Path("./charts")
charts_dir.mkdir(exist_ok=True)

# ============================================================
# Synthetic Data Generation (aligned with paper themes)
# ============================================================

# 1. Platform wage trends over time (2023-2026 monthly)
months = pd.date_range("2023-01", "2026-12", freq="ME")
n_months = len(months)

# Base wage trends: algorithmic platforms vs traditional informal
base_wage_algo = 15.0  # USD/hour starting point
base_wage_traditional = 12.0

# Algorithmic wage suppression effect: gradual decline
algo_trend = -0.03 * np.arange(n_months)  # -3 cents/month
traditional_trend = -0.01 * np.arange(n_months)  # -1 cent/month

# Add noise
algo_noise = rng.normal(0, 0.15, n_months)
trad_noise = rng.normal(0, 0.12, n_months)

wage_algo = base_wage_algo + algo_trend + algo_noise
wage_traditional = base_wage_traditional + traditional_trend + trad_noise

# 2. Cross-platform comparison (major gig platforms)
platforms = ["Platform A", "Platform B", "Platform C", "Platform D", "Platform E"]
platform_types = ["Algorithmic", "Algorithmic", "Hybrid", "Traditional", "Traditional"]

# Wage distributions per platform (2025-2026)
n_workers = 500
platform_wages = {}
for i, p in enumerate(platforms):
    if platform_types[i] == "Algorithmic":
        platform_wages[p] = rng.lognormal(mean=np.log(12), sigma=0.25, size=n_workers)
    elif platform_types[i] == "Hybrid":
        platform_wages[p] = rng.lognormal(mean=np.log(13.5), sigma=0.2, size=n_workers)
    else:
        platform_wages[p] = rng.lognormal(mean=np.log(14), sigma=0.18, size=n_workers)

# 3. Wage suppression vs platform adoption (regional panel)
regions = ["North America", "Western Europe", "Latin America", "Asia-Pacific", "Africa"]
n_regions = len(regions)
years = [2023, 2024, 2025, 2026]

regional_data = []
for r in regions:
    for y in years:
        adoption = rng.beta(2, 5) * 100  # platform adoption %
        # Wage suppression increases with adoption
        suppression = 0.5 + 0.3 * adoption/100 + rng.normal(0, 0.15)
        informal_wage = 10 + rng.normal(0, 1.5) - suppression
        regional_data.append({
            "region": r, "year": y,
            "platform_adoption_pct": adoption,
            "wage_suppression_usd": suppression,
            "informal_wage_usd": max(informal_wage, 3)
        })
regional_df = pd.DataFrame(regional_data)

# 4. Algorithmic pricing features correlation
n_obs = 1000
features = pd.DataFrame({
    "surge_multiplier": rng.uniform(1.0, 3.0, n_obs),
    "worker_rating": rng.uniform(3.5, 5.0, n_obs),
    "completion_rate": rng.uniform(0.7, 1.0, n_obs),
    "platform_fee_pct": rng.uniform(15, 30, n_obs),
    "task_complexity": rng.uniform(1, 5, n_obs),
})
# Wage outcome
features["effective_wage"] = (
    18 / features["surge_multiplier"] *
    (0.8 + 0.2 * features["worker_rating"] / 5) *
    (0.9 + 0.1 * features["completion_rate"]) *
    (1 - features["platform_fee_pct"] / 100) *
    (0.9 + 0.1 * features["task_complexity"] / 5) +
    rng.normal(0, 0.5, n_obs)
)

# ============================================================
# Chart 1: Wage Trends Over Time (Algorithmic vs Traditional)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(months, wage_algo, label="Algorithmic Platforms", color="#d62728", linewidth=2, alpha=0.9)
ax.plot(months, wage_traditional, label="Traditional Informal", color="#1f77b4", linewidth=2, alpha=0.9)
ax.axvline(pd.Timestamp("2025-01"), color="gray", linestyle="--", alpha=0.5, label="2025 Policy Shifts")
ax.set_title("Hourly Wage Trends: Algorithmic vs Traditional Informal Platforms (2023-2026)", fontsize=14, pad=15)
ax.set_ylabel("Hourly Wage (USD)", fontsize=12)
ax.set_xlabel("Month", fontsize=12)
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(charts_dir / "chart1.png", dpi=100)
plt.close()

# ============================================================
# Chart 2: Cross-Platform Wage Distribution (Box Plot)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
colors = ["#d62728", "#d62728", "#ff7f0e", "#1f77b4", "#1f77b4"]
bp = ax.boxplot(
    [platform_wages[p] for p in platforms],
    labels=platforms,
    patch_artist=True,
    showfliers=False,
    medianprops={"color": "black", "linewidth": 2}
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_title("Hourly Wage Distribution by Platform Type (2025-2026)", fontsize=14, pad=15)
ax.set_ylabel("Hourly Wage (USD)", fontsize=12)
ax.set_xlabel("Platform", fontsize=12)
# Add legend for platform types
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#d62728", alpha=0.6, label="Algorithmic"),
    Patch(facecolor="#ff7f0e", alpha=0.6, label="Hybrid"),
    Patch(facecolor="#1f77b4", alpha=0.6, label="Traditional")
]
ax.legend(handles=legend_elements, fontsize=11, framealpha=0.9, loc="upper right")
ax.grid(True, alpha=0.3, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(charts_dir / "chart2.png", dpi=100)
plt.close()

# ============================================================
# Chart 3: Platform Adoption vs Wage Suppression (Scatter with Trend)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
colors_reg = {"North America": "#1f77b4", "Western Europe": "#ff7f0e", 
              "Latin America": "#2ca02c", "Asia-Pacific": "#d62728", "Africa": "#9467bd"}
for r in regions:
    subset = regional_df[regional_df["region"] == r]
    ax.scatter(subset["platform_adoption_pct"], subset["wage_suppression_usd"],
               label=r, color=colors_reg[r], s=100, alpha=0.7, edgecolors="white", linewidth=0.5)

# Add trend line
x_trend = np.linspace(0, 100, 100)
y_trend = 0.5 + 0.3 * x_trend / 100
ax.plot(x_trend, y_trend, "k--", alpha=0.5, linewidth=1.5, label="Theoretical Trend")

ax.set_title("Platform Adoption vs. Wage Suppression by Region (2023-2026)", fontsize=14, pad=15)
ax.set_xlabel("Platform Adoption (%)", fontsize=12)
ax.set_ylabel("Wage Suppression (USD/hour)", fontsize=12)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(charts_dir / "chart3.png", dpi=100)
plt.close()

# ============================================================
# Chart 4: Feature Importance for Effective Wage (Correlation Heatmap)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
corr_matrix = features.corr()
im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(corr_matrix.columns, fontsize=10)

# Add correlation values
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                       ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white",
                       fontsize=9)

plt.colorbar(im, ax=ax, label="Correlation Coefficient")
ax.set_title("Correlation Matrix: Algorithmic Pricing Features vs Effective Wage", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(charts_dir / "chart4.png", dpi=100)
plt.close()

# ============================================================
# Chart 5: Regional Informal Wage Floor Trajectory
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
for r in regions:
    subset = regional_df[regional_df["region"] == r].sort_values("year")
    ax.plot(subset["year"], subset["informal_wage_usd"], marker="o", linewidth=2,
            label=r, color=colors_reg[r], markersize=8)

ax.set_title("Informal Sector Wage Floor by Region (2023-2026)", fontsize=14, pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Informal Wage Floor (USD/hour)", fontsize=12)
ax.set_xticks(years)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(charts_dir / "chart5.png", dpi=100)
plt.close()

# ============================================================
# Compute Metrics & Save Results
# ============================================================
metrics = {
    "wage_trends": {
        "algorithmic_2023_mean": float(np.mean(wage_algo[:12])),
        "algorithmic_2026_mean": float(np.mean(wage_algo[-12:])),
        "traditional_2023_mean": float(np.mean(wage_traditional[:12])),
        "traditional_2026_mean": float(np.mean(wage_traditional[-12:])),
        "algorithmic_decline_pct": float((np.mean(wage_algo[:12]) - np.mean(wage_algo[-12:])) / np.mean(wage_algo[:12]) * 100),
        "traditional_decline_pct": float((np.mean(wage_traditional[:12]) - np.mean(wage_traditional[-12:])) / np.mean(wage_traditional[:12]) * 100),
    },
    "platform_comparison": {
        p: {
            "mean_wage": float(np.mean(platform_wages[p])),
            "median_wage": float(np.median(platform_wages[p])),
            "p25": float(np.percentile(platform_wages[p], 25)),
            "p75": float(np.percentile(platform_wages[p], 75)),
            "type": platform_types[i]
        }
        for i, p in enumerate(platforms)
    },
    "regional_analysis": {
        r: {
            "avg_adoption_pct": float(regional_df[regional_df["region"] == r]["platform_adoption_pct"].mean()),
            "avg_suppression_usd": float(regional_df[regional_df["region"] == r]["wage_suppression_usd"].mean()),
            "wage_floor_2023": float(regional_df[(regional_df["region"] == r) & (regional_df["year"] == 2023)]["informal_wage_usd"].values[0]),
            "wage_floor_2026": float(regional_df[(regional_df["region"] == r) & (regional_df["year"] == 2026)]["informal_wage_usd"].values[0]),
        }
        for r in regions
    },
    "feature_correlations": {
        col: float(corr_matrix.loc[col, "effective_wage"])
        for col in corr_matrix.columns if col != "effective_wage"
    },
    "sample_sizes": {
        "monthly_observations": n_months,
        "workers_per_platform": n_workers,
        "regional_observations": len(regional_df),
        "feature_observations": n_obs
    }
}

with open("results.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Analysis complete. Charts saved to ./charts/")
print("Metrics saved to ./results.json")
