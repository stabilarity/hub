"""
The Retraining Treadmill: How Fast Enterprises Must Re-Skill Teams to Keep Pace with Model Churn
Analysis script generating charts and metrics for the article.

Reproduce: python3 analysis.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set up reproducible random state
rng = np.random.default_rng(42)

# Synthetic data: Foundation model release cadence (major releases per year, 2020-2025)
years = np.arange(2020, 2026)
model_releases = np.array([4, 6, 12, 20, 35, 50])  # Accelerating release cadence

# Enterprise training program half-life (months)
# Training becomes outdated as new models arrive
training_half_life = np.array([18, 15, 12, 9, 7, 5])  # Decreasing half-life

# Cost of capability gap (synthetic: millions USD per year per 1000 employees)
# Gap = model releases / (12 / half_life) = ratio of model pace to training retention
gap_ratio = model_releases / (12 / training_half_life)
capability_gap_cost = gap_ratio * rng.normal(2.5, 0.3, size=len(years))  # Base cost ~$2.5M per unit gap

# Cumulative retraining spend (millions USD)
annual_retraining_spend = np.cumsum(gap_ratio * rng.normal(1.8, 0.2, size=len(years)))

# Model capability improvement (benchmark score, normalized 0-100)
model_capability = np.array([35, 42, 55, 68, 78, 85])

# Enterprise adoption lag (months behind frontier)
adoption_lag = np.array([3, 4, 5, 6, 8, 10])

# Chart 1: Model Release Cadence vs Training Half-Life
fig1, ax1 = plt.subplots(figsize=(12, 8))
ax2 = ax1.twinx()
color1 = '#2E86AB'
color2 = '#A23B72'
ax1.bar(years - 0.2, model_releases, width=0.4, label='Model Releases/Year', color=color1, alpha=0.8)
ax2.plot(years, training_half_life, 'o-', color=color2, linewidth=2.5, markersize=8, label='Training Half-Life (months)')
ax1.set_xlabel('Year', fontsize=14)
ax1.set_ylabel('Model Releases per Year', fontsize=14, color=color1)
ax2.set_ylabel('Training Half-Life (months)', fontsize=14, color=color2)
ax1.set_title('Foundation Model Release Cadence vs. Enterprise Training Half-Life (2020–2025)', fontsize=16, pad=15)
ax1.tick_params(axis='y', labelcolor=color1)
ax2.tick_params(axis='y', labelcolor=color2)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./charts/chart1.png', dpi=100)
plt.close()

# Chart 2: Capability Gap Cost and Cumulative Retraining Spend
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax3.bar(years, capability_gap_cost, color='#F18F01', alpha=0.85, edgecolor='white', linewidth=0.5)
ax3.set_ylabel('Annual Capability Gap Cost ($M per 1K employees)', fontsize=13)
ax3.set_title('Annual Cost of Model-Training Capability Gap', fontsize=15, pad=12)
ax3.grid(True, alpha=0.3, axis='y')

ax4.plot(years, annual_retraining_spend, 'o-', color='#2E86AB', linewidth=3, markersize=8)
ax4.fill_between(years, 0, annual_retraining_spend, color='#2E86AB', alpha=0.15)
ax4.set_xlabel('Year', fontsize=13)
ax4.set_ylabel('Cumulative Retraining Spend ($M per 1K employees)', fontsize=13)
ax4.set_title('Cumulative Enterprise Retraining Investment', fontsize=15, pad=12)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./charts/chart2.png', dpi=100)
plt.close()

# Chart 3: Model Capability vs Enterprise Adoption Lag
fig3, ax5 = plt.subplots(figsize=(12, 8))
ax6 = ax5.twinx()
color3 = '#2E86AB'
color4 = '#C73E1D'
ax5.plot(years, model_capability, 's-', color=color3, linewidth=3, markersize=10, label='Frontier Model Capability')
ax6.bar(years + 0.15, adoption_lag, width=0.3, color=color4, alpha=0.8, label='Enterprise Adoption Lag (months)')
ax5.set_xlabel('Year', fontsize=14)
ax5.set_ylabel('Model Capability (Benchmark Score 0–100)', fontsize=14, color=color3)
ax6.set_ylabel('Adoption Lag (months)', fontsize=14, color=color4)
ax5.set_title('Frontier Model Capability vs. Enterprise Adoption Lag', fontsize=16, pad=15)
ax5.tick_params(axis='y', labelcolor=color3)
ax6.tick_params(axis='y', labelcolor=color4)
lines5, labels5 = ax5.get_legend_handles_labels()
lines6, labels6 = ax6.get_legend_handles_labels()
ax5.legend(lines5 + lines6, labels5 + labels6, loc='upper left', fontsize=12)
ax5.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./charts/chart3.png', dpi=100)
plt.close()

# Chart 4: Projected retraining frequency needed to keep pace
# If models release every N months, training must complete in < N months
months_per_release = 12 / model_releases
required_training_cycle = months_per_release * 0.7  # Need to complete before next major release
current_training_cycle = training_half_life * 1.5  # Typical full cycle ~1.5x half-life

fig4, ax7 = plt.subplots(figsize=(12, 8))
x_pos = np.arange(len(years))
width = 0.35
ax7.bar(x_pos - width/2, current_training_cycle, width, label='Current Training Cycle (months)', color='#A23B72', alpha=0.85)
ax7.bar(x_pos + width/2, required_training_cycle, width, label='Required Training Cycle (months)', color='#2E86AB', alpha=0.85)
ax7.set_xticks(x_pos)
ax7.set_xticklabels(years)
ax7.set_xlabel('Year', fontsize=14)
ax7.set_ylabel('Training Cycle Duration (months)', fontsize=14)
ax7.set_title('Current vs. Required Retraining Cycle to Keep Pace with Model Churn', fontsize=16, pad=15)
ax7.legend(fontsize=12)
ax7.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('./charts/chart4.png', dpi=100)
plt.close()

# Chart 5: Cumulative capability debt (gap * lag)
capability_debt = np.cumsum(gap_ratio * adoption_lag * rng.normal(0.8, 0.1, size=len(years)))

fig5, ax8 = plt.subplots(figsize=(12, 8))
ax8.plot(years, capability_debt, 'o-', color='#C73E1D', linewidth=3, markersize=10)
ax8.fill_between(years, 0, capability_debt, color='#C73E1D', alpha=0.15)
ax8.set_xlabel('Year', fontsize=14)
ax8.set_ylabel('Cumulative Capability Debt (arbitrary units)', fontsize=14)
ax8.set_title('Accumulating Capability Debt: The Retraining Treadmill Effect', fontsize=16, pad=15)
ax8.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./charts/chart5.png', dpi=100)
plt.close()

# Persist metrics to results.json
results = {
    "years": years.tolist(),
    "model_releases_per_year": model_releases.tolist(),
    "training_half_life_months": training_half_life.tolist(),
    "gap_ratio": gap_ratio.tolist(),
    "annual_capability_gap_cost_millions_usd_per_1k": capability_gap_cost.tolist(),
    "cumulative_retraining_spend_millions_usd_per_1k": annual_retraining_spend.tolist(),
    "model_capability_score": model_capability.tolist(),
    "enterprise_adoption_lag_months": adoption_lag.tolist(),
    "current_training_cycle_months": current_training_cycle.tolist(),
    "required_training_cycle_months": required_training_cycle.tolist(),
    "cumulative_capability_debt": capability_debt.tolist()
}

with open('./results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Analysis complete. Charts saved to ./charts/, metrics to ./results.json")
