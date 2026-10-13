"""
Interpretable Models vs Post-Hoc Explanations: True Cost Comparison for Enterprise AI
Analysis script generating cost comparison charts using synthetic data.

Reproduce: python3 analysis.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set seed for reproducibility
rng = np.random.default_rng(42)

# Synthetic data: cost comparison between interpretable models and post-hoc explanations
# Based on enterprise AI scenarios: training, inference, explanation generation, maintenance

# Scenario parameters
n_scenarios = 5
scenarios = ['Low Complexity\n(10K samples)', 'Medium Complexity\n(100K samples)', 
             'High Complexity\n(1M samples)', 'Real-time\n(<10ms latency)', 
             'Regulated\n(Full audit trail)']

# Training costs (USD) - interpretable models cheaper to train, less tuning needed
interpretable_train = rng.lognormal(mean=8.5, sigma=0.3, size=n_scenarios)  # ~5K-15K
blackbox_train = rng.lognormal(mean=9.5, sigma=0.4, size=n_scenarios)  # ~15K-50K

# Inference costs per 1M predictions (USD)
interpretable_infer = rng.lognormal(mean=5.5, sigma=0.2, size=n_scenarios)  # ~150-300
blackbox_infer = rng.lognormal(mean=6.0, sigma=0.25, size=n_scenarios)  # ~400-800

# Explanation generation costs (post-hoc: SHAP, LIME, etc.)
# Interpretable models need no post-hoc explanation
posthoc_explanation = rng.lognormal(mean=7.0, sigma=0.35, size=n_scenarios)  # ~1K-3K per 1M

# Maintenance/retraining frequency (times per year)
interpretable_maintenance = rng.integers(1, 3, size=n_scenarios)
blackbox_maintenance = rng.integers(2, 5, size=n_scenarios)

# Accuracy scores (0-100)
interpretable_accuracy = rng.normal(78, 5, size=n_scenarios).clip(65, 90)
blackbox_accuracy = rng.normal(85, 4, size=n_scenarios).clip(75, 95)

# Regulatory compliance score (0-100) - interpretable models score higher
interpretable_compliance = rng.normal(90, 5, size=n_scenarios).clip(80, 100)
blackbox_compliance = rng.normal(65, 10, size=n_scenarios).clip(40, 85)

# Compute total cost of ownership (TCO) over 3 years
years = 3
interpretable_tco = (interpretable_train + 
                     interpretable_infer * 12 * years +  # monthly inference
                     interpretable_maintenance * interpretable_train * 0.3 * years)  # retraining cost

blackbox_tco = (blackbox_train + 
                blackbox_infer * 12 * years +
                posthoc_explanation * 12 * years +  # monthly explanations
                blackbox_maintenance * blackbox_train * 0.3 * years)

# Accuracy-adjusted cost (cost per accuracy point)
interpretable_cost_per_acc = interpretable_tco / interpretable_accuracy
blackbox_cost_per_acc = blackbox_tco / blackbox_accuracy

# Prepare results JSON
results = {
    "scenarios": scenarios,
    "interpretable_model": {
        "training_cost_usd": interpretable_train.tolist(),
        "inference_cost_per_1M_usd": interpretable_infer.tolist(),
        "maintenance_per_year": interpretable_maintenance.tolist(),
        "accuracy": interpretable_accuracy.tolist(),
        "compliance_score": interpretable_compliance.tolist(),
        "tco_3yr_usd": interpretable_tco.tolist(),
        "cost_per_accuracy_point": interpretable_cost_per_acc.tolist()
    },
    "blackbox_with_posthoc": {
        "training_cost_usd": blackbox_train.tolist(),
        "inference_cost_per_1M_usd": blackbox_infer.tolist(),
        "explanation_cost_per_1M_usd": posthoc_explanation.tolist(),
        "maintenance_per_year": blackbox_maintenance.tolist(),
        "accuracy": blackbox_accuracy.tolist(),
        "compliance_score": blackbox_compliance.tolist(),
        "tco_3yr_usd": blackbox_tco.tolist(),
        "cost_per_accuracy_point": blackbox_cost_per_acc.tolist()
    },
    "summary": {
        "mean_tco_savings_pct": float(np.mean((blackbox_tco - interpretable_tco) / blackbox_tco * 100)),
        "mean_accuracy_gap_pct": float(np.mean(blackbox_accuracy - interpretable_accuracy)),
        "mean_compliance_gap_pct": float(np.mean(interpretable_compliance - blackbox_compliance)),
        "scenarios_where_interpretable_wins": int(np.sum(interpretable_tco < blackbox_tco))
    }
}

# Save results
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ============================================================
# Chart 1: Total Cost of Ownership (3-year) Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(n_scenarios)
width = 0.35

bars1 = ax.bar(x - width/2, interpretable_tco / 1000, width, label='Interpretable Model', 
               color='#2E86AB', edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + width/2, blackbox_tco / 1000, width, label='Black-Box + Post-Hoc', 
               color='#A23B72', edgecolor='white', linewidth=1.5)

ax.set_ylabel('Total Cost of Ownership (USD Thousands)', fontsize=14)
ax.set_title('3-Year TCO: Interpretable Models vs Black-Box + Post-Hoc Explanations', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=12)
ax.legend(fontsize=13, framealpha=0.9)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'${height:.0f}K',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('charts/chart1.png', dpi=100)
plt.close()

# ============================================================
# Chart 2: Cost Breakdown Stacked Bar
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Stack components for interpretable
interp_components = {
    'Training': interpretable_train / 1000,
    'Inference (3yr)': interpretable_infer * 12 * years / 1000,
    'Retraining': interpretable_maintenance * interpretable_train * 0.3 * years / 1000
}

# Stack components for blackbox
bbox_components = {
    'Training': blackbox_train / 1000,
    'Inference (3yr)': blackbox_infer * 12 * years / 1000,
    'Explanations (3yr)': posthoc_explanation * 12 * years / 1000,
    'Retraining': blackbox_maintenance * blackbox_train * 0.3 * years / 1000
}

colors_interp = ['#2E86AB', '#5BA4CF', '#8FC7E8']
colors_bbox = ['#A23B72', '#C86BA8', '#E8A5D0', '#F4D0E8']

bottom_interp = np.zeros(n_scenarios)
for i, (label, values) in enumerate(interp_components.items()):
    ax.bar(x - width/2, values, width, bottom=bottom_interp, label=f'Interp: {label}',
           color=colors_interp[i], edgecolor='white', linewidth=1)
    bottom_interp += values

bottom_bbox = np.zeros(n_scenarios)
for i, (label, values) in enumerate(bbox_components.items()):
    ax.bar(x + width/2, values, width, bottom=bottom_bbox, label=f'BB+PH: {label}',
           color=colors_bbox[i], edgecolor='white', linewidth=1)
    bottom_bbox += values

ax.set_ylabel('Cost (USD Thousands)', fontsize=14)
ax.set_title('Cost Breakdown by Component (3-Year Horizon)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=12)
ax.legend(fontsize=11, framealpha=0.9, ncol=2)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('charts/chart2.png', dpi=100)
plt.close()

# ============================================================
# Chart 3: Accuracy vs Cost Efficiency Trade-off
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot: accuracy vs cost per accuracy point
sc1 = ax.scatter(interpretable_cost_per_acc, interpretable_accuracy, 
                 s=300, c='#2E86AB', label='Interpretable Model', 
                 edgecolors='white', linewidth=2, alpha=0.9, marker='o')
sc2 = ax.scatter(blackbox_cost_per_acc, blackbox_accuracy, 
                 s=300, c='#A23B72', label='Black-Box + Post-Hoc', 
                 edgecolors='white', linewidth=2, alpha=0.9, marker='^')

# Add scenario labels
for i, scenario in enumerate(scenarios):
    ax.annotate(scenario.replace('\n', ' '), 
                (interpretable_cost_per_acc[i], interpretable_accuracy[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9, color='#2E86AB')
    ax.annotate(scenario.replace('\n', ' '), 
                (blackbox_cost_per_acc[i], blackbox_accuracy[i]),
                xytext=(5, -15), textcoords='offset points', fontsize=9, color='#A23B72')

ax.set_xlabel('Cost per Accuracy Point (USD Thousands)', fontsize=14)
ax.set_ylabel('Model Accuracy (%)', fontsize=14)
ax.set_title('Accuracy vs Cost Efficiency Trade-off\n(Lower cost per accuracy point = better)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=13, framealpha=0.9)
ax.grid(alpha=0.3)

# Add quadrant lines
ax.axvline(x=np.median(np.concatenate([interpretable_cost_per_acc, blackbox_cost_per_acc])), 
           color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=np.median(np.concatenate([interpretable_accuracy, blackbox_accuracy])), 
           color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('charts/chart3.png', dpi=100)
plt.close()

# ============================================================
# Chart 4: Regulatory Compliance vs Total Cost
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

sc3 = ax.scatter(interpretable_tco / 1000, interpretable_compliance, 
                 s=300, c='#2E86AB', label='Interpretable Model', 
                 edgecolors='white', linewidth=2, alpha=0.9, marker='o')
sc4 = ax.scatter(blackbox_tco / 1000, blackbox_compliance, 
                 s=300, c='#A23B72', label='Black-Box + Post-Hoc', 
                 edgecolors='white', linewidth=2, alpha=0.9, marker='^')

# Add scenario labels
for i, scenario in enumerate(scenarios):
    ax.annotate(scenario.replace('\n', ' '), 
                (interpretable_tco[i] / 1000, interpretable_compliance[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9, color='#2E86AB')
    ax.annotate(scenario.replace('\n', ' '), 
                (blackbox_tco[i] / 1000, blackbox_compliance[i]),
                xytext=(5, -15), textcoords='offset points', fontsize=9, color='#A23B72')

ax.set_xlabel('3-Year TCO (USD Thousands)', fontsize=14)
ax.set_ylabel('Regulatory Compliance Score (%)', fontsize=14)
ax.set_title('Regulatory Compliance vs Total Cost\n(Higher compliance, lower cost = ideal)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=13, framealpha=0.9)
ax.grid(alpha=0.3)
ax.set_ylim(30, 105)

plt.tight_layout()
plt.savefig('charts/chart4.png', dpi=100)
plt.close()

# ============================================================
# Chart 5: Annual Cost Projection
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

years_arr = np.arange(1, 4)
# Average across scenarios
interp_annual = np.mean(interpretable_train) + np.mean(interpretable_infer) * 12 + \
                np.mean(interpretable_maintenance) * np.mean(interpretable_train) * 0.3
bbox_annual = np.mean(blackbox_train) + np.mean(blackbox_infer) * 12 + \
              np.mean(posthoc_explanation) * 12 + \
              np.mean(blackbox_maintenance) * np.mean(blackbox_train) * 0.3

interp_cumulative = np.cumsum([interp_annual] * 3) / 1000
bbox_cumulative = np.cumsum([bbox_annual] * 3) / 1000

ax.plot(years_arr, interp_cumulative, 'o-', color='#2E86AB', linewidth=3, 
        markersize=10, label='Interpretable Model')
ax.plot(years_arr, bbox_cumulative, 's-', color='#A23B72', linewidth=3, 
        markersize=10, label='Black-Box + Post-Hoc')

# Fill between
ax.fill_between(years_arr, interp_cumulative, bbox_cumulative, 
                where=(bbox_cumulative > interp_cumulative), 
                color='#2E86AB', alpha=0.15, label='Savings with Interpretable')

# Add value labels
for y, ic, bc in zip(years_arr, interp_cumulative, bbox_cumulative):
    ax.annotate(f'${ic:.0f}K', (y, ic), xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=11, fontweight='bold', color='#2E86AB')
    ax.annotate(f'${bc:.0f}K', (y, bc), xytext=(0, -20), textcoords='offset points',
                ha='center', fontsize=11, fontweight='bold', color='#A23B72')

ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Cumulative Cost (USD Thousands)', fontsize=14)
ax.set_title('3-Year Cumulative Cost Projection (Average Across Scenarios)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(years_arr)
ax.legend(fontsize=13, framealpha=0.9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('charts/chart5.png', dpi=100)
plt.close()

print("Analysis complete. Generated 5 charts and results.json")
print(f"Mean TCO savings with interpretable models: {results['summary']['mean_tco_savings_pct']:.1f}%")
print(f"Mean accuracy gap (blackbox - interpretable): {results['summary']['mean_accuracy_gap_pct']:.1f}%")
print(f"Mean compliance gap (interpretable - blackbox): {results['summary']['mean_compliance_gap_pct']:.1f}%")
print(f"Scenarios where interpretable wins on cost: {results['summary']['scenarios_where_interpretable_wins']}/5")
