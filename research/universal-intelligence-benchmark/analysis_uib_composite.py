#!/usr/bin/env python3
"""Analysis for: The UIB Composite Score — Integration Across All Dimensions"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
    'savefig.bbox': 'tight'
})

with open("uib_model_data.json") as f:
    models = json.load(f)

# UIB Dimension weights (information-theoretic, based on variance contribution)
DIM_WEIGHTS = {
    "Reasoning": 0.18,
    "Causal": 0.14,
    "Temporal": 0.12,
    "Social": 0.10,
    "Efficiency": 0.15,
    "Transfer": 0.11,
    "Embodied": 0.08,
    "Tool-Use": 0.12,
}

# Map benchmark scores to UIB dimensions for each model
def compute_uib_dimensions(m):
    d = models[m]
    return {
        "Reasoning": (d["gpqa"] + d["math_500"]) / 2,
        "Causal": d["arc_agi"] * 1.1,  # ARC proxies causal reasoning
        "Temporal": d["arc_agi"] * 0.85 + d["math_500"] * 0.15,
        "Social": d["mmlu_pro"] * 0.6 + d["gpqa"] * 0.4,  # proxy
        "Efficiency": min(100, 100 * (1.0 / (d["cost_per_1M_tok"] + 0.1)) * 1.5),
        "Transfer": (d["mmlu_pro"] + d["humaneval"]) / 2,
        "Embodied": d["arc_agi"] * 0.7,  # low proxy
        "Tool-Use": (d["humaneval"] + d["arc_agi"]) / 2,
    }

# Compute composite UIB scores
uib_scores = {}
uib_dims = {}
for m in models:
    dims = compute_uib_dimensions(m)
    uib_dims[m] = dims
    composite = sum(DIM_WEIGHTS[k] * dims[k] for k in DIM_WEIGHTS)
    uib_scores[m] = composite

# Sort by UIB composite
sorted_models = sorted(uib_scores.keys(), key=lambda x: uib_scores[x], reverse=True)

# Chart 1: UIB Composite Score Bar Chart
fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_models)))
bars = ax.barh(range(len(sorted_models)), [uib_scores[m] for m in sorted_models], color=colors)
ax.set_yticks(range(len(sorted_models)))
ax.set_yticklabels(sorted_models)
ax.set_xlabel('UIB Composite Score')
ax.set_title('Universal Intelligence Benchmark: Composite Scores (March 2026)')
ax.invert_yaxis()
for i, m in enumerate(sorted_models):
    ax.text(uib_scores[m] + 0.3, i, f'{uib_scores[m]:.1f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('charts/01-uib-composite-scores.png')
plt.close()
print("Chart 1 saved")

# Chart 2: Radar/Spider chart for top 5 models
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
dims_list = list(DIM_WEIGHTS.keys())
angles = np.linspace(0, 2 * np.pi, len(dims_list), endpoint=False).tolist()
angles += angles[:1]

top5 = sorted_models[:5]
colors_radar = ['#111111', '#555555', '#888888', '#aaaaaa', '#cccccc']
for idx, m in enumerate(top5):
    values = [uib_dims[m][d] for d in dims_list]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=m, color=colors_radar[idx])
    ax.fill(angles, values, alpha=0.05, color=colors_radar[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dims_list, fontsize=10)
ax.set_title('UIB Dimension Profiles: Top 5 Models', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig('charts/02-uib-radar-top5.png')
plt.close()
print("Chart 2 saved")

# Chart 3: Cost-Efficiency Frontier (UIB Score vs Cost)
fig, ax = plt.subplots(figsize=(10, 7))
for m in models:
    cost = models[m]["cost_per_1M_tok"]
    score = uib_scores[m]
    size = models[m]["params_B"] / 10
    ax.scatter(cost, score, s=max(size, 30), color='#333333', alpha=0.7, edgecolors='#000', linewidth=1)
    ax.annotate(m, (cost, score), textcoords="offset points", xytext=(8, 4), fontsize=8)

ax.set_xlabel('Cost per 1M Tokens ($)')
ax.set_ylabel('UIB Composite Score')
ax.set_title('UIB Score vs. Inference Cost: The Efficiency Frontier')
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('charts/03-uib-cost-frontier.png')
plt.close()
print("Chart 3 saved")

# Chart 4: Dimension Weight Contribution (stacked bar)
fig, ax = plt.subplots(figsize=(12, 7))
bottom = np.zeros(len(sorted_models))
dim_colors = plt.cm.gray(np.linspace(0.2, 0.85, len(dims_list)))
for i, dim in enumerate(dims_list):
    contributions = [DIM_WEIGHTS[dim] * uib_dims[m][dim] for m in sorted_models]
    ax.barh(range(len(sorted_models)), contributions, left=bottom, label=dim, color=dim_colors[i])
    bottom += contributions

ax.set_yticks(range(len(sorted_models)))
ax.set_yticklabels(sorted_models)
ax.set_xlabel('Weighted Dimension Contribution to UIB Score')
ax.set_title('UIB Score Decomposition by Dimension')
ax.legend(loc='lower right', fontsize=8)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('charts/04-uib-dimension-decomposition.png')
plt.close()
print("Chart 4 saved")

# Chart 5: Heatmap of dimension scores
fig, ax = plt.subplots(figsize=(12, 8))
data_matrix = np.array([[uib_dims[m][d] for d in dims_list] for m in sorted_models])
im = ax.imshow(data_matrix, cmap='Greys', aspect='auto')
ax.set_xticks(range(len(dims_list)))
ax.set_xticklabels(dims_list, rotation=45, ha='right')
ax.set_yticks(range(len(sorted_models)))
ax.set_yticklabels(sorted_models)
ax.set_title('UIB Dimension Score Heatmap')
for i in range(len(sorted_models)):
    for j in range(len(dims_list)):
        ax.text(j, i, f'{data_matrix[i, j]:.0f}', ha='center', va='center', fontsize=8,
                color='white' if data_matrix[i, j] > 60 else 'black')
plt.colorbar(im, ax=ax, label='Score')
plt.tight_layout()
plt.savefig('charts/05-uib-dimension-heatmap.png')
plt.close()
print("Chart 5 saved")

# Print summary stats
print("\n--- UIB Composite Scores ---")
for m in sorted_models:
    print(f"  {m}: {uib_scores[m]:.1f}")

# Dimension variance analysis
print("\n--- Dimension Variance (information content) ---")
for dim in dims_list:
    vals = [uib_dims[m][dim] for m in models]
    print(f"  {dim}: mean={np.mean(vals):.1f}, std={np.std(vals):.1f}, CoV={np.std(vals)/np.mean(vals):.3f}")
