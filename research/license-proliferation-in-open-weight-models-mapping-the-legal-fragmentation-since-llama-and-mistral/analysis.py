"""
License Proliferation in Open-Weight Models: Analysis Script

Reproduce: python3 analysis.py

Analyzes the divergence of open-weight model licenses and quantifies 
the compliance burden for downstream adopters since Llama (2023) and Mistral (2023).
Uses synthetic data with numpy.random.default_rng(42) for reproducibility.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set random seed for reproducibility
rng = np.random.default_rng(42)

# Configuration
WORKDIR = Path(__file__).parent
CHARTS_DIR = WORKDIR / "charts"
RESULTS_FILE = WORKDIR / "results.json"

CHARTS_DIR.mkdir(exist_ok=True)

# ---- Synthetic Data Generation ----

# Timeline: 2022 (pre-Llama) through 2025
years = np.arange(2022, 2026)

# 1. Number of unique open-weight model licenses over time
# Base: ~5 licenses pre-2023, explosive growth after Llama/Mistral releases
base_licenses = 5
license_counts = []
for i, year in enumerate(years):
    if year < 2023:
        # Pre-Llama era: slow growth
        count = base_licenses + rng.poisson(1)
    elif year == 2023:
        # Llama 2 (July 2023) + Mistral 7B (Sept 2023) trigger proliferation
        count = base_licenses + rng.poisson(8)
    elif year == 2024:
        # Peak proliferation year
        count = license_counts[-1] + rng.poisson(12)
    else:  # 2025
        # Stabilization but continued growth
        count = license_counts[-1] + rng.poisson(6)
    license_counts.append(max(count, license_counts[-1] if license_counts else 0))

# 2. License type distribution (permissive, copyleft, custom/restrictive)
license_types = ["Permissive", "Copyleft", "Custom/Restrictive"]
# 2022 distribution (mostly permissive)
type_dist_2022 = [0.80, 0.15, 0.05]
# 2025 distribution (significant custom/restrictive growth)
type_dist_2025 = [0.45, 0.15, 0.40]

# 3. Compliance burden metrics
compliance_metrics = {
    "clauses_to_review": [],
    "incompatibility_pairs": [],
    "custom_license_count": [],
    "avg_license_length_kwords": []
}

for i, year in enumerate(years):
    n_licenses = license_counts[i]
    # Clauses scale superlinearly with license count
    clauses = int(n_licenses * rng.uniform(8, 15))
    # Incompatibility pairs ~ n^2 for custom licenses
    custom_ratio = 0.05 if year < 2023 else (0.25 if year == 2023 else (0.35 if year == 2024 else 0.40))
    n_custom = int(n_licenses * custom_ratio)
    incompat = int(n_custom * (n_custom - 1) / 2 * rng.uniform(0.3, 0.6))
    # Average license length in keywords
    avg_len = rng.uniform(2.5, 5.5) if year < 2023 else rng.uniform(4.0, 8.0)
    
    compliance_metrics["clauses_to_review"].append(clauses)
    compliance_metrics["incompatibility_pairs"].append(incompat)
    compliance_metrics["custom_license_count"].append(n_custom)
    compliance_metrics["avg_license_length_kwords"].append(round(avg_len, 1))

# 4. Model release timeline with license types
model_releases = [
    {"model": "LLaMA", "year": 2023, "month": 2, "license": "Custom (Meta)", "type": "Custom/Restrictive"},
    {"model": "LLaMA 2", "year": 2023, "month": 7, "license": "Llama 2 Community License", "type": "Custom/Restrictive"},
    {"model": "Falcon 40B", "year": 2023, "month": 5, "license": "Apache-2.0", "type": "Permissive"},
    {"model": "Mistral 7B", "year": 2023, "month": 9, "license": "Apache-2.0", "type": "Permissive"},
    {"model": "Mixtral 8x7B", "year": 2023, "month": 12, "license": "Apache-2.0", "type": "Permissive"},
    {"model": "Gemma", "year": 2024, "month": 2, "license": "Gemma License", "type": "Custom/Restrictive"},
    {"model": "Llama 3", "year": 2024, "month": 4, "license": "Llama 3 Community License", "type": "Custom/Restrictive"},
    {"model": "Nemotron 3 Ultra", "year": 2024, "month": 6, "license": "NVIDIA Open Model License", "type": "Custom/Restrictive"},
    {"model": "Qwen 2.5", "year": 2024, "month": 9, "license": "Tongyi Qianwen License", "type": "Custom/Restrictive"},
    {"model": "Nemotron 4 Ultra", "year": 2025, "month": 1, "license": "NVIDIA Open Model License", "type": "Custom/Restrictive"},
    {"model": "Llama 3.1", "year": 2025, "month": 7, "license": "Llama 3.1 Community License", "type": "Custom/Restrictive"},
    {"model": "Qwen 3", "year": 2025, "month": 10, "license": "Tongyi Qianwen License", "type": "Custom/Restrictive"},
]

# ---- Chart 1: License Count Over Time ----
fig1, ax1 = plt.subplots(figsize=(12, 8))
bars = ax1.bar(years, license_counts, color=['#2ecc71' if y < 2023 else '#e74c3c' for y in years], 
               edgecolor='black', linewidth=1.2, width=0.6)
ax1.set_title("Unique Open-Weight Model Licenses Over Time", fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel("Year", fontsize=13)
ax1.set_ylabel("Number of Unique Licenses", fontsize=13)
ax1.set_xticks(years)
ax1.set_xticklabels([str(y) for y in years], fontsize=12)

# Add value labels on bars
for bar, count in zip(bars, license_counts):
    ax1.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 5), textcoords='offset points', ha='center', va='bottom',
                 fontsize=14, fontweight='bold')

# Add annotation for Llama/Mistral
ax1.annotate('LLaMA 2 (Jul 2023)\nMistral 7B (Sep 2023)', 
             xy=(2023, license_counts[1]), xytext=(2022.5, license_counts[1] + 3),
             fontsize=11, ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='black'))

ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, max(license_counts) * 1.3)
fig1.tight_layout()
fig1.savefig(CHARTS_DIR / "chart1.png", dpi=100)
plt.close(fig1)

# ---- Chart 2: License Type Distribution (Stacked Bar) ----
fig2, ax2 = plt.subplots(figsize=(12, 8))

# Interpolate distributions for each year
type_distributions = []
for i, year in enumerate(years):
    if year == 2022:
        type_distributions.append(type_dist_2022)
    elif year == 2025:
        type_distributions.append(type_dist_2025)
    else:
        # Linear interpolation
        t = (year - 2022) / 3
        dist = [type_dist_2022[j] * (1-t) + type_dist_2025[j] * t for j in range(3)]
        type_distributions.append(dist)

type_distributions = np.array(type_distributions)
colors = ['#2ecc71', '#f39c12', '#e74c3c']
bottoms = np.zeros(len(years))

for i, (label, color) in enumerate(zip(license_types, colors)):
    bars = ax2.bar(years, type_distributions[:, i], bottom=bottoms, label=label, 
                   color=color, edgecolor='black', linewidth=1, width=0.6)
    # Add percentage labels
    for j, (year, val) in enumerate(zip(years, type_distributions[:, i])):
        if val > 0.05:
            ax2.annotate(f'{val*100:.0f}%', 
                         xy=(year, bottoms[j] + val/2), ha='center', va='center',
                         fontsize=11, fontweight='bold', color='white')
    bottoms += type_distributions[:, i]

ax2.set_title("License Type Distribution Among Open-Weight Models", fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel("Year", fontsize=13)
ax2.set_ylabel("Proportion of Models", fontsize=13)
ax2.set_xticks(years)
ax2.set_xticklabels([str(y) for y in years], fontsize=12)
ax2.set_ylim(0, 1.05)
ax2.legend(loc='upper right', fontsize=12, framealpha=0.9)
ax2.grid(axis='y', alpha=0.3)
fig2.tight_layout()
fig2.savefig(CHARTS_DIR / "chart2.png", dpi=100)
plt.close(fig2)

# ---- Chart 3: Compliance Burden Metrics ----
fig3, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

metrics_labels = [
    ("Clauses to Review", compliance_metrics["clauses_to_review"], "#3498db"),
    ("Incompatibility Pairs", compliance_metrics["incompatibility_pairs"], "#e74c3c"),
    ("Custom Licenses Count", compliance_metrics["custom_license_count"], "#9b59b6"),
    ("Avg License Length (k words)", compliance_metrics["avg_license_length_kwords"], "#f39c12")
]

for ax, (label, data, color) in zip(axes, metrics_labels):
    bars = ax.bar(years, data, color=color, edgecolor='black', linewidth=1, width=0.6)
    ax.set_title(label, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, data):
        ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

fig3.suptitle("Compliance Burden Metrics for Downstream Adopters", fontsize=16, fontweight='bold', y=1.02)
fig3.tight_layout()
fig3.savefig(CHARTS_DIR / "chart3.png", dpi=100)
plt.close(fig3)

# ---- Chart 4: Model Release Timeline with License Types ----
fig4, ax4 = plt.subplots(figsize=(12, 8))

type_colors = {"Permissive": "#2ecc71", "Copyleft": "#f39c12", "Custom/Restrictive": "#e74c3c"}
type_markers = {"Permissive": "o", "Copyleft": "s", "Custom/Restrictive": "D"}

for model in model_releases:
    x = model["year"] + (model["month"] - 1) / 12
    color = type_colors[model["type"]]
    marker = type_markers[model["type"]]
    ax4.scatter(x, 0, c=color, marker=marker, s=200, edgecolor='black', linewidth=1.5, zorder=5)
    # Label
    ax4.annotate(model["model"], xy=(x, 0), xytext=(0, 25 + rng.uniform(-5, 5)),
                 textcoords='offset points', ha='center', va='bottom',
                 fontsize=10, fontweight='bold', rotation=45,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color))
    # License type
    ax4.annotate(model["license"], xy=(x, 0), xytext=(0, -30),
                 textcoords='offset points', ha='center', va='top',
                 fontsize=8, style='italic', color=color)

# Legend - use Line2D for marker legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker=m, color='w', label=l, 
                          markerfacecolor=c, markeredgecolor='black', markersize=12, 
                          markeredgewidth=1.5)
                   for l, (c, m) in zip(license_types, zip(type_colors.values(), type_markers.values()))]
ax4.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), 
           ncol=3, fontsize=11, framealpha=0.9)

ax4.set_title("Major Open-Weight Model Releases by License Type (2023-2025)", fontsize=16, fontweight='bold', pad=20)
ax4.set_xlabel("Release Date", fontsize=13)
ax4.set_yticks([])
ax4.set_xlim(2022.5, 2025.8)
ax4.grid(axis='x', alpha=0.3)

# Add vertical lines for key events
ax4.axvline(x=2023.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax4.axvline(x=2024.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

fig4.tight_layout()
fig4.savefig(CHARTS_DIR / "chart4.png", dpi=100)
plt.close(fig4)

# ---- Chart 5: License Fragmentation Index (Herfindahl-Hirschman Index) ----
fig5, ax5 = plt.subplots(figsize=(12, 8))

# HHI calculation: sum of squared market shares
hhi_values = []
for dist in type_distributions:
    hhi = sum(d**2 for d in dist)
    hhi_values.append(hhi)

# Normalize fragmentation index: (1 - HHI) / (1 - 1/3) = 1.5 * (1 - HHI)
fragmentation_index = [1.5 * (1 - hhi) for hhi in hhi_values]

ax5.plot(years, fragmentation_index, 'o-', color='#8e44ad', linewidth=3, markersize=10, label='Fragmentation Index')
ax5.fill_between(years, fragmentation_index, alpha=0.2, color='#8e44ad')

# Add value labels
for year, val in zip(years, fragmentation_index):
    ax5.annotate(f'{val:.2f}', xy=(year, val), xytext=(0, 10),
                 textcoords='offset points', ha='center', va='bottom',
                 fontsize=13, fontweight='bold', color='#8e44ad')

ax5.set_title("License Fragmentation Index (Normalized Herfindahl-Hirschman)", fontsize=16, fontweight='bold', pad=20)
ax5.set_xlabel("Year", fontsize=13)
ax5.set_ylabel("Fragmentation Index (0=Concentrated, 1=Max Fragmented)", fontsize=13)
ax5.set_xticks(years)
ax5.set_xticklabels([str(y) for y in years], fontsize=12)
ax5.set_ylim(0, 1.05)
ax5.grid(axis='both', alpha=0.3)
ax5.legend(fontsize=12, loc='upper left')

# Annotate key events
ax5.annotate('Pre-Llama:\nLow fragmentation', xy=(2022, fragmentation_index[0]), 
             xytext=(2022, 0.2), fontsize=10, ha='center',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#2ecc71', alpha=0.3))
ax5.annotate('Post-Llama/Mistral:\nHigh fragmentation', xy=(2025, fragmentation_index[-1]), 
             xytext=(2025, 0.9), fontsize=10, ha='center',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e74c3c', alpha=0.3))

fig5.tight_layout()
fig5.savefig(CHARTS_DIR / "chart5.png", dpi=100)
plt.close(fig5)

# ---- Save Results JSON ----
results = {
    "license_counts_by_year": dict(zip([str(y) for y in years], [int(c) for c in license_counts])),
    "license_type_distribution": {
        str(y): dict(zip(license_types, [round(d, 3) for d in dist]))
        for y, dist in zip(years, type_distributions)
    },
    "compliance_metrics": {
        "clauses_to_review": dict(zip([str(y) for y in years], compliance_metrics["clauses_to_review"])),
        "incompatibility_pairs": dict(zip([str(y) for y in years], compliance_metrics["incompatibility_pairs"])),
        "custom_license_count": dict(zip([str(y) for y in years], compliance_metrics["custom_license_count"])),
        "avg_license_length_kwords": dict(zip([str(y) for y in years], compliance_metrics["avg_license_length_kwords"]))
    },
    "fragmentation_index": dict(zip([str(y) for y in years], [round(f, 3) for f in fragmentation_index])),
    "model_releases": model_releases,
    "summary": {
        "total_unique_licenses_2025": int(license_counts[-1]),
        "custom_license_share_2025": round(type_dist_2025[2], 2),
        "fragmentation_index_2025": round(fragmentation_index[-1], 3),
        "compliance_burden_increase_2022_to_2025": round(
            compliance_metrics["clauses_to_review"][-1] / compliance_metrics["clauses_to_review"][0], 1
        )
    }
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Generated {len(list(CHARTS_DIR.glob('*.png')))} charts in {CHARTS_DIR}")
print(f"Saved results to {RESULTS_FILE}")
print("Done!")
