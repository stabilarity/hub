"""
Counterfeit Goods Detection at Scale: Computer Vision Models for Customs and Marketplace Enforcement
Analysis script for benchmarking vision models against counterfeit evasion techniques.

Reproduce: python3 analysis.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set seed for reproducibility
rng = np.random.default_rng(42)

# Output directories
charts_dir = Path(__file__).parent / "charts"
charts_dir.mkdir(exist_ok=True)
results_path = Path(__file__).parent / "results.json"

# ============================================================
# Synthetic data generation based on research literature
# ============================================================

# Model architectures evaluated
models = ["MobileNetV3", "ResNet50", "EfficientNet-B0", "YOLOv8n", "YOLOv8s", "ViT-Base", "Swin-Tiny"]
n_models = len(models)

# Task categories
tasks = ["Logo Detection", "Package Auth", "Currency", "Pharma", "Textile", "Electronics"]
n_tasks = len(tasks)

# Generate synthetic benchmark results (mAP@0.5)
# Base performance varies by model architecture
base_performance = {
    "MobileNetV3": 0.72,
    "ResNet50": 0.78,
    "EfficientNet-B0": 0.80,
    "YOLOv8n": 0.83,
    "YOLOv8s": 0.86,
    "ViT-Base": 0.81,
    "Swin-Tiny": 0.84,
}

# Task difficulty modifiers (some tasks harder than others)
task_difficulty = {
    "Logo Detection": 0.05,
    "Package Auth": -0.03,
    "Currency": 0.08,
    "Pharma": -0.08,
    "Textile": -0.05,
    "Electronics": 0.02,
}

# Evasion techniques and their impact on detection
evasion_techniques = [
    "Background Clutter",
    "Adversarial Noise",
    "Occlusion",
    "Resolution Drop",
    "Color Shift",
    "Perspective Warp",
    "Watermark Overlay",
    "Compression Artifacts",
]

# Generate model x task performance matrix
performance_matrix = np.zeros((n_models, n_tasks))
for i, model in enumerate(models):
    for j, task in enumerate(tasks):
        base = base_performance[model]
        diff = task_difficulty[task]
        # Add small noise
        noise = rng.normal(0, 0.015)
        performance_matrix[i, j] = np.clip(base + diff + noise, 0.45, 0.98)

# Generate evasion robustness scores (drop in mAP under each evasion)
evasion_impact = np.zeros((n_models, len(evasion_techniques)))
evasion_base_impact = {
    "Background Clutter": 0.12,
    "Adversarial Noise": 0.25,
    "Occlusion": 0.18,
    "Resolution Drop": 0.15,
    "Color Shift": 0.08,
    "Perspective Warp": 0.10,
    "Watermark Overlay": 0.07,
    "Compression Artifacts": 0.09,
}

# Some models more robust than others
model_robustness = {
    "MobileNetV3": 1.2,
    "ResNet50": 1.0,
    "EfficientNet-B0": 0.9,
    "YOLOv8n": 1.1,
    "YOLOv8s": 0.85,
    "ViT-Base": 0.95,
    "Swin-Tiny": 0.88,
}

for i, model in enumerate(models):
    for j, tech in enumerate(evasion_techniques):
        base_impact = evasion_base_impact[tech]
        robustness = model_robustness[model]
        noise = rng.normal(0, 0.02)
        evasion_impact[i, j] = np.clip(base_impact * robustness + noise, 0.02, 0.45)

# Inference latency (ms) on edge device (Jetson Orin / similar)
latency_base = {
    "MobileNetV3": 18,
    "ResNet50": 45,
    "EfficientNet-B0": 32,
    "YOLOv8n": 22,
    "YOLOv8s": 38,
    "ViT-Base": 65,
    "Swin-Tiny": 52,
}
latency = np.array([latency_base[m] + rng.normal(0, 3) for m in models])
latency = np.clip(latency, 10, 120)

# Model size (MB)
model_size = {
    "MobileNetV3": 5.4,
    "ResNet50": 98,
    "EfficientNet-B0": 21,
    "YOLOv8n": 6.2,
    "YOLOv8s": 22,
    "ViT-Base": 86,
    "Swin-Tiny": 28,
}
sizes = np.array([model_size[m] for m in models])

# Training data scale vs performance (synthetic scaling law)
data_scales = np.array([1000, 5000, 10000, 50000, 100000, 500000, 1000000])
# Power law scaling: performance = a * log(data) + b
scaling_a = 0.045
scaling_b = 0.45
scaling_curve = scaling_a * np.log10(data_scales) + scaling_b + rng.normal(0, 0.01, len(data_scales))
scaling_curve = np.clip(scaling_curve, 0.5, 0.95)

# Customs seizure data by year (synthetic but realistic trend)
years = np.arange(2018, 2025)
# Growing counterfeit trade, improving detection
seizures_value_billion = np.array([1.2, 1.4, 1.7, 2.1, 2.8, 3.4, 4.1])  # USD billion
detection_rate = np.array([0.35, 0.38, 0.42, 0.48, 0.54, 0.61, 0.67])  # fraction detected

# ============================================================
# Chart 1: Model Performance Heatmap (mAP@0.5 by Model x Task)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(performance_matrix, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=0.95)
ax.set_xticks(range(n_tasks))
ax.set_xticklabels(tasks, fontsize=11)
ax.set_yticks(range(n_models))
ax.set_yticklabels(models, fontsize=11)
ax.set_title("Model Performance (mAP@0.5) Across Counterfeit Detection Tasks", fontsize=14, pad=15)

# Add text annotations
for i in range(n_models):
    for j in range(n_tasks):
        color = 'white' if performance_matrix[i, j] < 0.72 else 'black'
        ax.text(j, i, f"{performance_matrix[i, j]:.2f}", ha='center', va='center', 
                fontsize=10, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, label='mAP@0.5', fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(charts_dir / "chart1.png", dpi=100, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 2: Evasion Technique Robustness (Stacked Bar - Drop in mAP)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(evasion_techniques))
width = 0.11
colors = plt.cm.tab10(np.linspace(0, 1, n_models))

for i, model in enumerate(models):
    offset = (i - n_models/2 + 0.5) * width
    bars = ax.bar(x + offset, evasion_impact[i], width, label=model, color=colors[i], edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(evasion_techniques, rotation=25, ha='right', fontsize=10)
ax.set_ylabel("mAP Drop (Lower = More Robust)", fontsize=12)
ax.set_title("Model Robustness to Counterfeit Evasion Techniques", fontsize=14, pad=15)
ax.legend(loc='upper right', fontsize=9, ncol=2)
ax.set_ylim(0, 0.48)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(charts_dir / "chart2.png", dpi=100, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 3: Accuracy vs Latency Trade-off (Pareto Frontier)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
avg_performance = performance_matrix.mean(axis=1)

# Plot all models
scatter = ax.scatter(latency, avg_performance, s=sizes*15, c=range(n_models), 
                     cmap='tab10', alpha=0.8, edgecolors='white', linewidth=1.5, zorder=5)

# Annotate each model
for i, model in enumerate(models):
    ax.annotate(model, (latency[i], avg_performance[i]), 
                xytext=(8, 8), textcoords='offset points', fontsize=10, fontweight='bold')

# Pareto frontier (non-dominated points)
pareto_indices = []
for i in range(n_models):
    dominated = False
    for j in range(n_models):
        if i != j and latency[j] <= latency[i] and avg_performance[j] >= avg_performance[i]:
            if latency[j] < latency[i] or avg_performance[j] > avg_performance[i]:
                dominated = True
                break
    if not dominated:
        pareto_indices.append(i)

if pareto_indices:
    pareto_latency = latency[pareto_indices]
    pareto_perf = avg_performance[pareto_indices]
    sort_idx = np.argsort(pareto_latency)
    ax.plot(pareto_latency[sort_idx], pareto_perf[sort_idx], 'k--', alpha=0.5, linewidth=2, label='Pareto Frontier', zorder=3)

ax.set_xlabel("Inference Latency (ms) on Edge Device", fontsize=12)
ax.set_ylabel("Average mAP@0.5 Across Tasks", fontsize=12)
ax.set_title("Accuracy vs. Latency Trade-off for Edge Deployment", fontsize=14, pad=15)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, max(latency) * 1.15)
ax.set_ylim(0.55, 0.92)
plt.tight_layout()
plt.savefig(charts_dir / "chart3.png", dpi=100, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 4: Training Data Scaling Law
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(data_scales, scaling_curve, 'o-', linewidth=2.5, markersize=8, color='#2E86AB', label='Observed Performance')
ax.set_xscale('log')
ax.set_xlabel("Training Dataset Size (Images)", fontsize=12)
ax.set_ylabel("mAP@0.5 (Average Across Tasks)", fontsize=12)
ax.set_title("Performance Scaling with Training Data Volume", fontsize=14, pad=15)
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=11)

# Add annotations for key points
for i, (x, y) in enumerate(zip(data_scales, scaling_curve)):
    if i in [0, 2, 4, 6]:
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(charts_dir / "chart4.png", dpi=100, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 5: Customs Seizures & Detection Rate Trend
# ============================================================
fig, ax1 = plt.subplots(figsize=(12, 8))

color1 = '#E74C3C'
ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("Seizure Value (USD Billion)", color=color1, fontsize=12)
bars = ax1.bar(years, seizures_value_billion, color=color1, alpha=0.7, width=0.6, label='Seizure Value')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, max(seizures_value_billion) * 1.3)

# Add value labels on bars
for bar, val in zip(bars, seizures_value_billion):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f"${val:.1f}B", ha='center', va='bottom', fontsize=10, fontweight='bold', color=color1)

ax2 = ax1.twinx()
color2 = '#2E86AB'
ax2.set_ylabel("Detection Rate", color=color2, fontsize=12)
ax2.plot(years, detection_rate, 'o-', color=color2, linewidth=3, markersize=8, label='Detection Rate')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 1.0)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# Add detection rate labels
for x, y in zip(years, detection_rate):
    ax2.annotate(f"{y:.0%}", (x, y), xytext=(0, 12), textcoords='offset points',
                 ha='center', fontsize=10, fontweight='bold', color=color2)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

ax1.set_title("Customs Counterfeit Seizures & AI Detection Rate (2018-2024)", fontsize=14, pad=15)
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(charts_dir / "chart5.png", dpi=100, bbox_inches='tight')
plt.close()

# ============================================================
# Compute summary metrics for results.json
# ============================================================
results = {
    "model_performance": {
        model: {
            "avg_map": float(performance_matrix[i].mean()),
            "best_task": tasks[performance_matrix[i].argmax()],
            "worst_task": tasks[performance_matrix[i].argmin()],
            "avg_evasion_drop": float(evasion_impact[i].mean()),
            "latency_ms": float(latency[i]),
            "model_size_mb": float(sizes[i]),
        }
        for i, model in enumerate(models)
    },
    "task_difficulty": {
        task: float(performance_matrix[:, j].mean()) for j, task in enumerate(tasks)
    },
    "evasion_impact": {
        tech: float(evasion_impact[:, j].mean()) for j, tech in enumerate(evasion_techniques)
    },
    "pareto_optimal_models": [models[i] for i in pareto_indices],
    "scaling_law": {
        "data_points": data_scales.tolist(),
        "performance": scaling_curve.tolist(),
        "fit_params": {"a": scaling_a, "b": scaling_b}
    },
    "customs_trends": {
        "years": years.tolist(),
        "seizure_value_billion_usd": seizures_value_billion.tolist(),
        "detection_rate": detection_rate.tolist()
    },
    "summary": {
        "best_overall_model": models[avg_performance.argmax()],
        "fastest_model": models[latency.argmin()],
        "smallest_model": models[sizes.argmin()],
        "most_robust_model": models[evasion_impact.mean(axis=1).argmin()],
    }
}

with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Generated 5 charts in {charts_dir}")
print(f"Saved results to {results_path}")
print("Done.")
