"""
Continual Learning Without Catastrophic Forgetting: Production-Ready Approaches in 2026
Reproduce: python3 analysis.py

Generates charts and metrics from synthetic benchmark data representing
continual learning methods evaluated against forgetting benchmarks (2024-2026).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set seed for reproducibility
rng = np.random.default_rng(42)

# Create charts directory
charts_dir = Path("./charts")
charts_dir.mkdir(exist_ok=True)

# ============================================================
# SYNTHETIC BENCHMARK DATA (based on 2024-2026 literature)
# ============================================================

# Method categories and their typical performance characteristics
method_keys = [
    "Regularization",
    "Replay",
    "Architectural",
    "Prompt-Based",
    "Parameter-Isolation",
    "Hybrid",
]

methods = {
    "Regularization": {"forgetting": 0.15, "plasticity": 0.65, "overhead": 0.10},
    "Replay": {"forgetting": 0.08, "plasticity": 0.78, "overhead": 0.35},
    "Architectural": {"forgetting": 0.05, "plasticity": 0.72, "overhead": 0.40},
    "Prompt-Based": {"forgetting": 0.06, "plasticity": 0.80, "overhead": 0.15},
    "Parameter-Isolation": {"forgetting": 0.07, "plasticity": 0.75, "overhead": 0.20},
    "Hybrid": {"forgetting": 0.04, "plasticity": 0.82, "overhead": 0.25},
}

# Benchmark scenarios
scenarios = ["Class-Incremental", "Task-Incremental", "Domain-Incremental", "Continual Fine-Tuning"]

# Generate synthetic benchmark results across scenarios
results = {}
for scenario in scenarios:
    scenario_data = {}
    for method in method_keys:
        base = methods[method]
        # Add scenario-specific variation
        noise = rng.normal(0, 0.02, 3)
        scenario_data[method] = {
            "forgetting": np.clip(base["forgetting"] + noise[0], 0.01, 0.5),
            "plasticity": np.clip(base["plasticity"] + noise[1], 0.4, 0.95),
            "overhead": np.clip(base["overhead"] + noise[2], 0.05, 0.6),
        }
    results[scenario] = scenario_data

# Timeline data: yearly improvement in forgetting rate (2020-2026)
years = np.arange(2020, 2027)
forgetting_trend = {
    "Regularization": rng.uniform(0.35, 0.45, size=1) - 0.04 * (years - 2020) + rng.normal(0, 0.015, len(years)),
    "Replay": rng.uniform(0.25, 0.35, size=1) - 0.03 * (years - 2020) + rng.normal(0, 0.015, len(years)),
    "Architectural": rng.uniform(0.20, 0.30, size=1) - 0.025 * (years - 2020) + rng.normal(0, 0.015, len(years)),
    "Prompt-Based": rng.uniform(0.15, 0.25, size=1) - 0.035 * (years - 2020) + rng.normal(0, 0.015, len(years)),
}
for k in forgetting_trend:
    forgetting_trend[k] = np.clip(forgetting_trend[k], 0.02, 0.5)

# Production deployment metrics (latency, memory, accuracy retention)
deployment_metrics = {
    "Regularization (EWC)": {"latency_ms": 12, "memory_mb": 180, "retention": 0.82},
    "Replay (Experience)": {"latency_ms": 45, "memory_mb": 850, "retention": 0.89},
    "Architectural (PackNet)": {"latency_ms": 8, "memory_mb": 420, "retention": 0.92},
    "Prompt-Based (L2P)": {"latency_ms": 18, "memory_mb": 220, "retention": 0.90},
    "Parameter-Isolation (LoRA)": {"latency_ms": 22, "memory_mb": 310, "retention": 0.88},
    "Hybrid (COPE)": {"latency_ms": 30, "memory_mb": 380, "retention": 0.93},
}

# ============================================================
# CHART 1: Forgetting Rate by Method & Scenario (Grouped Bar)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(method_keys))
width = 0.2
colors = plt.cm.Set2(np.linspace(0, 1, len(scenarios)))

for i, scenario in enumerate(scenarios):
    forgetting_vals = [results[scenario][m]["forgetting"] for m in method_keys]
    ax.bar(x + i * width - width * 1.5, forgetting_vals, width, 
           label=scenario, color=colors[i], edgecolor='white', linewidth=0.5)

ax.set_ylabel("Forgetting Rate (lower is better)", fontsize=12)
ax.set_title("Catastrophic Forgetting Rate by Method Across Benchmark Scenarios (2024-2026)", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(method_keys, rotation=20, ha='right', fontsize=10)
ax.legend(title="Scenario", fontsize=10, title_fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.35)
plt.tight_layout()
plt.savefig(charts_dir / "chart1.png", dpi=100)
plt.close()

# ============================================================
# CHART 2: Forgetting Trend Over Time (Line Chart)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
markers = ['o', 's', '^', 'D', 'v', 'P']
colors = plt.cm.tab10(np.linspace(0, 1, len(forgetting_trend)))

for i, (method, vals) in enumerate(forgetting_trend.items()):
    ax.plot(years, vals, marker=markers[i], markersize=8, linewidth=2.5,
            label=method, color=colors[i], alpha=0.9)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Average Forgetting Rate", fontsize=12)
ax.set_title("Evolution of Catastrophic Forgetting Rates by Method Category (2020-2026)", fontsize=14, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.5)
plt.tight_layout()
plt.savefig(charts_dir / "chart2.png", dpi=100)
plt.close()

# ============================================================
# CHART 3: Production Trade-offs (Scatter: Latency vs Retention)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
methods_deploy = list(deployment_metrics.keys())
latencies = [deployment_metrics[m]["latency_ms"] for m in methods_deploy]
retentions = [deployment_metrics[m]["retention"] for m in methods_deploy]
memories = [deployment_metrics[m]["memory_mb"] for m in methods_deploy]

# Size points by memory usage
sizes = np.array(memories) / np.max(memories) * 800 + 200
scatter = ax.scatter(latencies, retentions, s=sizes, c=range(len(methods_deploy)), 
                     cmap='viridis', alpha=0.7, edgecolors='white', linewidth=1.5)

# Annotate each point
for i, method in enumerate(methods_deploy):
    ax.annotate(method, (latencies[i], retentions[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

ax.set_xlabel("Inference Latency (ms)", fontsize=12)
ax.set_ylabel("Accuracy Retention After 10 Tasks", fontsize=12)
ax.set_title("Production Deployment Trade-offs: Latency vs. Retention (bubble size = memory)", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 55)
ax.set_ylim(0.75, 0.98)
plt.tight_layout()
plt.savefig(charts_dir / "chart3.png", dpi=100)
plt.close()

# ============================================================
# CHART 4: Plasticity vs Stability Trade-off (Scatter)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
# Aggregate across scenarios for each method
plasticity_vals = []
forgetting_vals = []
for method in method_keys:
    p = np.mean([results[s][method]["plasticity"] for s in scenarios])
    f = np.mean([results[s][method]["forgetting"] for s in scenarios])
    plasticity_vals.append(p)
    forgetting_vals.append(f)

scatter = ax.scatter(forgetting_vals, plasticity_vals, s=300, c=range(len(method_keys)), 
                     cmap='plasma', alpha=0.8, edgecolors='white', linewidth=1.5)

for i, method in enumerate(method_keys):
    ax.annotate(method, (forgetting_vals[i], plasticity_vals[i]), 
                xytext=(8, 8), textcoords='offset points', fontsize=9, fontweight='bold')

ax.set_xlabel("Average Forgetting Rate (Stability)", fontsize=12)
ax.set_ylabel("Average Plasticity (New Task Learning)", fontsize=12)
ax.set_title("Stability-Plasticity Trade-off Across Continual Learning Methods", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 0.2)
ax.set_ylim(0.6, 0.85)
plt.tight_layout()
plt.savefig(charts_dir / "chart4.png", dpi=100)
plt.close()

# ============================================================
# CHART 5: Overhead vs Benefit (Radar Chart)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))

categories = ['Low Forgetting', 'High Plasticity', 'Low Overhead', 'Fast Inference', 'Low Memory']
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Map deployment methods to base method keys
deploy_to_base = {
    "Regularization (EWC)": "Regularization",
    "Replay (Experience)": "Replay",
    "Architectural (PackNet)": "Architectural",
    "Prompt-Based (L2P)": "Prompt-Based",
    "Parameter-Isolation (LoRA)": "Parameter-Isolation",
    "Hybrid (COPE)": "Hybrid",
}

for i, method in enumerate(methods_deploy):
    base_method = deploy_to_base[method]
    metrics = deployment_metrics[method]
    # Invert forgetting/latency/memory (lower is better)
    vals = [
        1 - np.mean([results[s][base_method]["forgetting"] for s in scenarios]),  # Low forgetting
        np.mean([results[s][base_method]["plasticity"] for s in scenarios]),       # High plasticity
        1 - methods[base_method]["overhead"],                                       # Low overhead
        1 - metrics["latency_ms"] / 55,                                             # Fast inference
        1 - metrics["memory_mb"] / 850,                                             # Low memory
    ]
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=2, label=method, alpha=0.8)
    ax.fill(angles, vals, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title("Multi-Dimensional Method Comparison (Normalized, Higher=Better)", fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(charts_dir / "chart5.png", dpi=100)
plt.close()

# ============================================================
# COMPUTE AND SAVE METRICS JSON
# ============================================================
metrics = {
    "benchmark_summary": {
        "methods_evaluated": len(method_keys),
        "scenarios_tested": len(scenarios),
        "year_range": [2020, 2026],
    },
    "forgetting_rates_by_method": {
        method: {
            "mean": float(np.mean([results[s][method]["forgetting"] for s in scenarios])),
            "std": float(np.std([results[s][method]["forgetting"] for s in scenarios])),
            "by_scenario": {s: float(results[s][method]["forgetting"]) for s in scenarios}
        }
        for method in method_keys
    },
    "plasticity_by_method": {
        method: {
            "mean": float(np.mean([results[s][method]["plasticity"] for s in scenarios])),
            "std": float(np.std([results[s][method]["plasticity"] for s in scenarios])),
        }
        for method in method_keys
    },
    "deployment_metrics": deployment_metrics,
    "yearly_trend_forgetting": {
        method: {int(y): float(v) for y, v in zip(years, vals)}
        for method, vals in forgetting_trend.items()
    },
    "best_overall_method": min(method_keys, key=lambda m: np.mean([results[s][m]["forgetting"] for s in scenarios])),
    "best_production_method": max(deployment_metrics, key=lambda m: deployment_metrics[m]["retention"]),
}

with open("./results.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Generated 5 charts in ./charts/")
print("Saved metrics to ./results.json")
print("Done.")
