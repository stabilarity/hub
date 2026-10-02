"""
Spec Drift Detection: Automatically Flagging When Implementation Diverges from Its Natural-Language Spec
Reproducible analysis script generating charts for the article.

Reproduce: python3 analysis.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
rng = np.random.default_rng(42)

# Ensure charts directory exists
charts_dir = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(charts_dir, exist_ok=True)

# ============================================================
# Synthetic Data Generation
# ============================================================

# Chart 1: Spec Conformance Score Over Time (Drift Detection)
# Simulates conformance scores of 5 AI systems tracked over 20 time periods
n_periods = 20
n_systems = 5
time_periods = np.arange(1, n_periods + 1)

# Each system starts at high conformance (~0.95) and drifts at different rates
drift_rates = rng.uniform(0.005, 0.03, n_systems)
conformance_data = {}
for i in range(n_systems):
    base = 0.95
    # Add some noise
    noise = rng.normal(0, 0.01, n_periods)
    # Drift with occasional recovery events
    drift = -drift_rates[i] * time_periods + noise
    # Clip to valid range
    scores = np.clip(base + drift, 0.3, 1.0)
    conformance_data[f'System {i+1}'] = scores

# Chart 2: Detection Method Comparison
# Precision, Recall, F1 for 4 detection methods
methods = ['LLM-based', 'Embedding\nSimilarity', 'Rule-based', 'Hybrid\n(Proposed)']
precision = np.array([0.72, 0.68, 0.81, 0.87])
recall = np.array([0.65, 0.71, 0.58, 0.84])
f1 = 2 * (precision * recall) / (precision + recall)

# Chart 3: Drift Type Distribution
# Types of spec drift detected across all systems
drift_types = ['Behavioral\nDeviation', 'API Signature\nChange', 'Logic\nInversion', 'Missing\nConstraint', 'Performance\nRegression']
drift_counts = np.array([42, 28, 15, 19, 12])
drift_pct = drift_counts / drift_counts.sum() * 100

# Chart 4: False Positive Rate by Detection Threshold
thresholds = np.linspace(0.1, 0.9, 17)
# Synthetic FPR curves for different methods
fpr_llm = 0.35 * np.exp(-3 * thresholds) + 0.02 + rng.normal(0, 0.005, len(thresholds))
fpr_embed = 0.28 * np.exp(-2.5 * thresholds) + 0.03 + rng.normal(0, 0.005, len(thresholds))
fpr_rule = 0.15 * np.exp(-2 * thresholds) + 0.05 + rng.normal(0, 0.005, len(thresholds))
fpr_hybrid = 0.12 * np.exp(-3.5 * thresholds) + 0.01 + rng.normal(0, 0.005, len(thresholds))
fpr_llm = np.clip(fpr_llm, 0, 0.5)
fpr_embed = np.clip(fpr_embed, 0, 0.5)
fpr_rule = np.clip(fpr_rule, 0, 0.5)
fpr_hybrid = np.clip(fpr_hybrid, 0, 0.5)

# Chart 5: Time-to-Detection (Days) by Drift Severity
severity_levels = ['Low', 'Medium', 'High', 'Critical']
ttd_llm = [14, 9, 5, 2]
ttd_embed = [12, 7, 4, 2]
ttd_rule = [10, 6, 3, 1]
ttd_hybrid = [7, 4, 2, 1]

# ============================================================
# Chart 1: Conformance Score Over Time
# ============================================================
fig1, ax1 = plt.subplots(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, n_systems))
for idx, (system, scores) in enumerate(conformance_data.items()):
    ax1.plot(time_periods, scores, marker='o', markersize=4, label=system, color=colors[idx], linewidth=2)
# Add drift threshold line
ax1.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Drift Alert Threshold (0.70)')
ax1.set_xlabel('Time Period', fontsize=14)
ax1.set_ylabel('Spec Conformance Score', fontsize=14)
ax1.set_title('Spec Conformance Score Over Time Across AI Systems', fontsize=16, fontweight='bold')
ax1.set_ylim(0.25, 1.05)
ax1.legend(loc='lower left', fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
chart1_path = os.path.join(charts_dir, 'chart1.png')
plt.savefig(chart1_path, dpi=100)
plt.close()

# ============================================================
# Chart 2: Detection Method Comparison (Precision, Recall, F1)
# ============================================================
fig2, ax2 = plt.subplots(figsize=(12, 8))
x = np.arange(len(methods))
width = 0.25
bars1 = ax2.bar(x - width, precision, width, label='Precision', color='#2ecc71', edgecolor='black')
bars2 = ax2.bar(x, recall, width, label='Recall', color='#3498db', edgecolor='black')
bars3 = ax2.bar(x + width, f1, width, label='F1 Score', color='#e74c3c', edgecolor='black')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('Score', fontsize=14)
ax2.set_title('Detection Method Performance Comparison', fontsize=16, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=12)
ax2.set_ylim(0, 1.1)
ax2.legend(fontsize=12)
ax2.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
chart2_path = os.path.join(charts_dir, 'chart2.png')
plt.savefig(chart2_path, dpi=100)
plt.close()

# ============================================================
# Chart 3: Drift Type Distribution (Pie Chart)
# ============================================================
fig3, ax3 = plt.subplots(figsize=(12, 8))
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(drift_types)))
wedges, texts, autotexts = ax3.pie(drift_counts, labels=drift_types, autopct='%1.1f%%',
                                    colors=colors_pie, startangle=90,
                                    textprops={'fontsize': 12})
for autotext in autotexts:
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')
ax3.set_title('Distribution of Spec Drift Types Detected', fontsize=16, fontweight='bold')
# Add legend with counts
legend_labels = [f'{t}: {c} ({p:.1f}%)' for t, c, p in zip(drift_types, drift_counts, drift_pct)]
ax3.legend(wedges, legend_labels, title="Drift Types", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11)
plt.tight_layout()
chart3_path = os.path.join(charts_dir, 'chart3.png')
plt.savefig(chart3_path, dpi=100)
plt.close()

# ============================================================
# Chart 4: False Positive Rate vs Detection Threshold
# ============================================================
fig4, ax4 = plt.subplots(figsize=(12, 8))
ax4.plot(thresholds, fpr_llm, marker='o', markersize=5, label='LLM-based', linewidth=2, color='#e74c3c')
ax4.plot(thresholds, fpr_embed, marker='s', markersize=5, label='Embedding Similarity', linewidth=2, color='#3498db')
ax4.plot(thresholds, fpr_rule, marker='^', markersize=5, label='Rule-based', linewidth=2, color='#f39c12')
ax4.plot(thresholds, fpr_hybrid, marker='D', markersize=5, label='Hybrid (Proposed)', linewidth=2, color='#2ecc71')
ax4.set_xlabel('Detection Threshold', fontsize=14)
ax4.set_ylabel('False Positive Rate', fontsize=14)
ax4.set_title('False Positive Rate vs. Detection Threshold by Method', fontsize=16, fontweight='bold')
ax4.set_ylim(0, 0.45)
ax4.legend(fontsize=12)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
chart4_path = os.path.join(charts_dir, 'chart4.png')
plt.savefig(chart4_path, dpi=100)
plt.close()

# ============================================================
# Chart 5: Time-to-Detection by Drift Severity
# ============================================================
fig5, ax5 = plt.subplots(figsize=(12, 8))
x = np.arange(len(severity_levels))
width = 0.2
bars1 = ax5.bar(x - 1.5*width, ttd_llm, width, label='LLM-based', color='#e74c3c', edgecolor='black')
bars2 = ax5.bar(x - 0.5*width, ttd_embed, width, label='Embedding Similarity', color='#3498db', edgecolor='black')
bars3 = ax5.bar(x + 0.5*width, ttd_rule, width, label='Rule-based', color='#f39c12', edgecolor='black')
bars4 = ax5.bar(x + 1.5*width, ttd_hybrid, width, label='Hybrid (Proposed)', color='#2ecc71', edgecolor='black')

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax5.annotate(f'{height}d',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

ax5.set_ylabel('Time to Detection (Days)', fontsize=14)
ax5.set_xlabel('Drift Severity', fontsize=14)
ax5.set_title('Time-to-Detection by Drift Severity and Method', fontsize=16, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(severity_levels, fontsize=12)
ax5.legend(fontsize=11)
ax5.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
chart5_path = os.path.join(charts_dir, 'chart5.png')
plt.savefig(chart5_path, dpi=100)
plt.close()

# ============================================================
# Persist Results JSON
# ============================================================
results = {
    "chart1_conformance_over_time": {
        "systems": list(conformance_data.keys()),
        "time_periods": time_periods.tolist(),
        "scores": {k: v.tolist() for k, v in conformance_data.items()},
        "drift_threshold": 0.70
    },
    "chart2_detection_method_comparison": {
        "methods": methods,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1_score": f1.tolist()
    },
    "chart3_drift_type_distribution": {
        "types": drift_types,
        "counts": drift_counts.tolist(),
        "percentages": drift_pct.tolist()
    },
    "chart4_fpr_vs_threshold": {
        "thresholds": thresholds.tolist(),
        "fpr_llm": fpr_llm.tolist(),
        "fpr_embedding": fpr_embed.tolist(),
        "fpr_rule": fpr_rule.tolist(),
        "fpr_hybrid": fpr_hybrid.tolist()
    },
    "chart5_ttd_by_severity": {
        "severity_levels": severity_levels,
        "ttd_llm": ttd_llm,
        "ttd_embedding": ttd_embed,
        "ttd_rule": ttd_rule,
        "ttd_hybrid": ttd_hybrid
    }
}

results_path = os.path.join(os.path.dirname(__file__), 'results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Generated 5 charts in {charts_dir}/")
print(f"Saved results to {results_path}")
print("Done.")
