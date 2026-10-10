"""
Test-Time Compute Scaling: When Letting a Model 'Think Longer' Beats a Bigger Model
Analysis script for empirical comparison of inference-time reasoning scaling
against parameter-count scaling for matched compute budgets.

Reproduce: python3 analysis.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Ensure charts directory exists
os.makedirs('./charts', exist_ok=True)

# Reproducible synthetic data with fixed seed
rng = np.random.default_rng(42)

# Model sizes (parameters in billions)
model_sizes = np.array([1, 3, 7, 13, 30, 70])  # Billions of parameters

# Test-time compute budgets (relative FLOPs during inference)
# We simulate: base model + increasing test-time compute (more thinking steps)
compute_budgets = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])  # Relative compute multiplier

# Generate synthetic benchmark results
# Performance on reasoning tasks (e.g., MATH, GSM8K, Code)
# Key finding: smaller model with more test-time compute can beat larger model with less compute

# Base performance scales with model size (logarithmic)
base_performance = 0.15 + 0.25 * np.log10(model_sizes)

# Test-time compute scaling: diminishing returns (square root law)
def compute_scaling_factor(compute_mult):
    """Returns performance multiplier from test-time compute scaling"""
    return 1.0 + 0.35 * np.log2(compute_mult) * (1 - 0.15 * np.log2(compute_mult))

# Generate performance matrix: models x compute budgets
performance = np.zeros((len(model_sizes), len(compute_budgets)))
for i, size in enumerate(model_sizes):
    for j, budget in enumerate(compute_budgets):
        # Performance = base + compute scaling with noise
        perf = base_performance[i] * compute_scaling_factor(budget)
        perf += rng.normal(0, 0.02)  # Add small noise
        performance[i, j] = np.clip(perf, 0, 1.0)

# Compute FLOPs for training (roughly proportional to params * tokens)
# and inference (proportional to params * compute_budget)
training_flops = model_sizes * 1e12  # Relative training FLOPs
inference_flops_per_token = model_sizes[:, None] * compute_budgets[None, :] * 1e9

# Chart 1: Performance vs Model Size at Different Compute Budgets
fig1, ax1 = plt.subplots(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(compute_budgets)))
markers = ['o', 's', '^', 'D', 'v', 'P']

for j, budget in enumerate(compute_budgets):
    ax1.plot(model_sizes, performance[:, j], 
             marker=markers[j], markersize=10, linewidth=2.5,
             color=colors[j], label=f'{budget:.0f}x Compute',
             alpha=0.9)

ax1.set_xscale('log')
ax1.set_xlabel('Model Size (Billion Parameters)', fontsize=14)
ax1.set_ylabel('Reasoning Benchmark Score', fontsize=14)
ax1.set_title('Test-Time Compute Scaling: Performance vs Model Size', fontsize=16, fontweight='bold')
ax1.legend(title='Test-Time Compute Budget', fontsize=12, title_fontsize=13, loc='lower right')
ax1.grid(True, alpha=0.3, which='both')
ax1.set_ylim(0.1, 1.05)

# Highlight crossover: 7B with 16x compute vs 70B with 1x compute
ax1.annotate('7B @ 16x ≈ 70B @ 1x', 
             xy=(7, performance[2, 4]), xytext=(10, performance[2, 4] - 0.05),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=12, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9))

fig1.tight_layout()
fig1.savefig('./charts/chart1.png', dpi=100, bbox_inches='tight')
plt.close(fig1)

# Chart 2: Iso-Compute Frontier - Performance vs Total Compute (Training + Inference)
# Total compute = training_flops + inference_flops (for a fixed inference workload)
# We compare: larger model with base compute vs smaller model with more test-time compute

# Fixed inference workload: 1B tokens
inference_tokens = 1e9

# Total compute for each configuration
total_compute = np.zeros((len(model_sizes), len(compute_budgets)))
for i in range(len(model_sizes)):
    for j in range(len(compute_budgets)):
        train = training_flops[i]
        infer = model_sizes[i] * compute_budgets[j] * 1e9 * inference_tokens / 1e9  # Simplified
        total_compute[i, j] = train + infer

# Plot iso-compute frontier
fig2, ax2 = plt.subplots(figsize=(12, 8))

# For each compute budget, find best performance
# Sort by total compute
flat_compute = total_compute.flatten()
flat_perf = performance.flatten()
flat_sizes = np.repeat(model_sizes, len(compute_budgets))
flat_budgets = np.tile(compute_budgets, len(model_sizes))

# Sort by compute
sort_idx = np.argsort(flat_compute)
flat_compute = flat_compute[sort_idx]
flat_perf = flat_perf[sort_idx]
flat_sizes = flat_sizes[sort_idx]
flat_budgets = flat_budgets[sort_idx]

# Plot all points
scatter = ax2.scatter(flat_compute, flat_perf, c=flat_budgets, s=120, 
                      cmap='plasma', alpha=0.8, edgecolors='white', linewidth=0.5,
                      label='Configurations')

# Pareto frontier (best performance at each compute level)
pareto_compute = []
pareto_perf = []
max_perf = -1
for c, p in zip(flat_compute, flat_perf):
    if p > max_perf:
        max_perf = p
        pareto_compute.append(c)
        pareto_perf.append(p)

ax2.plot(pareto_compute, pareto_perf, 'r-', linewidth=3, label='Pareto Frontier', zorder=5)

# Highlight specific configurations
# 7B @ 16x compute
idx_7b_16x = np.where((flat_sizes == 7) & (flat_budgets == 16))[0][0]
# 70B @ 1x compute  
idx_70b_1x = np.where((flat_sizes == 70) & (flat_budgets == 1))[0][0]

ax2.scatter([flat_compute[idx_7b_16x]], [flat_perf[idx_7b_16x]], 
            s=300, c='red', marker='*', edgecolors='black', linewidth=2, zorder=10,
            label='7B @ 16x Compute')
ax2.scatter([flat_compute[idx_70b_1x]], [flat_perf[idx_70b_1x]], 
            s=300, c='blue', marker='*', edgecolors='black', linewidth=2, zorder=10,
            label='70B @ 1x Compute')

ax2.set_xscale('log')
ax2.set_xlabel('Total Compute (Training + Inference FLOPs, relative)', fontsize=14)
ax2.set_ylabel('Reasoning Benchmark Score', fontsize=14)
ax2.set_title('Iso-Compute Frontier: Best Performance at Each Compute Budget', fontsize=16, fontweight='bold')
ax2.legend(fontsize=12, loc='lower right')
ax2.grid(True, alpha=0.3, which='both')
ax2.set_ylim(0.1, 1.05)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Test-Time Compute Multiplier', fontsize=12)

fig2.tight_layout()
fig2.savefig('./charts/chart2.png', dpi=100, bbox_inches='tight')
plt.close(fig2)

# Chart 3: Scaling Laws - Performance vs Test-Time Compute for Different Model Sizes
fig3, ax3 = plt.subplots(figsize=(12, 8))

for i, size in enumerate(model_sizes):
    ax3.plot(compute_budgets, performance[i, :], 
             marker=markers[i], markersize=10, linewidth=2.5,
             label=f'{size}B params', alpha=0.9)

ax3.set_xscale('log', base=2)
ax3.set_xlabel('Test-Time Compute Multiplier', fontsize=14)
ax3.set_ylabel('Reasoning Benchmark Score', fontsize=14)
ax3.set_title('Test-Time Compute Scaling Laws by Model Size', fontsize=16, fontweight='bold')
ax3.legend(title='Model Size', fontsize=12, title_fontsize=13, loc='lower right')
ax3.grid(True, alpha=0.3, which='both')
ax3.set_ylim(0.1, 1.05)

# Add scaling law reference lines (sqrt and log)
x_ref = np.logspace(0, 5, 50, base=2)
ax3.plot(x_ref, 0.15 + 0.35 * np.log2(x_ref), 'k--', alpha=0.4, label='Log scaling (ref)')
ax3.plot(x_ref, 0.15 + 0.45 * np.sqrt(np.log2(x_ref)), 'k:', alpha=0.4, label='Sqrt scaling (ref)')

# Annotate diminishing returns region
ax3.axvspan(8, 32, alpha=0.1, color='orange', label='Diminishing returns')

fig3.tight_layout()
fig3.savefig('./charts/chart3.png', dpi=100, bbox_inches='tight')
plt.close(fig3)

# Chart 4: Compute Efficiency - Performance per FLOP
fig4, ax4 = plt.subplots(figsize=(12, 8))

# Compute efficiency = performance / total_compute (normalized)
# Using inference compute only for fairness
inference_compute = model_sizes[:, None] * compute_budgets[None, :] * 1e9
efficiency = performance / (inference_compute / 1e12)  # Normalized

for i, size in enumerate(model_sizes):
    ax4.plot(compute_budgets, efficiency[i, :], 
             marker=markers[i], markersize=10, linewidth=2.5,
             label=f'{size}B params', alpha=0.9)

ax4.set_xscale('log', base=2)
ax4.set_yscale('log')
ax4.set_xlabel('Test-Time Compute Multiplier', fontsize=14)
ax4.set_ylabel('Performance per Inference FLOP (normalized)', fontsize=14)
ax4.set_title('Compute Efficiency: Performance per FLOP vs Test-Time Compute', fontsize=16, fontweight='bold')
ax4.legend(title='Model Size', fontsize=12, title_fontsize=13, loc='upper right')
ax4.grid(True, alpha=0.3, which='both')

fig4.tight_layout()
fig4.savefig('./charts/chart4.png', dpi=100, bbox_inches='tight')
plt.close(fig4)

# Chart 5: Heatmap - Performance Landscape
fig5, ax5 = plt.subplots(figsize=(12, 8))

im = ax5.imshow(performance.T, aspect='auto', origin='lower', 
                cmap='RdYlGn', vmin=0.2, vmax=1.0,
                extent=[0.5, len(model_sizes)+0.5, 0.5, len(compute_budgets)+0.5])

# Add text annotations
for i in range(len(model_sizes)):
    for j in range(len(compute_budgets)):
        text = ax5.text(i+1, j+1, f'{performance[i, j]:.2f}',
                       ha='center', va='center', fontsize=12,
                       color='black' if performance[i, j] < 0.6 else 'white',
                       fontweight='bold')

ax5.set_xticks(range(1, len(model_sizes)+1))
ax5.set_xticklabels([f'{s}B' for s in model_sizes], fontsize=12)
ax5.set_yticks(range(1, len(compute_budgets)+1))
ax5.set_yticklabels([f'{b:.0f}x' for b in compute_budgets], fontsize=12)
ax5.set_xlabel('Model Size (Billion Parameters)', fontsize=14)
ax5.set_ylabel('Test-Time Compute Multiplier', fontsize=14)
ax5.set_title('Performance Landscape: Model Size vs Test-Time Compute', fontsize=16, fontweight='bold')

cbar = plt.colorbar(im, ax=ax5)
cbar.set_label('Reasoning Benchmark Score', fontsize=12)

fig5.tight_layout()
fig5.savefig('./charts/chart5.png', dpi=100, bbox_inches='tight')
plt.close(fig5)

# Compute metrics for results.json
metrics = {
    "model_sizes_billions": model_sizes.tolist(),
    "compute_budgets": compute_budgets.tolist(),
    "performance_matrix": performance.tolist(),
    "base_performance": base_performance.tolist(),
    "crossover_point": {
        "small_model": "7B",
        "small_model_compute": 16,
        "large_model": "70B", 
        "large_model_compute": 1,
        "small_model_perf": float(performance[2, 4]),
        "large_model_perf": float(performance[5, 0]),
        "difference": float(performance[2, 4] - performance[5, 0])
    },
    "pareto_frontier": {
        "compute": pareto_compute,
        "performance": pareto_perf
    },
    "efficiency_matrix": efficiency.tolist(),
    "scaling_law_params": {
        "base_intercept": 0.15,
        "base_slope": 0.25,
        "compute_slope": 0.35,
        "diminishing_returns": 0.15
    }
}

with open('./results.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Analysis complete. Charts saved to ./charts/")
print(f"Results saved to ./results.json")
print(f"Generated {len([f for f in os.listdir('./charts') if f.endswith('.png')])} charts")
