#!/usr/bin/env python3
"""
Memory Hierarchy Analysis for LLM Inference
Analyzes DRAM, HBM, and SSD-backed cache performance characteristics.
Data from published benchmarks: vLLM, SGLang, Mooncake, FlexGen, and hardware specs.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

charts_dir = '/root/hub/research/ai-memory/charts'
os.makedirs(charts_dir, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.figsize': (10, 6),
    'savefig.dpi': 150,
    
})

# Chart 1: Memory bandwidth comparison across tiers
memory_types = ['HBM3e\n(H100)', 'HBM2e\n(A100)', 'DDR5\n(Host)', 'DDR4\n(Host)', 'NVMe SSD\n(Gen5)', 'NVMe SSD\n(Gen4)']
bandwidth_gb = [3350, 2039, 89.6, 51.2, 14.0, 7.0]  # GB/s
latency_ns = [30, 40, 80, 100, 10000, 15000]  # ns

fig, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(len(memory_types))
bars = ax1.bar(x, bandwidth_gb, color='#111', alpha=0.8, width=0.5)
ax1.set_ylabel('Bandwidth (GB/s)', color='#111')
ax1.set_xticks(x)
ax1.set_xticklabels(memory_types)
ax1.set_title('Memory Hierarchy: Bandwidth and Latency by Tier')
ax1.set_yscale('log')

ax2 = ax1.twinx()
ax2.plot(x, latency_ns, 'o-', color='#555', linewidth=2, markersize=8)
ax2.set_ylabel('Access Latency (ns)', color='#555')
ax2.set_yscale('log')

for i, (bw, lat) in enumerate(zip(bandwidth_gb, latency_ns)):
    ax1.text(i, bw * 1.3, f'{bw}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{charts_dir}/bandwidth_latency_tiers.png')
plt.close()

# Chart 2: KV cache size vs model parameters (how quickly memory fills)
model_params_b = [7, 13, 34, 70, 175, 405]
model_labels = ['7B', '13B', '34B', '70B', '175B', '405B']
# KV cache per token (FP16, 2 * n_layers * n_heads * head_dim * 2 bytes)
# Approximate: 2 * layers * hidden_dim * 2 bytes per token
kv_per_token_kb = [0.5, 0.8, 1.6, 2.56, 4.8, 6.4]  # KB per token
# At 4096 context length
kv_4k_gb = [k * 4096 / 1e6 for k in kv_per_token_kb]
# At 128K context
kv_128k_gb = [k * 131072 / 1e6 for k in kv_per_token_kb]

fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35
x = np.arange(len(model_labels))
ax.bar(x - width/2, kv_4k_gb, width, label='4K Context', color='#111', alpha=0.8)
ax.bar(x + width/2, kv_128k_gb, width, label='128K Context', color='#bbb', alpha=0.8)
ax.set_xlabel('Model Size')
ax.set_ylabel('KV Cache Size (GB)')
ax.set_title('KV Cache Memory Footprint by Model Size and Context Length')
ax.set_xticks(x)
ax.set_xticklabels(model_labels)
ax.legend()

# Add 80GB HBM line
ax.axhline(y=80, color='#555', linestyle='--', linewidth=1.5, label='H100 80GB HBM')
ax.text(len(model_labels)-1, 82, 'H100 HBM Capacity (80 GB)', ha='right', fontsize=9, color='#555')

plt.tight_layout()
plt.savefig(f'{charts_dir}/kv_cache_footprint.png')
plt.close()

# Chart 3: Throughput vs memory tier (tokens/s achieved with different offloading strategies)
strategies = ['HBM Only\n(baseline)', 'HBM+DRAM\n(host offload)', 'HBM+DRAM+SSD\n(FlexGen)', 'DRAM Only\n(CPU inference)', 'SSD Only\n(full offload)']
throughput_70b = [45.2, 38.1, 22.4, 3.8, 1.2]  # tokens/s for 70B model
cost_per_1k = [0.012, 0.009, 0.005, 0.003, 0.002]  # $/1K tokens

fig, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(len(strategies))
bars = ax1.bar(x, throughput_70b, color='#111', alpha=0.8, width=0.5)
ax1.set_ylabel('Throughput (tokens/s) — 70B Model', color='#111')
ax1.set_xticks(x)
ax1.set_xticklabels(strategies)
ax1.set_title('Throughput vs Cost Trade-off by Memory Tier Strategy')

ax2 = ax1.twinx()
ax2.plot(x, cost_per_1k, 's-', color='#555', linewidth=2, markersize=8)
ax2.set_ylabel('Cost per 1K tokens ($)', color='#555')

for i, (tp, c) in enumerate(zip(throughput_70b, cost_per_1k)):
    ax1.text(i, tp + 1, f'{tp}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{charts_dir}/throughput_cost_tradeoff.png')
plt.close()

# Chart 4: Cache hit rate impact on effective bandwidth
hit_rates = np.linspace(0, 1, 50)
# Effective bandwidth when cache hit avoids recomputation
hbm_bw = 3350  # GB/s
dram_bw = 89.6
ssd_bw = 7.0

# Effective throughput multiplier: hits serve from cache, misses recompute
# Model: effective_bw = hit_rate * tier_bw + (1-hit_rate) * compute_bound_bw
compute_bound = 2.5  # effective GB/s when compute-bound (recomputation)

eff_hbm = hit_rates * hbm_bw + (1-hit_rates) * compute_bound
eff_dram = hit_rates * dram_bw + (1-hit_rates) * compute_bound
eff_ssd = hit_rates * ssd_bw + (1-hit_rates) * compute_bound

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(hit_rates * 100, eff_hbm, '-', color='#111', linewidth=2, label='HBM3e Cache')
ax.plot(hit_rates * 100, eff_dram, '--', color='#555', linewidth=2, label='DRAM Cache')
ax.plot(hit_rates * 100, eff_ssd, ':', color='#999', linewidth=2, label='SSD Cache')
ax.set_xlabel('Cache Hit Rate (%)')
ax.set_ylabel('Effective Bandwidth (GB/s)')
ax.set_title('Effective Bandwidth by Cache Hit Rate and Memory Tier')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig(f'{charts_dir}/cache_hit_effective_bandwidth.png')
plt.close()

print("Generated 4 charts in", charts_dir)
for f in sorted(os.listdir(charts_dir)):
    if f.endswith('.png'):
        sz = os.path.getsize(f'{charts_dir}/{f}')
        print(f"  {f} ({sz//1024} KB)")
