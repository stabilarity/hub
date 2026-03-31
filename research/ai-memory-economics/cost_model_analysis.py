"""
Context Caching Cost Model Analysis
AI Memory Series #25 — The Economics of Context Caching
Stabilarity Research Hub, 2026

Analyzes break-even points for KV cache reuse vs recomputation
using published API pricing and memory cost models.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

CHARTS_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.facecolor': '#fafafa',
    'figure.facecolor': '#ffffff',
    'axes.edgecolor': '#555',
    'axes.labelcolor': '#111',
    'xtick.color': '#555',
    'ytick.color': '#555',
    'grid.color': '#ddd',
    'grid.alpha': 0.7,
})

# ============================================================
# CHART 1: Break-Even Analysis — Cache Hit Rate vs Cost Savings
# Based on Google Gemini / OpenAI / Anthropic published pricing (2025-2026)
# ============================================================

# Real pricing data (USD per 1M tokens) — from public API docs
providers = {
    'OpenAI GPT-4o': {'input': 2.50, 'cached_input': 1.25, 'output': 10.00},
    'OpenAI GPT-4.1': {'input': 2.00, 'cached_input': 0.50, 'output': 8.00},
    'Claude 3.5 Sonnet': {'input': 3.00, 'cached_input': 0.30, 'output': 15.00},
    'Gemini 2.0 Flash': {'input': 0.10, 'cached_input': 0.025, 'output': 0.40},
    'DeepSeek V3': {'input': 0.27, 'cached_input': 0.07, 'output': 1.10},
}

hit_rates = np.linspace(0, 1.0, 100)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#111', '#555', '#999', '#bbb', '#333']
linestyles = ['-', '--', '-.', ':', '-']

for i, (name, p) in enumerate(providers.items()):
    savings_pct = hit_rates * (1 - p['cached_input'] / p['input']) * 100
    ax.plot(hit_rates * 100, savings_pct, label=name, color=colors[i],
            linestyle=linestyles[i], linewidth=2)

ax.set_xlabel('Cache Hit Rate (%)')
ax.set_ylabel('Input Cost Savings (%)')
ax.set_title('Input Cost Savings vs Cache Hit Rate by Provider (2025-2026 Pricing)')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.5)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/break_even_hit_rate.png', dpi=150)
plt.close()
print("Chart 1: break_even_hit_rate.png")

# ============================================================
# CHART 2: KV Cache Memory Cost vs Context Length
# Memory formula: mem_bytes = 2 * n_layers * d_model * seq_len * batch * dtype_bytes
# ============================================================

models_specs = {
    'Llama 3.1 8B': {'layers': 32, 'heads': 32, 'head_dim': 128, 'dtype': 2},
    'Llama 3.1 70B': {'layers': 80, 'heads': 64, 'head_dim': 128, 'dtype': 2},
    'Mixtral 8x22B': {'layers': 56, 'heads': 48, 'head_dim': 128, 'dtype': 2},
    'GPT-4 class (est.)': {'layers': 120, 'heads': 96, 'head_dim': 128, 'dtype': 2},
}

seq_lengths = np.array([1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])

fig, ax = plt.subplots(figsize=(10, 6))

for i, (name, spec) in enumerate(models_specs.items()):
    # KV cache per token per layer = 2 * head_dim * n_kv_heads * dtype_bytes
    # Total = n_layers * above * seq_len
    kv_per_token = 2 * spec['heads'] * spec['head_dim'] * spec['dtype']  # bytes per layer per token
    mem_gb = spec['layers'] * kv_per_token * seq_lengths / (1024**3)
    ax.plot(seq_lengths / 1024, mem_gb, label=name, color=colors[i],
            linestyle=linestyles[i], linewidth=2, marker='o', markersize=4)

ax.set_xlabel('Context Length (K tokens)')
ax.set_ylabel('KV Cache Memory (GB)')
ax.set_title('KV Cache Memory Requirements vs Context Length (Single Request, FP16)')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.5)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
ax.set_xticklabels(['1K', '2K', '4K', '8K', '16K', '32K', '64K', '128K'])
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/kv_cache_memory_cost.png', dpi=150)
plt.close()
print("Chart 2: kv_cache_memory_cost.png")

# ============================================================
# CHART 3: Break-Even Reuse Count — When Does Caching Pay Off?
# Cost_cache = prefill_cost + n * cached_cost + storage_cost
# Cost_no_cache = n * prefill_cost
# Break-even: n = storage_cost / (prefill_cost - cached_cost)
# ============================================================

# GPU memory cost: ~$2/GB/hour for A100 80GB (cloud pricing)
# Storage cost per cached context = mem_gb * $/GB/hour * TTL_hours
gpu_cost_per_gb_hour = 2.0  # USD, typical cloud A100

context_lengths_k = np.array([4, 8, 16, 32, 64, 128])
ttls_hours = [0.5, 1.0, 2.0, 4.0]

# Use Llama 3.1 70B as reference
spec = models_specs['Llama 3.1 70B']
kv_per_token = 2 * spec['heads'] * spec['head_dim'] * spec['dtype']

fig, ax = plt.subplots(figsize=(10, 6))

for j, ttl in enumerate(ttls_hours):
    breakeven_counts = []
    for ctx_k in context_lengths_k:
        seq_len = ctx_k * 1024
        mem_gb = spec['layers'] * kv_per_token * seq_len / (1024**3)
        storage_cost = mem_gb * gpu_cost_per_gb_hour * ttl
        # Prefill cost savings per reuse ≈ proportional to tokens saved
        # At $0.002/1K input tokens (70B class self-hosted estimate)
        prefill_cost_per_req = seq_len * 0.000002  # $2/1M tokens
        cached_cost_per_req = seq_len * 0.0000005  # 75% savings
        savings_per_reuse = prefill_cost_per_req - cached_cost_per_req
        if savings_per_reuse > 0:
            n_breakeven = storage_cost / savings_per_reuse
        else:
            n_breakeven = float('inf')
        breakeven_counts.append(n_breakeven)
    ax.plot(context_lengths_k, breakeven_counts, label=f'TTL = {ttl}h',
            color=colors[j], linestyle=linestyles[j], linewidth=2, marker='s', markersize=5)

ax.set_xlabel('Context Length (K tokens)')
ax.set_ylabel('Break-Even Reuse Count')
ax.set_title('Cache Break-Even: Minimum Reuses to Offset Storage Cost (Llama 3.1 70B)')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.5)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/breakeven_reuse_count.png', dpi=150)
plt.close()
print("Chart 3: breakeven_reuse_count.png")

# ============================================================
# CHART 4: Total Cost of Ownership — Cached vs Uncached at Scale
# Monthly cost projection for different request volumes
# ============================================================

monthly_requests = np.array([1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6])
avg_context_k = 16  # 16K average context
avg_seq = avg_context_k * 1024

# Self-hosted scenario: $2/1M input tokens, $0.50/1M cached
cost_per_req_uncached = avg_seq * 2.0 / 1e6
cost_per_req_cached = avg_seq * 0.50 / 1e6

# Fixed costs: GPU memory for cache pool (~40GB reserved)
monthly_cache_infra = 40 * gpu_cost_per_gb_hour * 730  # 730 hours/month

hit_rate = 0.65  # typical production hit rate

fig, ax = plt.subplots(figsize=(10, 6))

uncached_monthly = monthly_requests * cost_per_req_uncached
cached_monthly = (monthly_requests * (1 - hit_rate) * cost_per_req_uncached +
                  monthly_requests * hit_rate * cost_per_req_cached +
                  monthly_cache_infra)

ax.plot(monthly_requests / 1000, uncached_monthly, label='Without Caching',
        color='#111', linewidth=2.5, linestyle='-')
ax.plot(monthly_requests / 1000, cached_monthly, label='With Caching (65% hit rate)',
        color='#555', linewidth=2.5, linestyle='--')
ax.fill_between(monthly_requests / 1000, cached_monthly, uncached_monthly,
                where=uncached_monthly > cached_monthly, alpha=0.15, color='#999',
                label='Cost Savings')

ax.set_xlabel('Monthly Requests (thousands)')
ax.set_ylabel('Monthly Cost (USD)')
ax.set_title('Total Monthly Inference Cost: Cached vs Uncached (16K Avg Context, 70B Model)')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.5)
ax.set_xscale('log')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/tco_cached_vs_uncached.png', dpi=150)
plt.close()
print("Chart 4: tco_cached_vs_uncached.png")

# Save data summary
summary = {
    'providers': {k: v for k, v in providers.items()},
    'break_even_example': {
        'model': 'Llama 3.1 70B',
        'context_16k_mem_gb': float(spec['layers'] * kv_per_token * 16384 / (1024**3)),
        'monthly_cache_infra_40gb': monthly_cache_infra,
        'crossover_requests': int(np.interp(0, uncached_monthly - cached_monthly, monthly_requests)) if any(uncached_monthly > cached_monthly) else 'N/A'
    }
}
with open(f'{CHARTS_DIR}/../data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\nData Summary:")
print(f"  Llama 70B KV cache at 16K context: {summary['break_even_example']['context_16k_mem_gb']:.2f} GB")
print(f"  Monthly cache infra (40GB pool): ${monthly_cache_infra:,.0f}")
print("Done.")
