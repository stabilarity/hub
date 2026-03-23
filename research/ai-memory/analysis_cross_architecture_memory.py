#!/usr/bin/env python3
"""Analysis for: Cross-Architecture Memory Comparison"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('charts', exist_ok=True)

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
    'savefig.bbox': 'tight'
})

# Chart 1: KV-Cache Size per Token
models = ['Llama 3.1\n8B', 'Llama 4\nScout 17B', 'Mistral 7B\nv0.3', 'Mistral\nSmall 3 24B',
          'Gemma 2\n9B', 'Gemma 3\n12B', 'Qwen 2.5\n7B', 'Qwen 3\n8B']

kv_bytes = np.array([131072, 196608, 131072, 163840, 172032, 196608, 57344, 73728])
effective_kv_8k = np.array([131072, 196608, 131072, 163840, 43008, 41984, 57344, 73728])
colors = ['#111111', '#555555', '#333333', '#777777', '#444444', '#999999', '#222222', '#666666']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(models))
width = 0.6

bars1 = ax1.bar(x, kv_bytes / 1024, width, color=colors, edgecolor='#000')
ax1.set_ylabel('KV-Cache per Token (KB, FP16)')
ax1.set_title('Raw KV-Cache Memory per Token')
ax1.set_xticks(x); ax1.set_xticklabels(models, fontsize=9)
for bar, val in zip(bars1, kv_bytes / 1024):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2, f'{val:.0f}', ha='center', va='bottom', fontsize=8)

bars2 = ax2.bar(x, effective_kv_8k / 1024, width, color=colors, edgecolor='#000')
ax2.set_ylabel('Effective KV-Cache per Token (KB)')
ax2.set_title('Effective KV-Cache at 8K Context\n(with Sliding Window Reduction)')
ax2.set_xticks(x); ax2.set_xticklabels(models, fontsize=9)
for bar, val in zip(bars2, effective_kv_8k / 1024):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, f'{val:.0f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout(); plt.savefig('charts/01-kv-cache-per-token-comparison.png'); plt.close()
print("Chart 1 saved")

# Chart 2: KV-Cache Growth with Context Length
context_lengths = np.array([1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
ctx_labels = ['1K', '2K', '4K', '8K', '16K', '32K', '64K', '128K']

def calc_kv(ctx, layers, kv_h, h_dim, gl=None, win=1024):
    bpl = 2 * kv_h * h_dim * 2
    if gl is None: gl = layers
    sw = layers - gl
    return (ctx * bpl * gl + min(ctx, win) * bpl * sw) / 1e9

specs = {
    'Llama 3.1 8B': (32, 8, 128, 32, None),
    'Gemma 3 12B': (48, 4, 256, 8, 1024),
    'Qwen 3 8B': (36, 4, 128, 36, None),
    'Mistral Small 24B': (40, 8, 128, 40, None),
}

fig, ax = plt.subplots(figsize=(10, 6))
ls = ['-', '--', '-.', ':']; mk = ['o', 's', '^', 'D']
for i, (name, (ly, kh, hd, gl, w)) in enumerate(specs.items()):
    vals = [calc_kv(c, ly, kh, hd, gl, w if w else 1024) for c in context_lengths]
    ax.plot(context_lengths/1024, vals, linestyle=ls[i], marker=mk[i], label=name, linewidth=2, markersize=6)

ax.set_xlabel('Context Length (K tokens)'); ax.set_ylabel('Total KV-Cache (GB, FP16)')
ax.set_title('KV-Cache Memory Growth by Context Length')
ax.legend(loc='upper left'); ax.set_xticks(context_lengths/1024); ax.set_xticklabels(ctx_labels)
ax.set_yscale('log')
plt.tight_layout(); plt.savefig('charts/02-kv-cache-growth-by-context.png'); plt.close()
print("Chart 2 saved")

# Chart 3: Query vs KV Heads
fig, ax = plt.subplots(figsize=(12, 5))
cats = ['Llama 3.1 8B', 'Llama 4 Scout', 'Mistral 7B v0.3', 'Mistral Small 3',
        'Gemma 2 9B', 'Gemma 3 12B', 'Qwen 2.5 7B', 'Qwen 3 8B']
q_h = [32, 32, 32, 32, 16, 32, 28, 32]
kv_h = [8, 8, 8, 8, 4, 4, 4, 4]
ratios = [4, 4, 4, 4, 4, 8, 7, 8]

x = np.arange(len(cats)); w = 0.35
ax.bar(x - w/2, q_h, w, label='Query Heads', color='#555555', edgecolor='#000')
ax.bar(x + w/2, kv_h, w, label='KV Heads', color='#bbb', edgecolor='#000')
ax.set_ylabel('Number of Heads'); ax.set_title('Query vs KV Head Counts Across Architectures')
ax.set_xticks(x); ax.set_xticklabels(cats, rotation=30, ha='right', fontsize=9); ax.legend()
for i, r in enumerate(ratios):
    ax.annotate(f'{r}:1', xy=(x[i], max(q_h[i], kv_h[i]) + 1), ha='center', fontsize=9, fontweight='bold')
plt.tight_layout(); plt.savefig('charts/03-query-vs-kv-heads.png'); plt.close()
print("Chart 3 saved")

# Chart 4: Memory Efficiency
fig, ax = plt.subplots(figsize=(10, 6))
mn = ['Llama 3.1\n8B', 'Llama 4\nScout', 'Mistral\n7B v0.3', 'Mistral\nSmall 3',
      'Gemma 2\n9B', 'Gemma 3\n12B', 'Qwen 2.5\n7B', 'Qwen 3\n8B']
eff = np.array([7619, 5079, 7619, 6095, 5714, 23273, 17534, 13617])
bars = ax.barh(mn, eff, color=['#111','#555','#333','#777','#444','#999','#222','#666'], edgecolor='#000')
ax.set_xlabel('Context Tokens per GB of KV-Cache')
ax.set_title('Memory Efficiency: Context Capacity per GB of KV-Cache\n(Higher = More Efficient)')
for bar, val in zip(bars, eff):
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2., f'{val:,}', ha='left', va='center', fontsize=9)
plt.tight_layout(); plt.savefig('charts/04-memory-efficiency-score.png'); plt.close()
print("Chart 4 saved")

# Chart 5: Q4 Quantization Impact
fig, ax = plt.subplots(figsize=(8, 5))
q4m = ['Gemma 3 12B', 'Llama 3.1 8B', 'DeepSeek-Coder\nV2-Lite 16B']
ppl = [-0.7, 2.8, 3.0]
bars = ax.bar(q4m, ppl, color=['#555555', '#333333', '#999999'], edgecolor='#000', width=0.5)
ax.axhline(y=0, color='#000', linewidth=0.8)
ax.set_ylabel('Perplexity Change from FP16 Baseline (%)')
ax.set_title('Impact of Q4 KV-Cache Quantization on Perplexity\n(Shkolnikov et al., 2026)')
for bar, val in zip(bars, ppl):
    yp = val + 0.15 if val >= 0 else val - 0.3
    ax.text(bar.get_x() + bar.get_width()/2., yp, f'{val:+.1f}%', ha='center',
            va='bottom' if val >= 0 else 'top', fontsize=11, fontweight='bold')
plt.tight_layout(); plt.savefig('charts/05-q4-perplexity-impact.png'); plt.close()
print("Chart 5 saved")

print("\nAll charts generated!")
