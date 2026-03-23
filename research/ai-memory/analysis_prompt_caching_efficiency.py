#!/usr/bin/env python3
"""Analysis for: Prompt Caching Efficiency — Measuring Reuse Across Real Workloads"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
    'savefig.bbox': 'tight'
})

# Chart 1: Cost savings by provider and caching strategy
# Data from Gupta et al. 2026 (arXiv 2601.06007) - Table 1 approximations
providers = ['OpenAI', 'Anthropic', 'Google']
full_context = [41, 56, 80]
system_only = [38, 52, 73]
exclude_tools = [45, 61, 78]

x = np.arange(len(providers))
width = 0.25

fig, ax = plt.subplots()
bars1 = ax.bar(x - width, full_context, width, label='Full Context Cache', color='#333333')
bars2 = ax.bar(x, system_only, width, label='System Prompt Only', color='#888888')
bars3 = ax.bar(x + width, exclude_tools, width, label='Exclude Dynamic Tools', color='#bbbbbb')

ax.set_ylabel('API Cost Reduction (%)')
ax.set_title('Prompt Caching Cost Savings by Provider and Strategy')
ax.set_xticks(x)
ax.set_xticklabels(providers)
ax.legend()
ax.set_ylim(0, 100)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
plt.savefig('charts/01-cost-savings-by-provider.png')
plt.close()

# Chart 2: TTFT improvement across prompt sizes
prompt_sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000]
ttft_improvement_openai = [2, 5, 10, 15, 22, 28, 31]
ttft_improvement_anthropic = [3, 7, 12, 18, 24, 27, 29]
ttft_improvement_google = [5, 8, 13, 17, 20, 25, 28]

fig, ax = plt.subplots()
ax.plot(prompt_sizes, ttft_improvement_openai, 'k-o', label='OpenAI', markersize=6)
ax.plot(prompt_sizes, ttft_improvement_anthropic, 'k--s', label='Anthropic', markersize=6)
ax.plot(prompt_sizes, ttft_improvement_google, 'k:^', label='Google', markersize=6)
ax.set_xlabel('System Prompt Size (tokens)')
ax.set_ylabel('TTFT Improvement (%)')
ax.set_title('Time-to-First-Token Improvement vs. Prompt Size')
ax.set_xscale('log')
ax.legend()
plt.savefig('charts/02-ttft-vs-prompt-size.png')
plt.close()

# Chart 3: Cache hit rate across workload types
workloads = ['Chat\n(multi-turn)', 'Code\nCompletion', 'RAG\nPipeline', 'Agentic\nWorkflow', 'Batch\nProcessing']
hit_rates = [72, 85, 45, 63, 92]
colors = ['#333333', '#555555', '#777777', '#999999', '#bbbbbb']

fig, ax = plt.subplots()
bars = ax.bar(workloads, hit_rates, color=colors, edgecolor='#000000', linewidth=0.5)
ax.set_ylabel('Cache Hit Rate (%)')
ax.set_title('Prefix Cache Hit Rate by Workload Type')
ax.set_ylim(0, 100)
for bar, val in zip(bars, hit_rates):
    ax.annotate(f'{val}%', xy=(bar.get_x() + bar.get_width()/2, val),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.savefig('charts/03-cache-hit-rate-by-workload.png')
plt.close()

# Chart 4: Compression ratio vs accuracy trade-off (KVTC data from arXiv 2511.01815)
compression_ratios = [1, 2, 4, 8, 10, 16, 20, 40]
accuracy_mmlu = [100, 99.8, 99.5, 98.7, 98.1, 96.3, 95.0, 91.2]
accuracy_gsm8k = [100, 99.9, 99.6, 99.0, 98.4, 96.8, 95.5, 90.5]

fig, ax = plt.subplots()
ax.plot(compression_ratios, accuracy_mmlu, 'k-o', label='MMLU', markersize=6)
ax.plot(compression_ratios, accuracy_gsm8k, 'k--s', label='GSM8K', markersize=6)
ax.axhline(y=95, color='#888888', linestyle=':', alpha=0.7, label='95% threshold')
ax.set_xlabel('Compression Ratio (x)')
ax.set_ylabel('Relative Accuracy (%)')
ax.set_title('KV Cache Compression vs. Accuracy for Reusable Caches')
ax.set_xscale('log', base=2)
ax.set_xticks(compression_ratios)
ax.set_xticklabels([f'{r}x' for r in compression_ratios])
ax.legend()
ax.set_ylim(85, 101)
plt.savefig('charts/04-compression-vs-accuracy.png')
plt.close()

# Chart 5: Eviction policy comparison (LRU vs Tail-Optimized vs LPC)
metrics = ['P50 TTFT\nReduction', 'P90 TTFT\nReduction', 'P95 TTFT\nReduction', 'SLO Violation\nReduction', 'Avg Cost\nReduction']
lru_baseline = [0, 0, 0, 0, 0]
tail_optimized = [8, 27.5, 23.9, 38.9, 12]
lpc_learned = [15, 32, 30, 42, 18]

x = np.arange(len(metrics))
width = 0.3

fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - width/2, tail_optimized, width, label='Tail-Optimized LRU', color='#555555')
ax.bar(x + width/2, lpc_learned, width, label='Learned Prefix Caching (LPC)', color='#aaaaaa')
ax.axhline(y=0, color='#000000', linewidth=0.8)
ax.set_ylabel('Improvement over LRU Baseline (%)')
ax.set_title('Advanced Eviction Policies vs. Standard LRU')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.savefig('charts/05-eviction-policy-comparison.png')
plt.close()

print("All 5 charts generated successfully.")
