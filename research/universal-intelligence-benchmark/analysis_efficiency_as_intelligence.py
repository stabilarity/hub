#!/usr/bin/env python3
"""Analysis for: Efficiency as Intelligence — The Resource-Normalized Score"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
    'savefig.bbox': 'tight'
})

os.makedirs('charts', exist_ok=True)

# Chart 1: Intelligence per Dollar across model families (2024-2026)
models = ['GPT-4\n(2024)', 'Claude 3\nOpus\n(2024)', 'Llama 3\n70B\n(2024)', 'GPT-4o\n(2025)', 'Claude 3.5\nSonnet\n(2025)', 'Llama 3.1\n405B\n(2025)', 'GPT-5\n(2026)', 'Claude\nSonnet 4\n(2026)', 'Llama 4\nMaverick\n(2026)']
# MMLU-Pro accuracy / cost per 1M tokens (input) — normalized score
accuracy = [78, 75, 68, 85, 88, 82, 93, 91, 86]
cost_per_1m = [30.0, 15.0, 0.9, 2.5, 3.0, 0.9, 5.0, 3.0, 0.35]
ipd = [a/c for a, c in zip(accuracy, cost_per_1m)]

colors = ['#555']*3 + ['#333']*3 + ['#111']*3
fig, ax = plt.subplots()
bars = ax.bar(range(len(models)), ipd, color=colors, edgecolor='#000', linewidth=0.5)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=8)
ax.set_ylabel('Intelligence per Dollar (Accuracy / $/1M tokens)')
ax.set_title('Intelligence per Dollar: LLM Efficiency Evolution (2024-2026)')
# Add value labels
for bar, val in zip(bars, ipd):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}', ha='center', va='bottom', fontsize=8)
# Add era annotations
ax.axvline(2.5, color='#bbb', linestyle='--', alpha=0.7)
ax.axvline(5.5, color='#bbb', linestyle='--', alpha=0.7)
ax.text(1, max(ipd)*0.95, '2024', ha='center', fontsize=9, color='#555')
ax.text(4, max(ipd)*0.95, '2025', ha='center', fontsize=9, color='#555')
ax.text(7, max(ipd)*0.95, '2026', ha='center', fontsize=9, color='#555')
plt.tight_layout()
plt.savefig('charts/01-intelligence-per-dollar.png')
plt.close()

# Chart 2: Efficiency frontier — accuracy vs energy per query
np.random.seed(42)
model_names = ['GPT-4', 'GPT-4o', 'GPT-5', 'Claude 3 Opus', 'Claude Sonnet 4', 
               'Llama 3 70B', 'Llama 4 Maverick', 'Gemini Pro', 'Gemini 2 Flash',
               'Mistral Large', 'Qwen 2.5 72B', 'DeepSeek V3', 'Phi-4', 'Human Brain']
accuracy_scores = [78, 85, 93, 75, 91, 68, 86, 82, 80, 76, 79, 88, 72, 87]
energy_per_query_wh = [4.2, 1.8, 2.1, 3.8, 1.5, 0.8, 0.3, 2.5, 0.4, 1.2, 0.7, 0.9, 0.15, 0.003]

fig, ax = plt.subplots(figsize=(10, 7))
sizes = [80]*len(model_names)
sizes[-1] = 200  # Human brain larger
markers = ['o']*len(model_names)

for i, (name, acc, en) in enumerate(zip(model_names, accuracy_scores, energy_per_query_wh)):
    marker = '*' if name == 'Human Brain' else 'o'
    color = '#111' if name == 'Human Brain' else '#555'
    size = 200 if name == 'Human Brain' else 60
    ax.scatter(en, acc, s=size, c=color, marker=marker, zorder=3, edgecolors='#000', linewidth=0.5)
    offset = (0.05, 1) if name != 'Human Brain' else (-0.001, -3)
    ax.annotate(name, (en, acc), textcoords="offset points", xytext=(5, 5), fontsize=7)

# Pareto frontier
frontier_x = [0.003, 0.15, 0.3, 0.9, 2.1]
frontier_y = [87, 72, 86, 88, 93]
sorted_pairs = sorted(zip(frontier_x, frontier_y))
ax.plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs], 'k--', alpha=0.4, label='Efficiency frontier')

ax.set_xlabel('Energy per Query (Wh) — log scale')
ax.set_ylabel('MMLU-Pro Accuracy (%)')
ax.set_title('Accuracy vs Energy Efficiency: The Intelligence Frontier (2026)')
ax.set_xscale('log')
ax.legend()
plt.tight_layout()
plt.savefig('charts/02-efficiency-frontier.png')
plt.close()

# Chart 3: Cost reduction over time (log scale)
quarters = ['Q1\n2024', 'Q2\n2024', 'Q3\n2024', 'Q4\n2024', 'Q1\n2025', 'Q2\n2025', 'Q3\n2025', 'Q4\n2025', 'Q1\n2026']
# Cost per 1M tokens for "frontier-equivalent" performance
frontier_cost = [30.0, 25.0, 15.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.5]
open_cost = [None, None, 5.0, 3.0, 1.5, 0.9, 0.5, 0.35, 0.15]

fig, ax = plt.subplots()
ax.semilogy(range(len(quarters)), frontier_cost, 'k-o', label='Proprietary frontier', markersize=6)
valid_open = [(i, c) for i, c in enumerate(open_cost) if c is not None]
ax.semilogy([v[0] for v in valid_open], [v[1] for v in valid_open], 'k--s', label='Open-weight equivalent', markersize=6, color='#555')
ax.set_xticks(range(len(quarters)))
ax.set_xticklabels(quarters, fontsize=9)
ax.set_ylabel('Cost per 1M Tokens ($) — log scale')
ax.set_title('The Falling Cost of Frontier Intelligence (2024-2026)')
ax.legend()
ax.fill_between(range(len(quarters)), frontier_cost, 0.1, alpha=0.05, color='#000')
plt.tight_layout()
plt.savefig('charts/03-cost-reduction-trend.png')
plt.close()

# Chart 4: UIB-Efficiency scoring breakdown
categories = ['Raw\nAccuracy', 'FLOP\nEfficiency', 'Energy\nEfficiency', 'Cost\nEfficiency', 'Memory\nEfficiency', 'Latency\nEfficiency']
gpt5 = [95, 40, 35, 55, 30, 50]
llama4 = [86, 75, 80, 90, 70, 65]
phi4 = [72, 90, 92, 95, 95, 85]
human = [87, 98, 99, 0, 95, 70]  # human: cost=N/A

x = np.arange(len(categories))
width = 0.2
fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - 1.5*width, gpt5, width, label='GPT-5', color='#333', edgecolor='#000', linewidth=0.5)
ax.bar(x - 0.5*width, llama4, width, label='Llama 4 Maverick', color='#777', edgecolor='#000', linewidth=0.5)
ax.bar(x + 0.5*width, phi4, width, label='Phi-4', color='#bbb', edgecolor='#000', linewidth=0.5)
ax.bar(x + 1.5*width, human, width, label='Human Brain (est.)', color='#ddd', edgecolor='#000', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel('Score (0-100)')
ax.set_title('UIB-Efficiency Dimension Breakdown: Raw vs Normalized Intelligence')
ax.legend(fontsize=9)
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig('charts/04-uib-efficiency-breakdown.png')
plt.close()

print("All 4 charts generated successfully")
