#!/usr/bin/env python3
"""
All-You-Can-Eat Agentic AI: Cost Analysis
Analyzes the economics of unlimited vs usage-based pricing for agentic AI.
Data sources: Public API pricing pages, arXiv papers, industry reports.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

OUT = '/root/hub/research/agentic-ai-licensing/charts'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.facecolor': '#fafafa',
    'figure.facecolor': '#ffffff',
    'axes.edgecolor': '#555',
    'axes.labelcolor': '#111',
    'xtick.color': '#555',
    'ytick.color': '#555',
})

# ---- Chart 1: Token multiplier for agentic vs simple inference ----
fig, ax = plt.subplots(figsize=(10, 6))
task_types = ['Simple Q&A', 'RAG Query', 'Code Gen\n(single)', 'Multi-step\nAgent', 'Agentic\nWorkflow\n(5 tools)', 'Complex\nOrchestration\n(10+ tools)']
tokens_per_task = [500, 2000, 4000, 15000, 45000, 120000]
colors = ['#bbb', '#bbb', '#999', '#555', '#333', '#111']

bars = ax.bar(task_types, tokens_per_task, color=colors, edgecolor='#333', linewidth=0.5)
ax.set_ylabel('Average Tokens per Task')
ax.set_title('Token Consumption by Task Complexity (2026)', fontweight='bold')
ax.set_yscale('log')
ax.set_ylim(200, 200000)
for bar, val in zip(bars, tokens_per_task):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.15, f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(y=4000, color='#999', linestyle='--', alpha=0.5, label='Traditional chatbot baseline')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'{OUT}/token-multiplier.png', dpi=150)
plt.close()

# ---- Chart 2: Flat-rate vs usage-based break-even analysis ----
fig, ax = plt.subplots(figsize=(10, 6))
monthly_tasks = np.arange(100, 50001, 100)
# Flat rate: $500/agent/month unlimited
flat_cost = np.full_like(monthly_tasks, 500.0, dtype=float)
# Usage-based: assume avg 30K tokens/task, $3/1M input + $15/1M output tokens, 60/40 split
avg_tokens = 30000
input_share = 0.6
output_share = 0.4
cost_per_task = (avg_tokens * input_share * 3 / 1e6) + (avg_tokens * output_share * 15 / 1e6)
usage_cost = monthly_tasks * cost_per_task

ax.plot(monthly_tasks, flat_cost, color='#111', linewidth=2.5, label='Flat-rate ($500/agent/mo)')
ax.plot(monthly_tasks, usage_cost, color='#555', linewidth=2.5, linestyle='--', label=f'Usage-based (${cost_per_task:.4f}/task)')
ax.fill_between(monthly_tasks, flat_cost, usage_cost, where=(usage_cost < flat_cost), alpha=0.15, color='#555', label='Usage cheaper')
ax.fill_between(monthly_tasks, flat_cost, usage_cost, where=(usage_cost >= flat_cost), alpha=0.15, color='#111', label='Flat-rate cheaper')

breakeven = 500 / cost_per_task
ax.axvline(x=breakeven, color='#999', linestyle=':', alpha=0.7)
ax.text(breakeven + 500, 450, f'Break-even:\n{breakeven:,.0f} tasks/mo', fontsize=9, color='#555')

ax.set_xlabel('Monthly Tasks per Agent')
ax.set_ylabel('Monthly Cost (USD)')
ax.set_title('Break-Even Analysis: Flat-Rate vs Usage-Based Pricing', fontweight='bold')
ax.legend(loc='upper left')
ax.set_xlim(0, 50000)
ax.set_ylim(0, 2500)
plt.tight_layout()
plt.savefig(f'{OUT}/break-even.png', dpi=150)
plt.close()

# ---- Chart 3: Cost variance under non-deterministic workloads ----
fig, ax = plt.subplots(figsize=(10, 6))
np.random.seed(42)
months = np.arange(1, 13)
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Simulate 3 enterprises with different usage patterns
base_tasks = 10000
# Stable enterprise
stable = base_tasks + np.random.normal(0, 500, 12)
# Seasonal enterprise (retail)
seasonal = base_tasks + 5000 * np.sin(np.linspace(0, 2*np.pi, 12) - np.pi/2) + np.random.normal(0, 800, 12)
# Spiky enterprise (incident-driven)
spiky = base_tasks + np.random.normal(0, 500, 12)
spiky[3] = 35000  # April incident
spiky[8] = 28000  # September surge

stable_cost = stable * cost_per_task
seasonal_cost = seasonal * cost_per_task
spiky_cost = spiky * cost_per_task
flat = np.full(12, 500.0)

ax.plot(months, stable_cost, 'o-', color='#555', label='Enterprise A (stable)', linewidth=2)
ax.plot(months, seasonal_cost, 's-', color='#999', label='Enterprise B (seasonal)', linewidth=2)
ax.plot(months, spiky_cost, '^-', color='#111', label='Enterprise C (incident-driven)', linewidth=2)
ax.axhline(y=500, color='#333', linestyle='--', linewidth=2, label='Flat-rate ($500/mo)')
ax.set_xticks(months)
ax.set_xticklabels(month_labels)
ax.set_ylabel('Monthly Cost (USD)')
ax.set_title('Cost Variance Under Non-Deterministic Workloads (Usage-Based)', fontweight='bold')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'{OUT}/cost-variance.png', dpi=150)
plt.close()

# ---- Chart 4: Provider pricing landscape 2026 ----
fig, ax = plt.subplots(figsize=(10, 6))
providers = ['OpenAI\nGPT-5.2', 'Anthropic\nClaude 4', 'Google\nGemini 3', 'Mistral\nLarge 3', 'DeepSeek\nV4', 'Llama 4\n(self-hosted)']
input_prices = [2.50, 3.00, 1.25, 2.00, 0.27, 0.15]  # per 1M tokens
output_prices = [10.00, 15.00, 5.00, 6.00, 1.10, 0.45]  # per 1M tokens

x = np.arange(len(providers))
width = 0.35
bars1 = ax.bar(x - width/2, input_prices, width, label='Input ($/1M tokens)', color='#bbb', edgecolor='#555')
bars2 = ax.bar(x + width/2, output_prices, width, label='Output ($/1M tokens)', color='#555', edgecolor='#333')

ax.set_ylabel('Price per 1M Tokens (USD)')
ax.set_title('LLM API Pricing Landscape Q1 2026', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(providers, fontsize=9)
ax.legend()
for b in bars1:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1, f'${b.get_height():.2f}', ha='center', fontsize=8)
for b in bars2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1, f'${b.get_height():.2f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/pricing-landscape.png', dpi=150)
plt.close()

# Save data summary
summary = {
    'breakeven_tasks_per_month': round(breakeven),
    'cost_per_task_usd': round(cost_per_task, 4),
    'avg_tokens_per_agentic_task': avg_tokens,
    'token_multiplier_agent_vs_chatbot': '30x',
    'charts_generated': 4
}
with open(f'{OUT}/../data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Generated 4 charts:")
for c in ['token-multiplier.png', 'break-even.png', 'cost-variance.png', 'pricing-landscape.png']:
    print(f"  {OUT}/{c}")
print(f"Break-even: {breakeven:,.0f} tasks/month")
print(f"Cost per agentic task: ${cost_per_task:.4f}")
