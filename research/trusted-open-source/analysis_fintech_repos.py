#!/usr/bin/env python3
"""Analysis for: Fresh Repositories Watch: Financial Technology"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
from collections import Counter

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
    'savefig.bbox': 'tight'
})

with open('/root/hub/research/trusted-open-source/fintech_repos.json') as f:
    repos = json.load(f)

# Filter: only repos with stars > 5
repos = [r for r in repos if r['stars'] > 5]

# --- Chart 1: Language Distribution ---
fig, ax = plt.subplots()
langs = Counter(r.get('language') or 'Unknown' for r in repos)
top_langs = langs.most_common(8)
labels, counts = zip(*top_langs)
colors = ['#111', '#555', '#888', '#aaa', '#bbb', '#ccc', '#ddd', '#eee']
bars = ax.barh(labels[::-1], counts[::-1], color=colors[:len(labels)][::-1])
ax.set_xlabel('Number of Repositories')
ax.set_title('Programming Language Distribution in New Fintech Repositories (2026)')
for bar, count in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, str(count), va='center', fontsize=10)
plt.tight_layout()
plt.savefig('/root/hub/research/trusted-open-source/charts/04-fintech-language-dist.png')
plt.close()

# --- Chart 2: License Distribution ---
fig, ax = plt.subplots()
licenses = Counter(r.get('license') or 'None' for r in repos)
top_lic = licenses.most_common(6)
labels, counts = zip(*top_lic)
wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct='%1.0f%%', colors=['#111','#555','#888','#aaa','#ccc','#eee'][:len(labels)])
for t in autotexts:
    t.set_fontsize(10)
ax.set_title('License Distribution in New Fintech Repositories (2026)')
plt.tight_layout()
plt.savefig('/root/hub/research/trusted-open-source/charts/04-fintech-license-dist.png')
plt.close()

# --- Chart 3: Stars vs Age (scatter) ---
fig, ax = plt.subplots()
from datetime import datetime
today = datetime.now()
ages = []
stars = []
categories = []
for r in repos:
    created = datetime.strptime(r['created'], '%Y-%m-%d')
    age = (today - created).days
    ages.append(age)
    stars.append(r['stars'])
    desc = (r.get('description') or '').lower()
    if any(k in desc for k in ['polymarket', 'prediction market', 'arbitrage']):
        categories.append('Prediction Markets')
    elif any(k in desc for k in ['trading', 'bot', 'binance', 'crypto']):
        categories.append('Trading Bots')
    elif any(k in desc for k in ['risk', 'portfolio', 'quant']):
        categories.append('Risk/Portfolio')
    elif any(k in desc for k in ['payment', 'fintech', 'banking']):
        categories.append('Payments/Banking')
    else:
        categories.append('Other Financial')

cat_colors = {'Prediction Markets': '#111', 'Trading Bots': '#555', 'Risk/Portfolio': '#888', 'Payments/Banking': '#aaa', 'Other Financial': '#ccc'}
for cat in cat_colors:
    mask = [c == cat for c in categories]
    ax.scatter([a for a, m in zip(ages, mask) if m], [s for s, m in zip(stars, mask) if m], 
               c=cat_colors[cat], label=cat, alpha=0.7, s=40)
ax.set_xlabel('Repository Age (days)')
ax.set_ylabel('GitHub Stars')
ax.set_title('Star Accumulation by Repository Category and Age')
ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig('/root/hub/research/trusted-open-source/charts/04-fintech-stars-age.png')
plt.close()

# --- Chart 4: Category Distribution Bar ---
fig, ax = plt.subplots()
cat_counts = Counter(categories)
cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
labels, counts = zip(*cats)
ax.bar(labels, counts, color=['#111','#555','#888','#aaa','#ccc'][:len(labels)])
ax.set_ylabel('Number of Repositories')
ax.set_title('Repository Category Distribution: Financial Technology (Jan-Mar 2026)')
plt.xticks(rotation=15, ha='right')
for i, c in enumerate(counts):
    ax.text(i, c + 0.3, str(c), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('/root/hub/research/trusted-open-source/charts/04-fintech-category-dist.png')
plt.close()

print("Charts generated successfully")
print(f"Total repos analyzed: {len(repos)}")
print(f"Categories: {dict(cat_counts)}")
print(f"Languages: {dict(langs.most_common(5))}")
print(f"Licenses: {dict(licenses.most_common(5))}")
