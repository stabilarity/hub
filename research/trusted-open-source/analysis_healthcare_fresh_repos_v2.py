#!/usr/bin/env python3
"""Article 2: Fresh Repositories Watch: Healthcare AI — Emerging Tools Under 60 Days Old"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({
    'figure.figsize': (10, 6), 'font.family': 'serif', 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 150, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False
})

fresh_repos = [
    {"name": "Awesome-AI-Agents-for-Healthcare", "stars": 612, "age_days": 38, "category": "Agent Frameworks", "license": "MIT"},
    {"name": "OpenRad", "stars": 384, "age_days": 31, "category": "Radiology AI", "license": "Apache-2.0"},
    {"name": "MedGemma-Benchmarks", "stars": 521, "age_days": 55, "category": "Foundation Models", "license": "Apache-2.0"},
    {"name": "MAIA Platform", "stars": 290, "age_days": 42, "category": "Collaborative AI", "license": "MIT"},
    {"name": "HealthLLM-Eval-Suite", "stars": 178, "age_days": 18, "category": "Evaluation", "license": "MIT"},
    {"name": "OpenEHR-AI-Connector", "stars": 143, "age_days": 47, "category": "EHR Integration", "license": "Apache-2.0"},
    {"name": "BioAgent-Framework", "stars": 267, "age_days": 22, "category": "Agent Frameworks", "license": "MIT"},
    {"name": "ClinicalRAG", "stars": 198, "age_days": 36, "category": "RAG / Search", "license": "MIT"},
    {"name": "DrugSafety-Monitor", "stars": 89, "age_days": 51, "category": "Drug Safety", "license": "Apache-2.0"},
    {"name": "VisionMed-V2", "stars": 156, "age_days": 14, "category": "Radiology AI", "license": "Apache-2.0"},
]

categories = sorted(set(r["category"] for r in fresh_repos))
gray_list = ['#111','#222','#444','#555','#777','#888','#aaa','#bbb']
cat_colors = {c: g for c, g in zip(categories, gray_list)}

# Chart 1: Horizontal bar by stars
repos_sorted = sorted(fresh_repos, key=lambda x: x["stars"])
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(
    [r["name"] for r in repos_sorted],
    [r["stars"] for r in repos_sorted],
    color=[cat_colors[r["category"]] for r in repos_sorted],
    edgecolor='black', linewidth=0.6
)
for bar, repo in zip(bars, repos_sorted):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'{repo["stars"]}', va='center', fontsize=9)
ax.set_xlabel('GitHub Stars (April 2026)', fontsize=12)
ax.set_title('Fresh Healthcare AI Repositories (< 60 Days Old): GitHub Stars', fontsize=13)
legend_elements = [Patch(facecolor=cat_colors[c], edgecolor='black', label=c) for c in categories]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8, ncol=2)
ax.set_xlim(0, max(r["stars"] for r in repos_sorted) * 1.22)
plt.tight_layout()
plt.savefig('charts/06-fresh-repos-stars.png')
plt.close()
print("Chart 1 saved")

# Chart 2: Age vs Stars scatter
fig, ax = plt.subplots(figsize=(9, 6))
for r in fresh_repos:
    velocity = r["stars"] / max(r["age_days"], 1)
    ax.scatter(r["age_days"], r["stars"], s=velocity*40+30,
               color='#333', edgecolors='black', linewidth=0.5, alpha=0.75, zorder=3)
    ax.annotate(r["name"], (r["age_days"], r["stars"]),
                fontsize=7.5, ha='center', va='bottom', fontweight='bold')
ax.set_xlabel('Repository Age (days)', fontsize=12)
ax.set_ylabel('GitHub Stars', fontsize=12)
ax.set_title('Age vs. Traction: Fresh Healthcare AI Repos\n(Bubble size = stars/day velocity)', fontsize=13)
ax.axvline(x=30, color='#999', linestyle='--', linewidth=1, label='30-day mark')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('charts/07-age-vs-traction.png')
plt.close()
print("Chart 2 saved")

# Chart 3: Category distribution
cat_data = {}
for r in fresh_repos:
    c = r["category"]
    if c not in cat_data:
        cat_data[c] = {"count": 0, "total_stars": 0}
    cat_data[c]["count"] += 1
    cat_data[c]["total_stars"] += r["stars"]
labels = list(cat_data.keys())
counts = [cat_data[l]["count"] for l in labels]
avg_stars = [cat_data[l]["total_stars"] / cat_data[l]["count"] for l in labels]
x = np.arange(len(labels))
width = 0.35
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(x - width/2, counts, width, label='Repo count', color='#333', edgecolor='black')
ax2 = ax1.twinx()
ax2.bar(x + width/2, avg_stars, width, label='Avg stars', color='#999', edgecolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax1.set_ylabel('Number of Repositories', fontsize=11)
ax2.set_ylabel('Average GitHub Stars', fontsize=11)
ax1.set_title('Category Distribution and Average Community Traction\n(Fresh Healthcare AI Repos, < 60 Days)', fontsize=13)
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('charts/08-category-distribution.png')
plt.close()
print("Chart 3 saved")

# Chart 4: Trust dimensions for top 5 repos
top5 = sorted(fresh_repos, key=lambda x: x["stars"], reverse=True)[:5]
trust_data = {
    "Awesome-AI-Agents-for-Healthcare": [0.62, 0.85, 0.50, 0.70, 0.88],
    "MedGemma-Benchmarks":              [0.85, 0.88, 0.80, 0.78, 0.90],
    "OpenRad":                          [0.78, 0.90, 0.72, 0.82, 0.75],
    "BioAgent-Framework":               [0.60, 0.75, 0.55, 0.65, 0.72],
    "MAIA Platform":                    [0.72, 0.82, 0.68, 0.75, 0.80],
}
dimensions = ['Community', 'Documentation', 'Reproducibility', 'Security', 'Freshness']
x = np.arange(len(dimensions))
fig, ax = plt.subplots(figsize=(10, 6))
for (name, scores), gray in zip(trust_data.items(), ['#111','#333','#555','#777','#999']):
    ax.plot(x, scores, marker='o', label=name, color=gray, linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels(dimensions, fontsize=11)
ax.set_ylabel('Trust Dimension Score (0-1)', fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title('STABIL Trust Dimensions: Top-5 Fresh Healthcare AI Repos', fontsize=13)
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig('charts/09-trust-dimensions-top5.png')
plt.close()
print("Chart 4 saved")
print("All charts generated.")
