#!/usr/bin/env python3
"""Analysis for: Regional Disparities in Shadow Economy — Oblasts-Level Analysis 2015-2025"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import json

plt.rcParams.update({
    'figure.figsize': (12, 7),
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False
})

with open("regional_shadow_data.json") as f:
    data = json.load(f)

# ── Chart 1: Regional Shadow Economy Score — Horizontal Bar Chart ──
fig, ax = plt.subplots(figsize=(12, 9))
oblasts = list(data['regional_shadow_score'].keys())
scores = list(data['regional_shadow_score'].values())
# Sort descending
sorted_pairs = sorted(zip(scores, oblasts), reverse=True)
scores_s, oblasts_s = zip(*sorted_pairs)

colors = ['#333333' if s < 35 else '#777777' if s < 50 else '#aaaaaa' for s in scores_s]
bars = ax.barh(oblasts_s, scores_s, color=colors, edgecolor='#000', linewidth=0.5)
ax.axvline(x=np.mean(scores_s), color='black', linestyle='--', linewidth=1.5, label=f'National Avg ({np.mean(scores_s):.1f})')
ax.set_xlabel('Shadow Economy Composite Index (0–100)', fontsize=12)
ax.set_title('Figure 1. Shadow Economy Composite Index by Oblast, 2021\n(Higher = More Informal Activity)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
# Legend patches
p1 = mpatches.Patch(color='#333333', label='Low shadow (<35)')
p2 = mpatches.Patch(color='#777777', label='Moderate (35–50)')
p3 = mpatches.Patch(color='#aaaaaa', label='High (>50)')
ax.legend(handles=[p1, p2, p3] + [plt.Line2D([0], [0], color='black', linestyle='--', label=f'Mean: {np.mean(scores_s):.1f}')], fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('charts/01-regional-shadow-scores.png', dpi=150)
plt.close()
print("Chart 1 saved")

# ── Chart 2: Shadow Economy Trend 2015-2025 with war phase annotation ──
fig, ax = plt.subplots(figsize=(12, 6))
years = list(data['shadow_economy_trend'].keys())
pct_gdp = [data['shadow_economy_trend'][y]['national_pct_gdp'] for y in years]

ax.plot(years, pct_gdp, 'k-o', linewidth=2, markersize=7, label='Shadow Economy % GDP')
ax.fill_between(years, pct_gdp, alpha=0.1, color='black')

# Annotate war period
ax.axvspan(2022, 2025, alpha=0.07, color='black', label='Full-scale war period')
ax.axvline(x=2022, color='black', linestyle=':', linewidth=1.5)
ax.annotate('Full-scale\ninvasion', xy=(2022, 38.7), xytext=(2022.1, 42), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='black'), fontweight='bold')

# Annotate DiIA/digitalization
ax.annotate('Diia platform\nlaunched', xy=(2020, 32.3), xytext=(2018.3, 29),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='#555'))

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Shadow Economy (% of GDP)', fontsize=12)
ax.set_title('Figure 2. Ukraine Shadow Economy Dynamics 2015–2025\n(IMF/Ministry of Finance Estimates)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(25, 55)
ax.set_xticks(years)
plt.tight_layout()
plt.savefig('charts/02-shadow-economy-trend.png', dpi=150)
plt.close()
print("Chart 2 saved")

# ── Chart 3: Scatter — GRP per Capita vs Shadow Economy Score ──
fig, ax = plt.subplots(figsize=(11, 7))
shadow_scores = data['regional_shadow_score']
grp = data['grp_per_capita_2021']
common = set(shadow_scores.keys()) & set(grp.keys())
x_vals = [grp[o] for o in common]
y_vals = [shadow_scores[o] for o in common]
labels = list(common)

scatter = ax.scatter(x_vals, y_vals, c='black', s=60, alpha=0.8, zorder=5)

# Regression line
z = np.polyfit(x_vals, y_vals, 1)
p = np.poly1d(z)
x_line = np.linspace(min(x_vals), max(x_vals), 100)
ax.plot(x_line, p(x_line), 'k--', linewidth=1.5, alpha=0.7, label=f'Trend (r={np.corrcoef(x_vals,y_vals)[0,1]:.2f})')

# Label key outliers
for i, lbl in enumerate(labels):
    if lbl in ["Kyiv City", "Zakarpattia", "Luhansk", "Dnipropetrovsk", "Poltava"]:
        ax.annotate(lbl, (x_vals[i], y_vals[i]), textcoords="offset points", xytext=(6, 3), fontsize=8)

ax.set_xlabel('GRP per Capita (UAH thousand), 2021', fontsize=12)
ax.set_ylabel('Shadow Economy Composite Index', fontsize=12)
ax.set_title('Figure 3. GRP per Capita vs. Shadow Economy Index by Oblast\n(R² indicates inverse relationship)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('charts/03-grp-vs-shadow.png', dpi=150)
plt.close()
print("Chart 3 saved")

# ── Chart 4: Informal Employment vs Shadow Economy Score (Bubble) ──
fig, ax = plt.subplots(figsize=(11, 7))
inf_emp = data['informal_employment_2021']
common2 = set(shadow_scores.keys()) & set(inf_emp.keys()) & set(grp.keys())
x2 = [inf_emp[o] for o in common2]
y2 = [shadow_scores[o] for o in common2]
sizes = [grp[o] / 5 for o in common2]  # bubble size = GRP
labels2 = list(common2)

sc2 = ax.scatter(x2, y2, s=sizes, c='#555555', alpha=0.6, edgecolors='black', linewidths=0.7)
z2 = np.polyfit(x2, y2, 1)
p2 = np.poly1d(z2)
x_line2 = np.linspace(min(x2), max(x2), 100)
ax.plot(x_line2, p2(x_line2), 'k-', linewidth=1.5, label=f'Trend (r={np.corrcoef(x2,y2)[0,1]:.2f})')

for i, lbl in enumerate(labels2):
    if lbl in ["Kyiv City", "Zakarpattia", "Luhansk", "Kharkiv", "Lviv"]:
        ax.annotate(lbl, (x2[i], y2[i]), textcoords="offset points", xytext=(5, 3), fontsize=8)

ax.set_xlabel('Informal Employment Share (%), 2021', fontsize=12)
ax.set_ylabel('Shadow Economy Composite Index', fontsize=12)
ax.set_title('Figure 4. Informal Employment vs. Shadow Economy by Oblast\n(Bubble size proportional to GRP per Capita)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('charts/04-informal-employment-bubble.png', dpi=150)
plt.close()
print("Chart 4 saved")

print("All charts generated successfully.")
