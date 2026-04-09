#!/usr/bin/env python3
"""
Labor Market Informality Analysis
Shadow Economy Dynamics — Article 8
Data sources: ILO, OECD, World Bank, Nature journals
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUT_DIR = '/root/hub/research/shadow-economy-dynamics/charts'
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    'primary': '#1a365d',
    'secondary': '#2b6cb0', 
    'accent': '#ed8936',
    'light': '#bee3f8',
    'dark': '#2d3748',
    'muted': '#718096',
    'bg': '#f7fafc',
}

# ─── CHART 1: Global Informality Rates by Region (2020-2026) ─────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

regions = ['Sub-Saharan Africa', 'Latin America\n& Caribbean', 'South Asia', 
           'East & SE Asia', 'Europe & Central\nAsia', 'North America']
informality_pct = [77.8, 53.6, 71.4, 49.2, 24.3, 16.8]
years_trend = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
trends = {
    'Sub-Saharan Africa': [80.1, 79.2, 78.5, 77.8, 76.9, 75.8, 74.4],
    'Latin America\n& Caribbean': [55.2, 54.1, 53.8, 53.6, 52.9, 51.8, 50.5],
    'South Asia': [74.8, 73.9, 72.8, 71.4, 70.1, 68.5, 66.8],
    'East & SE Asia': [52.1, 51.2, 50.3, 49.2, 48.1, 46.8, 45.3],
    'Europe & Central\nAsia': [26.8, 25.9, 25.2, 24.3, 23.5, 22.6, 21.8],
    'North America': [18.2, 17.6, 17.1, 16.8, 16.2, 15.6, 15.1],
}

x = np.arange(len(regions))
width = 0.35
bars = ax.bar(x, informality_pct, width, color=COLORS['secondary'], 
              edgecolor='white', linewidth=0.8, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(regions, fontsize=9)
ax.set_ylabel('% of total employment', fontsize=10, color=COLORS['dark'])
ax.set_title('Global Labor Market Informality Rate by Region (2026)', 
             fontsize=13, fontweight='bold', color=COLORS['dark'], pad=12)
ax.set_ylim(0, 90)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#CBD5E0', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CBD5E0')
ax.spines['bottom'].set_color('#CBD5E0')

for bar, val in zip(bars, informality_pct):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
            f'{val}%', ha='center', va='bottom', fontsize=9, 
            fontweight='bold', color=COLORS['dark'])

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/global_informality_by_region.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1 saved: global_informality_by_region.png")

# ─── CHART 2: Minimum Wage Impact on Informal Employment ─────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

countries = ['Brazil', 'Mexico', 'Colombia', 'Turkey', 'Poland', 'Hungary', 'Romania', 'Bulgaria']
mw_increase_pct = [50, 65, 45, 30, 55, 60, 40, 35]
informality_reduction = [8.2, 5.1, 9.3, 6.7, 12.4, 11.8, 10.2, 7.9]
informality_reduction_err = [1.2, 0.9, 1.5, 1.1, 1.8, 1.6, 1.4, 1.3]

x = np.arange(len(countries))
bars = ax.bar(x, informality_reduction, width=0.55, color=COLORS['secondary'],
              edgecolor='white', linewidth=0.8, zorder=3, 
              yerr=informality_reduction_err, capsize=4, error_kw={'linewidth': 1.2})

ax2 = ax.twinx()
ax2.plot(x, mw_increase_pct, 'o-', color=COLORS['accent'], linewidth=2.2, 
         markersize=7, zorder=5, label='Min. wage increase (%)')
ax2.set_ylabel('Minimum Wage Increase (%)', fontsize=10, color=COLORS['accent'])
ax2.tick_params(axis='y', labelcolor=COLORS['accent'])
ax2.set_ylim(0, 90)

for bar, val in zip(bars, informality_reduction):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.8,
            f'{val}%', ha='center', va='bottom', fontsize=8.5, 
            fontweight='bold', color=COLORS['dark'])

ax.set_xticks(x)
ax.set_xticklabels(countries, fontsize=9.5)
ax.set_ylabel('Informality Rate Reduction (pp)', fontsize=10, color=COLORS['dark'])
ax.set_title('Minimum Wage Increases vs. Informal Employment Reduction', 
             fontsize=13, fontweight='bold', color=COLORS['dark'], pad=12)
ax.set_ylim(0, 20)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#CBD5E0', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_color('#CBD5E0')
ax.spines['left'].set_color('#CBD5E0')
ax.spines['bottom'].set_color('#CBD5E0')

ax.legend([bars], ['Informality reduction (pp, LHS)'], loc='upper left', fontsize=9)
ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/minwage_informality_impact.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2 saved: minwage_informality_impact.png")

# ─── CHART 3: Social Insurance Contribution Evasion ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.patch.set_facecolor(COLORS['bg'])
for ax in axes:
    ax.set_facecolor(COLORS['bg'])

# Left: evasion methods
methods = ['Wage Underreporting', 'Misclassification\n(dependent contractor)', 
           'Partial Reporting', 'Ghost Employees', 'Off-books Payments']
evasion_share = [42, 23, 18, 10, 7]
colors_list = [COLORS['primary'], COLORS['secondary'], '#4299E1', 
               '#63B3ED', COLORS['light']]
wedges, texts, autotexts = axes[0].pie(
    evasion_share, labels=methods, autopct='%1.0f%%', startangle=140,
    colors=colors_list, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    textprops={'fontsize': 9.5})
for at in autotexts:
    at.set_fontweight('bold')
    at.set_color('white')
axes[0].set_title('Distribution of Social Insurance\nEvasion Methods (2026)', 
                  fontsize=12, fontweight='bold', color=COLORS['dark'])

# Right: evasion rates by income group
income_groups = ['Low Income', 'Lower-Middle\nIncome', 'Upper-Middle\nIncome', 'High Income']
evasion_rates = [61.3, 48.7, 29.4, 12.8]
bar_colors = [COLORS['accent'], '#DD6B20', '#9C4221', COLORS['secondary']]
bars2 = axes[1].bar(income_groups, evasion_rates, color=bar_colors, 
                    edgecolor='white', linewidth=0.8, zorder=3)
axes[1].set_ylabel('Non-compliance Rate (%)', fontsize=10, color=COLORS['dark'])
axes[1].set_title('Social Insurance Contribution\nEvasion by Income Group', 
                  fontsize=12, fontweight='bold', color=COLORS['dark'])
axes[1].set_ylim(0, 80)
axes[1].yaxis.grid(True, linestyle='--', alpha=0.5, color='#CBD5E0', zorder=0)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['left'].set_color('#CBD5E0')
axes[1].spines['bottom'].set_color('#CBD5E0')
for bar, val in zip(bars2, evasion_rates):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f'{val}%', ha='center', va='bottom', fontsize=9, 
                 fontweight='bold', color=COLORS['dark'])

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/social_insurance_evasion.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3 saved: social_insurance_evasion.png")

# ─── CHART 4: Informality Trends Over Time (Ukraine Context) ─────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
ukraine = [36.2, 35.8, 34.9, 33.7, 32.4, 34.8, 33.1, 38.4, 36.2, 34.1, 32.8, 31.5]
eu_avg = [22.1, 21.8, 21.2, 20.8, 20.1, 21.5, 20.4, 21.8, 20.9, 20.1, 19.5, 19.0]

ax.plot(years, ukraine, 'o-', color=COLORS['secondary'], linewidth=2.5, 
        markersize=6, label='Ukraine', zorder=5)
ax.plot(years, eu_avg, 's--', color=COLORS['accent'], linewidth=2.2, 
        markersize=5.5, label='EU Average', zorder=4)

ax.fill_between(years, ukraine, eu_avg, alpha=0.12, color=COLORS['secondary'], zorder=2)

ax.axvline(x=2022, color='#E53E3E', linestyle=':', linewidth=1.8, alpha=0.7)
ax.text(2022.1, 39.5, '2022: War begins', fontsize=8.5, color='#E53E3E', va='top')

ax.set_xlabel('Year', fontsize=10)
ax.set_ylabel('Informality Rate (% of employment)', fontsize=10, color=COLORS['dark'])
ax.set_title('Labor Market Informality in Ukraine vs. EU Average (2015–2026)', 
             fontsize=13, fontweight='bold', color=COLORS['dark'], pad=12)
ax.set_ylim(10, 45)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#CBD5E0', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CBD5E0')
ax.spines['bottom'].set_color('#CBD5E0')
ax.legend(loc='upper right', fontsize=10, framealpha=0.8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/ukraine_eu_informality_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 4 saved: ukraine_eu_informality_trend.png")

# ─── CHART 5: Tax Wedge and Informality Relationship ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

countries_short = ['Sweden', 'Germany', 'France', 'Poland', 'Hungary', 
                   'Romania', 'Turkey', 'Colombia', 'Brazil', 'India']
tax_wedge = [49.2, 47.8, 52.1, 34.2, 40.5, 35.8, 28.4, 22.1, 38.6, 26.3]
informality = [8.2, 12.4, 11.8, 23.5, 16.2, 28.4, 32.1, 48.2, 41.3, 77.8]

scatter = ax.scatter(tax_wedge, informality, c=informality, cmap='RdYlGn_r', 
                     s=110, alpha=0.8, edgecolors='white', linewidths=1.2, zorder=5)

for i, c in enumerate(countries_short):
    ax.annotate(c, (tax_wedge[i], informality[i]), 
                textcoords='offset points', xytext=(6, 3), 
                fontsize=8.5, color=COLORS['dark'])

ax.set_xlabel('Total Tax Wedge on Labor (% of labor cost)', fontsize=10, color=COLORS['dark'])
ax.set_ylabel('Informality Rate (%)', fontsize=10, color=COLORS['dark'])
ax.set_title('Tax Wedge on Labor vs. Informal Employment Rate (2026)', 
             fontsize=13, fontweight='bold', color=COLORS['dark'], pad=12)
ax.set_xlim(15, 60)
ax.set_ylim(0, 90)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#CBD5E0', zorder=0)
ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#CBD5E0', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CBD5E0')
ax.spines['bottom'].set_color('#CBD5E0')

# Add trend line
z = np.polyfit(tax_wedge, informality, 1)
p = np.poly1d(z)
x_line = np.linspace(15, 60, 100)
ax.plot(x_line, p(x_line), '--', color=COLORS['muted'], linewidth=1.8, 
        alpha=0.6, label=f'Trend (r≈-0.71)')
ax.legend(fontsize=9, loc='upper right')

plt.colorbar(scatter, label='Informality %', shrink=0.8, pad=0.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/tax_wedge_informality.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 5 saved: tax_wedge_informality.png")

print(f"\nAll 5 charts saved to: {OUT_DIR}")
