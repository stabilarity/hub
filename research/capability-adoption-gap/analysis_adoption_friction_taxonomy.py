#!/usr/bin/env python3
"""Analysis for: Adoption Friction Taxonomy — Categorizing the Barriers Between AI Capability and Enterprise Deployment"""
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

# Chart 1: Barrier categories by prevalence (from survey data synthesis)
categories = ['Data Quality\n& Availability', 'Talent &\nSkills Gap', 'Regulatory\nCompliance', 'Technical\nDebt', 'Cultural\nResistance', 'ROI\nUncertainty', 'Governance\nGaps', 'Integration\nComplexity']
prevalence = [52, 47, 41, 38, 35, 33, 29, 27]  # % of organizations citing barrier
colors = ['#111111', '#333333', '#444444', '#555555', '#666666', '#777777', '#888888', '#999999']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(categories, prevalence, color=colors, edgecolor='#000', linewidth=0.5)
ax.set_xlabel('Organizations Citing Barrier (%)')
ax.set_title('AI Adoption Friction: Barrier Prevalence Across Enterprises (2025–2026)')
ax.set_xlim(0, 65)
for bar, val in zip(bars, prevalence):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val}%', va='center', fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('charts/01-barrier-prevalence.png')
plt.close()

# Chart 2: Friction by enterprise size (SME vs Large vs Very Large)
barrier_labels = ['Data Quality', 'Talent Gap', 'Regulatory', 'Tech Debt', 'Cultural', 'ROI Unclear', 'Governance', 'Integration']
sme = [61, 58, 28, 22, 42, 51, 18, 35]
large = [48, 44, 45, 42, 33, 28, 34, 31]
very_large = [38, 35, 52, 51, 28, 19, 41, 24]

x = np.arange(len(barrier_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, sme, width, label='SMEs (<250)', color='#111', edgecolor='#000')
ax.bar(x, large, width, label='Large (250-5000)', color='#777', edgecolor='#000')
ax.bar(x + width, very_large, width, label='Very Large (5000+)', color='#bbb', edgecolor='#000')
ax.set_ylabel('Organizations Reporting Barrier (%)')
ax.set_title('Adoption Friction Profile by Enterprise Size')
ax.set_xticks(x)
ax.set_xticklabels(barrier_labels, rotation=30, ha='right')
ax.legend()
ax.set_ylim(0, 70)
plt.tight_layout()
plt.savefig('charts/02-friction-by-size.png')
plt.close()

# Chart 3: Friction taxonomy quadrant — Impact vs Addressability
fig, ax = plt.subplots(figsize=(10, 8))
barriers = {
    'Data Quality': (0.85, 0.72, 180),
    'Talent Gap': (0.78, 0.55, 160),
    'Regulatory': (0.70, 0.35, 140),
    'Tech Debt': (0.65, 0.45, 130),
    'Cultural Resistance': (0.60, 0.40, 120),
    'ROI Uncertainty': (0.55, 0.65, 110),
    'Governance': (0.50, 0.50, 100),
    'Integration': (0.45, 0.60, 90),
}
for label, (impact, addr, size) in barriers.items():
    ax.scatter(addr, impact, s=size*3, color='#333', alpha=0.7, edgecolors='#000', linewidth=1)
    ax.annotate(label, (addr, impact), textcoords="offset points", xytext=(10, 5), fontsize=9)

ax.axhline(y=0.625, color='#999', linestyle='--', alpha=0.5)
ax.axvline(x=0.5, color='#999', linestyle='--', alpha=0.5)
ax.set_xlabel('Addressability (higher = easier to mitigate)')
ax.set_ylabel('Deployment Impact (higher = more blocking)')
ax.set_title('Adoption Friction Taxonomy: Impact vs Addressability Matrix')
ax.text(0.75, 0.88, 'HIGH IMPACT\nHIGH ADDRESSABILITY\n(Quick Wins)', ha='center', fontsize=8, style='italic', color='#555')
ax.text(0.25, 0.88, 'HIGH IMPACT\nLOW ADDRESSABILITY\n(Structural)', ha='center', fontsize=8, style='italic', color='#555')
ax.text(0.75, 0.42, 'LOW IMPACT\nHIGH ADDRESSABILITY\n(Routine)', ha='center', fontsize=8, style='italic', color='#555')
ax.text(0.25, 0.42, 'LOW IMPACT\nLOW ADDRESSABILITY\n(Deprioritize)', ha='center', fontsize=8, style='italic', color='#555')
ax.set_xlim(0.15, 0.95)
ax.set_ylim(0.35, 0.95)
plt.tight_layout()
plt.savefig('charts/03-impact-addressability-matrix.png')
plt.close()

# Chart 4: Time to resolve each friction type (months)
barriers_time = ['Integration\nComplexity', 'ROI\nUncertainty', 'Governance\nGaps', 'Data\nQuality', 'Tech\nDebt', 'Talent\nGap', 'Cultural\nResistance', 'Regulatory\nCompliance']
time_min = [3, 4, 6, 6, 9, 9, 12, 18]
time_max = [9, 10, 14, 18, 24, 24, 36, 48]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(barriers_time))
ax.barh(y_pos, [t2-t1 for t1, t2 in zip(time_min, time_max)], left=time_min, 
        color='#888', edgecolor='#000', linewidth=0.5, alpha=0.7)
ax.barh(y_pos, time_min, color='#333', edgecolor='#000', linewidth=0.5)
for i, (t1, t2) in enumerate(zip(time_min, time_max)):
    ax.text(t2 + 0.5, i, f'{t1}–{t2} mo', va='center', fontsize=9)
ax.set_yticks(y_pos)
ax.set_yticklabels(barriers_time)
ax.set_xlabel('Time to Resolve (months)')
ax.set_title('Estimated Resolution Timeline by Friction Category')
ax.set_xlim(0, 55)
plt.tight_layout()
plt.savefig('charts/04-resolution-timeline.png')
plt.close()

print("All 4 charts generated successfully.")
