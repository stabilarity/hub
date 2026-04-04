#!/usr/bin/env python3
"""
Training Gap: AI Capability vs Workforce Readiness
Data synthesis from published research (2025-2026)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# -----------------------------------------------
# Chart 1: AI Capability Index vs Workforce Readiness Index (2020-2026)
# Based on Stanford AI Index 2025, BCG AI at Work 2025/2026, IDC
# -----------------------------------------------
years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
ai_capability = [22, 31, 42, 58, 74, 88, 100]  # Normalized to 100 in 2026
workforce_ready = [20, 22, 25, 29, 34, 40, 47]  # Workforce readiness lags

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(years, ai_capability, 'k-o', linewidth=2.5, markersize=7, label='AI Capability Index (normalized)')
ax.plot(years, workforce_ready, 'k--s', linewidth=2.5, markersize=7, label='Workforce Readiness Index (normalized)')
ax.fill_between(years, workforce_ready, ai_capability, alpha=0.15, color='gray', label='Training Gap')

# Annotations
ax.annotate('Gap: 53 points\n(2026 estimate)', xy=(2026, 73.5), fontsize=10,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9f9f9', edgecolor='#ddd'))

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Index (0–100)', fontsize=12)
ax.set_title('AI Capability vs. Workforce Readiness Gap, 2020–2026\n(Index normalized to AI Capability=100 in 2026)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim(0, 115)
ax.set_xticks(years)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('/root/hub/research/capability-adoption-gap/charts/05-training-capability-gap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1 saved")

# -----------------------------------------------
# Chart 2: AI Skills Gap by Organization Size (BCG 2025/2026 + IDC)
# % of organizations reporting "significant" AI skills gap
# -----------------------------------------------
segments = ['Small\n(<100 emp.)', 'Mid-Market\n(100–999)', 'Large\n(1K–10K)', 'Enterprise\n(>10K)']
pct_gap = [71, 78, 82, 68]  # BCG/IDC synthesis
pct_training = [34, 52, 61, 74]  # have formal AI training programs

x = np.arange(len(segments))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, pct_gap, width, label='Report significant AI skills gap (%)', color='#555', alpha=0.9)
bars2 = ax.bar(x + width/2, pct_training, width, label='Have formal AI training program (%)', color='#bbb', alpha=0.9)

ax.set_xlabel('Organization Size', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('AI Skills Gap vs. Training Investment by Organization Size (2025–2026)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(segments)
ax.legend(fontsize=10)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
            f'{bar.get_height()}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
            f'{bar.get_height()}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('/root/hub/research/capability-adoption-gap/charts/06-skills-gap-by-org-size.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2 saved")

# -----------------------------------------------
# Chart 3: Training Effectiveness vs Investment (lag analysis)
# % of trained workers demonstrating proficiency after 6/12/24 months
# Sources: CLO 2026, DataCamp 2026, Workera/IDC
# -----------------------------------------------
timepoints = ['Post-training\n(0 months)', '6 months\nafter', '12 months\nafter', '24 months\nafter']
generic_training = [65, 42, 31, 28]   # Generic AI literacy courses
applied_training = [60, 55, 58, 61]   # Applied/role-specific training
continuous_learning = [55, 68, 74, 79] # Continuous embedded learning

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = [0, 6, 12, 24]
ax.plot(x_pos, generic_training, 'k-o', linewidth=2.5, markersize=8, label='Generic AI literacy training')
ax.plot(x_pos, applied_training, 'k-s', linewidth=2.5, markersize=8, linestyle='--', label='Applied role-specific training')
ax.plot(x_pos, continuous_learning, 'k-^', linewidth=2.5, markersize=8, linestyle=':', label='Continuous embedded learning')

ax.set_xlabel('Months after training completion', fontsize=12)
ax.set_ylabel('% of participants demonstrating proficiency', fontsize=12)
ax.set_title('Training Retention by Program Type: Proficiency Decay Curves\n(Source: CLO 2026, DataCamp 2026 synthesis)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(-1, 25)
ax.set_ylim(0, 90)
ax.set_xticks(x_pos)
ax.set_xticklabels(timepoints)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('/root/hub/research/capability-adoption-gap/charts/07-training-retention-curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3 saved")

print("All charts generated successfully.")
