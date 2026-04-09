"""
Generate charts for Poland Shadow Economy Analysis
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Read data
with open('/root/hub/research/shadow-economy-poland/charts/data.json', 'r') as f:
    data = json.load(f)

# Set monochrome style
plt.style.use('grayscale')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11

# Chart 1: VAT Gap Reduction Timeline
fig1, ax1 = plt.subplots(figsize=(10, 6))
years = data['vat_gap']['years']
ax1.plot(years, data['vat_gap']['vat_gap_pct'], 'k-', marker='o', linewidth=2.5, markersize=8, label='VAT Gap %')
ax1.fill_between(years, data['vat_gap']['vat_gap_pct'], alpha=0.15, color='black')

# Annotate key events
ax1.annotate('JPK_VAT\n(2016)', xy=(2016, 25.0), xytext=(2015, 30), 
             fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='gray'))
ax1.annotate('Split Payment\n(2017)', xy=(2017, 23.6), xytext=(2017.5, 28),
             fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='gray'))
ax1.annotate('JPK Mandatory\n(2018)', xy=(2018, 19.1), xytext=(2019, 24),
             fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='gray'))

ax1.set_xlabel('Year')
ax1.set_ylabel('VAT Gap (% of Total Liability)')
ax1.set_title('Poland VAT Gap Reduction: Impact of Digital Tax Measures (2015-2023)')
ax1.set_xlim(2014.5, 2023.5)
ax1.set_ylim(5, 32)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right')
plt.tight_layout()
plt.savefig('/root/hub/research/shadow-economy-poland/charts/fig1_vat_gap_trend.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print("✓ fig1_vat_gap_trend.png")

# Chart 2: KAS Efficiency Metrics
fig2, ax2 = plt.subplots(figsize=(10, 6))
kas = data['kas']
ax2.bar([str(y) for y in kas['years'][:-1]], kas['audit_findings_pln_billion'][:-1], 
        color='black', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Year')
ax2.set_ylabel('Audit Findings (PLN Billion)')
ax2.set_title('Polish Tax Authority Efficiency: Audit Findings (2018-2024)')
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add 2025 projection as pattern
ax2.bar(str(kas['years'][-1]), kas['audit_findings_pln_billion'][-1], 
        color='white', edgecolor='black', hatch='///', label='2025 projected')
ax2.legend(loc='upper left')
plt.tight_layout()
plt.savefig('/root/hub/research/shadow-economy-poland/charts/fig2_kas_efficiency.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("✓ fig2_kas_efficiency.png")

# Chart 3: EU VAT Gap Comparison
fig3, ax3 = plt.subplots(figsize=(11, 6))
comp = data['comparison']
pos = np.arange(len(comp['countries']))
colors = ['#333333' if c == 'Poland' else '#999999' if c == 'EU_Average' else '#CCCCCC' 
          for c in comp['countries']]
bars = ax3.barh(pos, comp['vat_gap_pct'], color=colors, edgecolor='black')
ax3.set_yticks(pos)
ax3.set_yticklabels(comp['countries'])
ax3.set_xlabel('VAT Compliance Gap (% of Total Liability) - 2023')
ax3.set_title('EU VAT Gap Comparison: Poland vs Member States')
ax3.axvline(x=18.7, color='black', linestyle='--', alpha=0.5, label='EU Average (18.7%)')
ax3.legend()
plt.tight_layout()
plt.savefig('/root/hub/research/shadow-economy-poland/charts/fig3_eu_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig3)
print("✓ fig3_eu_comparison.png")

# Chart 4: Digital Transformation Impact Radar-style
fig4, ax4 = plt.subplots(figsize=(10, 6))
impact = data['digital_impact']
x = np.arange(len(impact['dimensions']))
width = 0.35
bars1 = ax4.bar(x - width/2, impact['pre_2016_score'], width, label='Pre-2016', 
                color='#666666', edgecolor='black')
bars2 = ax4.bar(x + width/2, impact['post_2023_score'], width, label='Post-2023',
                color='#000000', edgecolor='black')
ax4.set_ylabel('Effectiveness Score (0-100)')
ax4.set_title('Digital Tax System Impact: Transformation Across Key Dimensions')
ax4.set_xticks(x)
ax4.set_xticklabels(impact['dimensions'], rotation=15, ha='right')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('/root/hub/research/shadow-economy-poland/charts/fig4_digital_transformation.png', dpi=150, bbox_inches='tight')
plt.close(fig4)
print("✓ fig4_digital_transformation.png")

print("\nAll 4 charts generated successfully!")
