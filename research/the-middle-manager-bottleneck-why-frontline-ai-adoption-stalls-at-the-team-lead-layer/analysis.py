"""
The Middle-Manager Bottleneck: Why Frontline AI Adoption Stalls at the Team-Lead Layer
Reproduce: python3 analysis.py

Synthetic data analysis of AI tool adoption funnel showing drop-off between
individual contributors (ICs) and their direct managers (Team Leads).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Reproducible synthetic data
rng = np.random.default_rng(42)

# Simulate survey of 500 teams across org sizes
n_teams = 500

# Org size distribution
org_sizes = rng.choice(['Small (1-50)', 'Medium (51-500)', 'Large (501-5000)', 'Enterprise (5000+)'],
                       size=n_teams, p=[0.3, 0.35, 0.25, 0.1])

# IC adoption rates (higher)
ic_adoption_base = rng.beta(5, 2, n_teams) * 100  # ~70% mean

# Team lead adoption rates (lower - the bottleneck)
# Team leads adopt at ~60% of IC rate due to time constraints, approval burden
tl_adoption_rate = ic_adoption_base * rng.beta(3, 2, n_teams) * 0.6 + rng.normal(5, 3, n_teams)
tl_adoption_rate = np.clip(tl_adoption_rate, 0, 100)

# Compute drop-off
dropoff = ic_adoption_base - tl_adoption_rate

# Metrics by org size
metrics = {}
for size in ['Small (1-50)', 'Medium (51-500)', 'Large (501-5000)', 'Enterprise (5000+)']:
    mask = np.array([s == size for s in org_sizes])
    metrics[size] = {
        'ic_adoption_mean': float(np.mean(ic_adoption_base[mask])),
        'tl_adoption_mean': float(np.mean(tl_adoption_rate[mask])),
        'dropoff_mean': float(np.mean(dropoff[mask])),
        'n_teams': int(np.sum(mask))
    }

# Chart 1: Adoption Funnel by Org Size
fig, ax = plt.subplots(figsize=(12, 8))
sizes = list(metrics.keys())
x = np.arange(len(sizes))
width = 0.35

ic_means = [metrics[s]['ic_adoption_mean'] for s in sizes]
tl_means = [metrics[s]['tl_adoption_mean'] for s in sizes]

bars1 = ax.bar(x - width/2, ic_means, width, label='Individual Contributors', color='#2E86AB', alpha=0.85)
bars2 = ax.bar(x + width/2, tl_means, width, label='Team Leads', color='#A23B72', alpha=0.85)

ax.set_ylabel('AI Tool Adoption Rate (%)', fontsize=14)
ax.set_title('AI Adoption Funnel: ICs vs Team Leads by Organization Size', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sizes, fontsize=12)
ax.legend(fontsize=13)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    ax.annotate(f'{bar.get_height():.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=11)
for bar in bars2:
    ax.annotate(f'{bar.get_height():.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('./charts/chart1.png', dpi=100)
plt.close()

# Chart 2: Drop-off Distribution
fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(dropoff, bins=30, color='#F18F01', edgecolor='white', alpha=0.85, density=True)
ax.axvline(np.mean(dropoff), color='#C73E1D', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dropoff):.1f}%')
ax.axvline(np.median(dropoff), color='#2E86AB', linestyle='--', linewidth=2, label=f'Median: {np.median(dropoff):.1f}%')
ax.set_xlabel('Adoption Drop-off (IC% - TL%)', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.set_title('Distribution of Adoption Drop-off Between ICs and Team Leads', fontsize=16, fontweight='bold')
ax.legend(fontsize=13)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('./charts/chart2.png', dpi=100)
plt.close()

# Chart 3: Adoption Gap vs Org Size (boxplot)
fig, ax = plt.subplots(figsize=(12, 8))
data_by_size = [dropoff[np.array([s == sz for s in org_sizes])] for sz in sizes]
bp = ax.boxplot(data_by_size, labels=sizes, patch_artist=True, widths=0.6)
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Adoption Drop-off (Percentage Points)', fontsize=14)
ax.set_title('Adoption Gap Distribution by Organization Size', fontsize=16, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('./charts/chart3.png', dpi=100)
plt.close()

# Chart 4: Correlation heatmap of factors (synthetic)
# Simulate factors: approval_burden, time_pressure, team_size, tech_comfort
factors = rng.normal(0, 1, (n_teams, 4))
factors[:, 0] = dropoff / 100 + rng.normal(0, 0.3, n_teams)  # approval burden correlates with dropoff
factors[:, 1] = dropoff / 120 + rng.normal(0, 0.4, n_teams)  # time pressure
factors[:, 2] = rng.normal(0, 1, n_teams)  # team size (weak)
factors[:, 3] = -dropoff / 80 + rng.normal(0, 0.5, n_teams)  # tech comfort (negative corr)

corr_matrix = np.corrcoef(factors.T)
factor_names = ['Approval Burden', 'Time Pressure', 'Team Size', 'Tech Comfort']

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(factor_names, fontsize=12, rotation=45, ha='right')
ax.set_yticklabels(factor_names, fontsize=12)
ax.set_title('Correlation Matrix: Factors Driving the Middle-Manager Bottleneck', fontsize=16, fontweight='bold')

# Add correlation values
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', fontsize=14,
                color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

plt.colorbar(im, ax=ax, label='Correlation Coefficient')
plt.tight_layout()
plt.savefig('./charts/chart4.png', dpi=100)
plt.close()

# Chart 5: Time-series simulation (adoption over quarters)
quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024', 'Q1 2025', 'Q2 2025']
ic_trend = [45, 52, 58, 63, 67, 70]
tl_trend = [28, 32, 35, 38, 41, 43]

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(quarters, ic_trend, 'o-', label='Individual Contributors', color='#2E86AB', linewidth=3, markersize=10)
ax.plot(quarters, tl_trend, 's-', label='Team Leads', color='#A23B72', linewidth=3, markersize=10)
ax.fill_between(range(len(quarters)), ic_trend, tl_trend, alpha=0.15, color='gray', label='Bottleneck Gap')

ax.set_ylabel('Adoption Rate (%)', fontsize=14)
ax.set_title('AI Adoption Trajectory: Widening Gap Over Time', fontsize=16, fontweight='bold')
ax.legend(fontsize=13)
ax.set_ylim(0, 85)
ax.grid(axis='y', alpha=0.3)

# Annotate gap
for i, (ic, tl) in enumerate(zip(ic_trend, tl_trend)):
    ax.annotate(f'{ic-tl}pp', xy=(i, (ic+tl)/2), ha='center', va='center',
                fontsize=11, fontweight='bold', color='#C73E1D',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#C73E1D', alpha=0.9))

plt.tight_layout()
plt.savefig('./charts/chart5.png', dpi=100)
plt.close()

# Persist results
results = {
    'summary': {
        'total_teams_surveyed': n_teams,
        'overall_ic_adoption': float(np.mean(ic_adoption_base)),
        'overall_tl_adoption': float(np.mean(tl_adoption_rate)),
        'overall_dropoff': float(np.mean(dropoff)),
        'dropoff_std': float(np.std(dropoff))
    },
    'by_org_size': metrics,
    'correlation_matrix': corr_matrix.tolist(),
    'factor_names': factor_names,
    'time_series': {
        'quarters': quarters,
        'ic_adoption': ic_trend,
        'tl_adoption': tl_trend
    }
}

with open('./results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Analysis complete. Generated 5 charts and results.json")
print(f"Overall IC adoption: {results['summary']['overall_ic_adoption']:.1f}%")
print(f"Overall TL adoption: {results['summary']['overall_tl_adoption']:.1f}%")
print(f"Mean drop-off: {results['summary']['overall_dropoff']:.1f}pp")
