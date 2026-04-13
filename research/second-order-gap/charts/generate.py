import matplotlib.pyplot as plt
import numpy as np
import json

# Load data
with open('/root/hub/research/second-order-gap/data/chart_data.json') as f:
    data = json.load(f)

# Chart 1: Grouped bar chart - Skills gap emergence
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(data['chart1']['categories']))
width = 0.25
for i, series in enumerate(data['chart1']['series']):
    ax.bar(x + i*width, series['data'], width, label=series['name'])
ax.set_xticks(x + width)
ax.set_xticklabels(data['chart1']['categories'], rotation=45, ha='right')
ax.set_ylabel('% Organizations Reporting Gap')
ax.set_title(data['chart1']['title'])
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('/root/hub/research/second-order-gap/charts/chart1_skills_gap_emergence.png', dpi=150)
plt.close()

# Chart 2: Scatter/bubble - Productivity vs gap creation
fig, ax = plt.subplots(figsize=(9, 6))
points = data['chart2']['points']
sizes = [200, 280, 360, 440]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
for i, p in enumerate(points):
    ax.scatter(p['productivity_gain'], p['gaps_reported'], s=sizes[i], c=colors[i], alpha=0.7, label=p['label'])
ax.set_xlabel('Productivity Gain (%)')
ax.set_ylabel('Average New Gaps Reported')
ax.set_title(data['chart2']['title'])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/root/hub/research/second-order-gap/charts/chart2_productivity_paradox.png', dpi=150)
plt.close()

# Chart 3: Horizontal bar - Gap distribution by function
fig, ax = plt.subplots(figsize=(9, 6))
cats = data['chart3']['categories']
vals = data['chart3']['values']
bars = ax.barh(cats, vals, color='#2c3e50')
ax.set_xlabel('% Organizations Reporting Gap')
ax.set_title(data['chart3']['title'])
for bar, val in zip(bars, vals):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val}%', va='center')
ax.set_xlim(0, 100)
plt.tight_layout()
plt.savefig('/root/hub/research/second-order-gap/charts/chart3_gap_distribution.png', dpi=150)
plt.close()

# Chart 4: Line chart - Gap severity over time
fig, ax = plt.subplots(figsize=(10, 6))
months = data['chart4']['months']
ax.plot(months, data['chart4']['large_enterprise'], 'o-', label='Large Enterprise', linewidth=2)
ax.plot(months, data['chart4']['mid_enterprise'], 's--', label='Mid-size', linewidth=2)
ax.plot(months, data['chart4']['small_enterprise'], '^-.', label='Small', linewidth=2)
ax.set_xlabel('Months Post-Adoption')
ax.set_ylabel('Gap Severity Score (1-10)')
ax.set_title(data['chart4']['title'])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/root/hub/research/second-order-gap/charts/chart4_gap_severity.png', dpi=150)
plt.close()

print("4 charts generated successfully")