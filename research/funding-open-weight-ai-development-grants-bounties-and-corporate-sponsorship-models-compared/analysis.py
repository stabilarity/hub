"""
Funding Open-Weight AI Development: Grants, Bounties, and Corporate Sponsorship Models Compared
Reproduce: python3 analysis.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure charts directory exists
charts_dir = Path(__file__).parent / "charts"
charts_dir.mkdir(exist_ok=True)

# Reproducible synthetic data
rng = np.random.default_rng(42)

# Synthetic data for 3 funding mechanisms across multiple open-weight AI projects
mechanisms = ["Grants", "Bounties", "Corporate Sponsorship"]
projects = ["Llama", "Mistral", "Falcon", "MPT", "BLOOM", "StableLM", "Pythia", "RedPajama"]

n_projects = len(projects)
n_mechanisms = len(mechanisms)

# Funding amounts (USD millions) per project per mechanism
funding_data = {}
for mech in mechanisms:
    funding_data[mech] = rng.lognormal(mean=2.5, sigma=0.8, size=n_projects).round(1)

# Release cadence (months between releases) per mechanism
cadence_data = {}
for mech in mechanisms:
    base = {"Grants": 18, "Bounties": 8, "Corporate Sponsorship": 6}[mech]
    cadence_data[mech] = (base + rng.normal(0, 2, size=n_projects)).clip(3, 36).round(1)

# Contributor count per mechanism
contributors_data = {}
for mech in mechanisms:
    base = {"Grants": 15, "Bounties": 45, "Corporate Sponsorship": 80}[mech]
    contributors_data[mech] = (base + rng.normal(0, base*0.3, size=n_projects)).clip(5, 200).astype(int)

# Compute metrics
metrics = {
    "funding_by_mechanism": {m: float(funding_data[m].sum()) for m in mechanisms},
    "avg_funding_per_project": {m: float(funding_data[m].mean()) for m in mechanisms},
    "avg_release_cadence_months": {m: float(np.mean(cadence_data[m])) for m in mechanisms},
    "avg_contributors": {m: float(np.mean(contributors_data[m])) for m in mechanisms},
    "total_projects": n_projects,
    "projects": projects,
    "mechanisms": mechanisms
}

# Save results.json
with open(Path(__file__).parent / "results.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Chart 1: Total Funding by Mechanism (Bar Chart)
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['#2ecc71', '#3498db', '#e74c3c']
totals = [metrics["funding_by_mechanism"][m] for m in mechanisms]
bars = ax.bar(mechanisms, totals, color=colors, edgecolor='black', linewidth=1.2)
for bar, val in zip(bars, totals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'${val:.1f}M',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Funding (USD Millions)', fontsize=14)
ax.set_title('Total Funding by Mechanism Across Open-Weight AI Projects', fontsize=16, fontweight='bold')
ax.set_ylim(0, max(totals) * 1.2)
plt.tight_layout()
plt.savefig(charts_dir / "chart1.png", dpi=100)
plt.close()

# Chart 2: Average Release Cadence by Mechanism (Horizontal Bar Chart)
fig, ax = plt.subplots(figsize=(12, 8))
cadences = [metrics["avg_release_cadence_months"][m] for m in mechanisms]
bars = ax.barh(mechanisms, cadences, color=colors, edgecolor='black', linewidth=1.2)
for bar, val in zip(bars, cadences):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f} months',
            ha='left', va='center', fontsize=12, fontweight='bold')
ax.set_xlabel('Average Months Between Releases', fontsize=14)
ax.set_title('Average Release Cadence by Funding Mechanism', fontsize=16, fontweight='bold')
ax.invert_yaxis()
ax.set_xlim(0, max(cadences) * 1.3)
plt.tight_layout()
plt.savefig(charts_dir / "chart2.png", dpi=100)
plt.close()

# Chart 3: Contributors vs Funding (Scatter Plot with mechanism coloring)
fig, ax = plt.subplots(figsize=(12, 8))
for i, mech in enumerate(mechanisms):
    x = funding_data[mech]
    y = contributors_data[mech]
    ax.scatter(x, y, c=colors[i], label=mech, s=120, alpha=0.7, edgecolors='black', linewidth=1)
    # Add project labels
    for j, proj in enumerate(projects):
        ax.annotate(proj, (x[j], y[j]), fontsize=9, alpha=0.7,
                   xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('Funding (USD Millions)', fontsize=14)
ax.set_ylabel('Number of Contributors', fontsize=14)
ax.set_title('Funding vs Contributor Count by Mechanism', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, title='Mechanism', title_fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(charts_dir / "chart3.png", dpi=100)
plt.close()

# Chart 4: Funding Distribution by Project (Stacked Bar)
fig, ax = plt.subplots(figsize=(12, 8))
bottom = np.zeros(n_projects)
for i, mech in enumerate(mechanisms):
    values = funding_data[mech]
    ax.bar(projects, values, bottom=bottom, label=mech, color=colors[i], edgecolor='black', linewidth=0.8)
    bottom += values
ax.set_ylabel('Funding (USD Millions)', fontsize=14)
ax.set_title('Funding Composition by Project and Mechanism', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, title='Mechanism', title_fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(charts_dir / "chart4.png", dpi=100)
plt.close()

# Chart 5: Release Cadence vs Funding (Bubble Chart)
fig, ax = plt.subplots(figsize=(12, 8))
for i, mech in enumerate(mechanisms):
    x = funding_data[mech]
    y = cadence_data[mech]
    sizes = contributors_data[mech] * 5
    ax.scatter(x, y, s=sizes, c=colors[i], label=mech, alpha=0.6, edgecolors='black', linewidth=1)
    for j, proj in enumerate(projects):
        ax.annotate(proj, (x[j], y[j]), fontsize=9, alpha=0.7,
                   xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('Funding (USD Millions)', fontsize=14)
ax.set_ylabel('Release Cadence (Months)', fontsize=14)
ax.set_title('Funding vs Release Cadence (Bubble Size = Contributors)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, title='Mechanism', title_fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(charts_dir / "chart5.png", dpi=100)
plt.close()

print("Analysis complete. Generated 5 charts and results.json")
print(f"Charts saved to: {charts_dir}")
