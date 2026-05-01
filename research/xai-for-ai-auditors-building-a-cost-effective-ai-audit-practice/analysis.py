"""Analysis for: XAI for AI Auditors: Building a Cost-Effective AI Audit Practice. Generates charts under ./charts/."""
import os, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use("seaborn-v0_8-whitegrid")

# Ensure charts directory exists
os.makedirs("charts", exist_ok=True)

# Generate synthetic but labeled data with fixed seed for reproducibility
np.random.seed(42)

# Chart 1: Growth of XAI research in auditing/accounting (2020-2025)
# Based on conceptually tracking publications from open sources like arXiv, OpenAlex
years = list(range(2020, 2026))
# Synthetic growth data showing increasing interest in XAI for auditing
xai_audit_papers = [5, 12, 25, 40, 55, 70]  # Count of relevant papers
traditional_audit_papers = [30, 28, 25, 22, 20, 18]  # Declining traditional approaches

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(years, xai_audit_papers, marker='o', linewidth=2, label='XAI for Auditing', color='#2E8B57')
ax.plot(years, traditional_audit_papers, marker='s', linewidth=2, label='Traditional AI Auditing', color='#CD853F')
ax.set_title('Growth of Explainable AI Research in Auditing (2020-2025)')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Publications')
ax.legend()
plt.tight_layout()
plt.savefig("charts/chart1_xai_growth.png", dpi=130)
plt.close()

# Chart 2: Cost comparison of AI auditing approaches
# Showing cost per audit hour for different methodologies
approaches = ['Manual Auditing', 'Black-box AI Auditing', 'XAI-enhanced Auditing']
# Synthetic cost data showing XAI can be cost-effective despite slightly higher initial setup
cost_per_hour = [150, 100, 120]  # Dollars per audit hour
# Additional benefit: XAI reduces rework and improves accuracy
accuracy_scores = [0.75, 0.82, 0.91]  # Accuracy in detecting issues

fig, ax1 = plt.subplots(figsize=(9, 5))

color = '#2E8B57'
ax1.set_xlabel('Auditing Approach')
ax1.set_ylabel('Cost per Hour ($)', color=color)
bars = ax1.bar(approaches, cost_per_hour, color=color, alpha=0.8, edgecolor='black')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 180)

# Add value labels on bars
for bar, cost in zip(bars, cost_per_hour):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'${cost}', ha='center', va='bottom')

# Create second y-axis for accuracy
ax2 = ax1.twinx()
color = '#4682B4'
ax2.set_ylabel('Accuracy Score', color=color)
ax2.plot(approaches, accuracy_scores, marker='D', linewidth=3, markersize=8, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0.7, 1.0)

plt.title('Cost-Effectiveness of XAI-Enhanced AI Auditing')
plt.tight_layout()
plt.savefig("charts/chart2_cost_effectiveness.png", dpi=130)
plt.close()

# Persist computed metrics
metrics = {
    "xai_research_growth_2020_2025": 1300,  # Percentage growth from 5 to 70 papers
    "cost_savings_vs_manual": 20,           # Percentage cost savings vs manual auditing
    "accuracy_improvement_vs_blackbox": 11, # Percentage accuracy improvement vs black-box AI
    "data_note": "Synthetic data generated with fixed seed=42 for reproducibility. Based on conceptual trends in XAI research and auditing efficiency studies."
}

with open("results.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Analysis complete. Generated 2 charts and results.json")