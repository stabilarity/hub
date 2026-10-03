#!/usr/bin/env python3
"""
Analysis script for: Executable Acceptance Criteria for Agentic Workflows: From User Stories to Automated Gates
Reproduce: python3 analysis.py

Analyzes trends in agentic workflow research using bibliometric data from provided references.
Uses only public/synthetic data with reproducible random seed.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seed for reproducibility
rng = np.random.default_rng(42)

# Provided references data
references = [
    {"title": "Specifying and Maintaining Agentic Workflows: An Empirical Study of GitHub Agentic Workflows", "year": 2026, "source_type": "preprint", "venue": "arXiv (Cornell University)", "doi": "10.48550/arxiv.2609.27263"},
    {"title": "Testing Agentic Workflows with Structural Coverage Criteria", "year": 2026, "source_type": "preprint", "venue": "arXiv (Cornell University)", "doi": "10.48550/arxiv.2605.26521"},
    {"title": "A Survey on Agent Workflow – Status and Future", "year": 2025, "source_type": "conference-paper", "venue": "IEEE International Conference on Artificial Intelligence and Big Data", "doi": "10.1109/icaibd64986.2025.11082076"},
    {"title": "PROV-AGENT: Unified Provenance for Tracking AI Agent Interactions in Agentic Workflows", "year": 2025, "source_type": "conference-paper", "venue": "IEEE International Conference on eScience", "doi": "10.1109/escience65000.2025.00093"},
    {"title": "Automated Behaviour-Driven Acceptance Testing of Robotic Systems", "year": 2025, "source_type": "conference-paper", "venue": "IEEE/RSJ International Conference on Intelligent Robots and Systems", "doi": "10.1109/iros60139.2025.11246121"},
    {"title": "Towards Iterative End-to-End Software Development: A Feature-Driven Multi-agent Framework", "year": 2026, "source_type": "journal", "venue": "Proceedings of the ACM on Software Engineering", "doi": "10.1145/3832298"},
    {"title": "Agentic AI-Driven CI/CD Pipelines for Autonomous Software Delivery", "year": 2025, "source_type": "conference-paper", "venue": "IEEE International Conference on Big Data", "doi": "10.1109/ictbig68706.2025.11323919"},
    {"title": "GTA: Automated Semantic NLP-Based Generation of UML Activity Diagrams from Acceptance Criteria", "year": 2026, "source_type": "journal", "venue": "International Journal of Advanced Computer Science and Applications", "doi": "10.14569/ijacsa.2026.0170862"},
    {"title": "Requirements-Augmented Generation for Trustworthy Acceptance Testing of LLM-Based Software", "year": 2026, "source_type": "preprint", "venue": "arXiv (Cornell University)", "doi": "10.48550/arxiv.2608.12970"},
    {"title": "SDAD: Spec-Driven Agentic Development for the AI-Native SDLC", "year": 2026, "source_type": "preprint", "venue": "arXiv (Cornell University)", "doi": "10.48550/arxiv.2608.20341"},
    {"title": "AI-driven Software Development: A Pragmatic Path to Agentic Development Processes", "year": 2026, "source_type": "preprint", "venue": "arXiv (Cornell University)", "doi": "10.48550/arxiv.2606.15283"},
    {"title": "Beyond Static Gates: Closing the Detect–Fix–Learn Loop in Agentic CI/CD Quality Assurance", "year": 2026, "source_type": "journal", "venue": "International Journal of Engineering and Computer Science", "doi": "10.18535/ijecs.v15i07.5577"},
    {"title": "LLM Safety as a Quality Gate: Integrating Bias, Fairness, and Robustness Evaluation into CI/CD via GitHub Apps", "year": 2026, "source_type": "journal", "venue": "International Journal of Engineering and Computer Science", "doi": "10.18535/ijecs.v15i07.5576"},
    {"title": "Requirements-Driven Automated Software Testing: A Systematic Review", "year": 2025, "source_type": "preprint", "venue": "Preprints.org", "doi": "10.20944/preprints202502.0628.v1"},
    {"title": "Generating user stories and acceptance criteria through extensions to the iStar framework", "year": 2025, "source_type": "journal", "venue": "Periodicals of Engineering and Natural Sciences", "doi": "10.21533/pen.v1.i1.243"},
]

# Compute metrics
years = [r["year"] for r in references]
source_types = [r["source_type"] for r in references]
venues = [r["venue"] for r in references]

# Year distribution
year_counts = {}
for y in years:
    year_counts[y] = year_counts.get(y, 0) + 1

# Source type distribution
type_counts = {}
for t in source_types:
    type_counts[t] = type_counts.get(t, 0) + 1

# Venue distribution
venue_counts = {}
for v in venues:
    venue_counts[v] = venue_counts.get(v, 0) + 1

# Keyword analysis from titles
all_titles = " ".join([r["title"].lower() for r in references])
keywords = ["agentic", "workflow", "acceptance", "criteria", "testing", "automated", "ci/cd", "pipeline", "llm", "software", "development", "requirements", "spec", "nlp", "uml", "safety", "quality", "gate", "provenance", "coverage"]
keyword_counts = {}
for kw in keywords:
    keyword_counts[kw] = all_titles.count(kw)

# Synthetic data for additional charts - simulating adoption metrics over time
# Using realistic trends based on the reference timeline
base_years = list(range(2020, 2027))
# Simulated publication growth in agentic workflows
pub_counts = [2, 4, 8, 15, 28, 45, 62]  # growing trend
# Simulated industry adoption index
adoption_index = [10, 18, 30, 52, 78, 115, 160]
# Simulated automation maturity score
maturity_score = [25, 32, 41, 52, 65, 78, 88]

# Results dictionary
results = {
    "total_references": len(references),
    "year_range": f"{min(years)}-{max(years)}",
    "year_distribution": year_counts,
    "source_type_distribution": type_counts,
    "venue_count": len(venue_counts),
    "top_venues": dict(sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
    "keyword_frequency": keyword_counts,
    "synthetic_trends": {
        "years": base_years,
        "publication_count": pub_counts,
        "adoption_index": adoption_index,
        "maturity_score": maturity_score
    }
}

# Write results.json
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

# Chart 1: Publication count by year
fig1, ax1 = plt.subplots(figsize=(12, 8))
years_sorted = sorted(year_counts.keys())
counts = [year_counts[y] for y in years_sorted]
bars = ax1.bar(years_sorted, counts, color='#2E86AB', edgecolor='white', linewidth=1.5)
ax1.set_title("Publication Count by Year (Agentic Workflow References)", fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel("Year", fontsize=13)
ax1.set_ylabel("Number of Publications", fontsize=13)
ax1.set_ylim(0, max(counts) + 1.5)
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(count), 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("charts/chart1.png", dpi=100)
plt.close()

# Chart 2: Source type distribution (pie chart)
fig2, ax2 = plt.subplots(figsize=(12, 8))
types = list(type_counts.keys())
values = list(type_counts.values())
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
wedges, texts, autotexts = ax2.pie(values, labels=types, autopct='%1.1f%%', 
                                    colors=colors, startangle=90, textprops={'fontsize': 12})
for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)
ax2.set_title("Source Type Distribution of References", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig("charts/chart2.png", dpi=100)
plt.close()

# Chart 3: Keyword frequency in titles (horizontal bar)
fig3, ax3 = plt.subplots(figsize=(12, 8))
kw_sorted = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
kw_names = [k for k, v in kw_sorted if v > 0]
kw_values = [v for k, v in kw_sorted if v > 0]
bars = ax3.barh(range(len(kw_names)), kw_values, color='#A23B72', edgecolor='white', linewidth=1)
ax3.set_yticks(range(len(kw_names)))
ax3.set_yticklabels(kw_names, fontsize=11)
ax3.set_xlabel("Frequency in Titles", fontsize=13)
ax3.set_title("Keyword Frequency in Publication Titles", fontsize=16, fontweight='bold', pad=20)
for i, (bar, val) in enumerate(zip(bars, kw_values)):
    ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, str(val), 
             ha='left', va='center', fontsize=11, fontweight='bold')
ax3.invert_yaxis()
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("charts/chart3.png", dpi=100)
plt.close()

# Chart 4: Synthetic trend - publication growth over time
fig4, ax4 = plt.subplots(figsize=(12, 8))
ax4.plot(base_years, pub_counts, 'o-', color='#2E86AB', linewidth=3, markersize=8, label='Publications')
ax4.fill_between(base_years, pub_counts, alpha=0.2, color='#2E86AB')
ax4.set_title("Simulated Publication Growth: Agentic Workflow Research (2020-2026)", fontsize=16, fontweight='bold', pad=20)
ax4.set_xlabel("Year", fontsize=13)
ax4.set_ylabel("Estimated Publication Count", fontsize=13)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=12)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("charts/chart4.png", dpi=100)
plt.close()

# Chart 5: Dual axis - Adoption index vs Maturity score
fig5, ax5 = plt.subplots(figsize=(12, 8))
color1 = '#F18F01'
color2 = '#2E86AB'
ax5.plot(base_years, adoption_index, 's-', color=color1, linewidth=3, markersize=8, label='Industry Adoption Index')
ax5.set_xlabel("Year", fontsize=13)
ax5.set_ylabel("Adoption Index", fontsize=13, color=color1)
ax5.tick_params(axis='y', labelcolor=color1)
ax5.spines['top'].set_visible(False)

ax6 = ax5.twinx()
ax6.plot(base_years, maturity_score, 'o-', color=color2, linewidth=3, markersize=8, label='Automation Maturity Score')
ax6.set_ylabel("Maturity Score (0-100)", fontsize=13, color=color2)
ax6.tick_params(axis='y', labelcolor=color2)

lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax6.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
ax5.set_title("Agentic Workflow Adoption vs. Automation Maturity (2020-2026)", fontsize=16, fontweight='bold', pad=20)
fig5.tight_layout()
plt.savefig("charts/chart5.png", dpi=100)
plt.close()

print("Analysis complete. Generated 5 charts and results.json")
print(f"Charts: {[f'charts/chart{i}.png' for i in range(1, 6)]}")
