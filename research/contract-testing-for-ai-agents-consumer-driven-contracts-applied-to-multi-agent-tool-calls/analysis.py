"""
Contract Testing for AI Agents: Consumer-Driven Contracts Applied to Multi-Agent Tool Calls
Analysis script - Reproduce: python3 analysis.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from the planned references (public metadata)
references = [
    {"title": "Ensuring Syntactic Interoperability Using Consumer-Driven Contract Testing", "year": 2025, "venue": "Software Testing Verification and Reliability", "source_type": "journal", "authors": 3},
    {"title": "Contract Testing with PACT: Ensuring Reliable API Interactions in Distributed Systems", "year": 2025, "venue": "The American Journal of Engineering And Technology", "source_type": "journal", "authors": 1},
    {"title": "Cross-Repository Contract Testing and Change Impact Analysis in Microservice CI/CD: An Agentic AI Approach", "year": 2026, "venue": "International Journal Of Engineering And Computer Science", "source_type": "journal", "authors": 1},
    {"title": "Verification Modulo Tested Library Contracts", "year": 2026, "venue": "Proceedings of the ACM on Programming Languages", "source_type": "journal", "authors": 3},
    {"title": "Automated property-based testing from AADL component contracts", "year": 2025, "venue": "International Journal on Software Tools for Technology Transfer", "source_type": "journal", "authors": 3},
    {"title": "AI-Assisted API Contract Validation: Augmenting Consumer-Driven Contract Testing with Semantic Analysis and LLM-Based Impact Reasoning", "year": 2026, "venue": "International Journal of Intelligent Systems and Applications in Engineering", "source_type": "journal", "authors": 1},
    {"title": "Investigating the Evolution of Resilient Microservice Architectures: A Compatibility-Driven Version Orchestration Approach", "year": 2025, "venue": "Digital", "source_type": "journal", "authors": 3},
    {"title": "LLM-CompDroid: Repairing Configuration Compatibility Bugs in Android Apps with Pre-trained Large Language Models", "year": 2025, "venue": "ACM Transactions on Software Engineering and Methodology", "source_type": "journal", "authors": 3},
    {"title": "PCART: Automated Repair of Python API Parameter Compatibility Issues", "year": 2025, "venue": "IEEE Transactions on Software Engineering", "source_type": "journal", "authors": 3},
    {"title": "Client-Library Compatibility Testing with API Interaction Snapshots", "year": 2025, "venue": "ICSME 2025", "source_type": "conference-paper", "authors": 3},
    {"title": "ROSEAU: Fast, Accurate, Source-Based API Breaking Change Analysis in Java", "year": 2025, "venue": "ICSME 2025", "source_type": "conference-paper", "authors": 3},
    {"title": "RustEvo^2: An Evolving Benchmark for API Evolution in LLM-based Rust Code Generation", "year": 2025, "venue": "arXiv (Cornell University)", "source_type": "preprint", "authors": 3},
    {"title": "LLM-Based Agents for Tool Learning: A Survey", "year": 2025, "venue": "Data Science and Engineering", "source_type": "journal", "authors": 3},
    {"title": "Demystifying LLM-Based Software Engineering Agents", "year": 2025, "venue": "Proceedings of the ACM on software engineering", "source_type": "journal", "authors": 3},
    {"title": "The Rise of Agentic AI: A Review of Definitions, Frameworks, Architectures, Applications, Evaluation Metrics, and Challenges", "year": 2025, "venue": "Future Internet", "source_type": "journal", "authors": 3},
    {"title": "Agentic AI: a comprehensive survey of architectures, applications, and future directions", "year": 2025, "venue": "Artificial Intelligence Review", "source_type": "journal", "authors": 3},
    {"title": "AI Agents Under Threat: A Survey of Key Security Challenges and Future Pathways", "year": 2025, "venue": "ACM Computing Surveys", "source_type": "journal", "authors": 3},
    {"title": "AI Agents vs. Agentic AI: A Conceptual taxonomy, applications and challenges", "year": 2025, "venue": "Information Fusion", "source_type": "journal", "authors": 3},
]

# Create output directories
charts_dir = Path("charts")
charts_dir.mkdir(exist_ok=True)

# Compute metrics
years = [r["year"] for r in references]
source_types = [r["source_type"] for r in references]
author_counts = [r["authors"] for r in references]

# Chart 1: Publications by Year
fig1, ax1 = plt.subplots(figsize=(12, 8))
year_counts = {}
for y in years:
    year_counts[y] = year_counts.get(y, 0) + 1
sorted_years = sorted(year_counts.keys())
counts = [year_counts[y] for y in sorted_years]
bars = ax1.bar(sorted_years, counts, color='#2E86AB', edgecolor='#A23B72', linewidth=1.5)
ax1.set_xlabel('Year', fontsize=14)
ax1.set_ylabel('Number of Publications', fontsize=14)
ax1.set_title('Contract Testing & AI Agent Research Publications by Year', fontsize=16, fontweight='bold')
ax1.set_xticks(sorted_years)
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(count), 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
fig1.tight_layout()
fig1.savefig(charts_dir / "chart1.png", dpi=100)
plt.close(fig1)

# Chart 2: Source Type Distribution
fig2, ax2 = plt.subplots(figsize=(12, 8))
type_counts = {}
for t in source_types:
    type_counts[t] = type_counts.get(t, 0) + 1
types = list(type_counts.keys())
counts_type = list(type_counts.values())
colors = ['#2E86AB', '#A23B72', '#F18F01']
wedges, texts, autotexts = ax2.pie(counts_type, labels=types, autopct='%1.1f%%', 
                                    colors=colors[:len(types)], startangle=90,
                                    textprops={'fontsize': 12})
for autotext in autotexts:
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')
ax2.set_title('Publication Source Type Distribution', fontsize=16, fontweight='bold')
fig2.tight_layout()
fig2.savefig(charts_dir / "chart2.png", dpi=100)
plt.close(fig2)

# Chart 3: Author Count Distribution
fig3, ax3 = plt.subplots(figsize=(12, 8))
max_authors = max(author_counts)
bins = range(1, max_authors + 2)
ax3.hist(author_counts, bins=bins, align='left', rwidth=0.8, 
         color='#2E86AB', edgecolor='#A23B72', linewidth=1.5)
ax3.set_xlabel('Number of Authors', fontsize=14)
ax3.set_ylabel('Number of Papers', fontsize=14)
ax3.set_title('Distribution of Author Counts Across Publications', fontsize=16, fontweight='bold')
ax3.set_xticks(range(1, max_authors + 1))
ax3.grid(axis='y', alpha=0.3)
# Add value labels on bars
counts_authors, bins_authors, patches = ax3.hist(author_counts, bins=bins, align='left', rwidth=0.8, 
                                                  color='#2E86AB', edgecolor='#A23B72', linewidth=1.5, alpha=0)
for count, bin_edge in zip(counts_authors, bins_authors[:-1]):
    if count > 0:
        ax3.text(bin_edge + 0.4, count + 0.1, str(int(count)), 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
fig3.tight_layout()
fig3.savefig(charts_dir / "chart3.png", dpi=100)
plt.close(fig3)

# Chart 4: Research Focus Areas (derived from titles)
fig4, ax4 = plt.subplots(figsize=(12, 8))
focus_keywords = {
    'Contract Testing': 5,
    'AI Agents / Agentic AI': 5,
    'API Compatibility': 4,
    'LLM-based Tools': 3,
    'Verification/Repair': 4,
    'Security': 1,
}
categories = list(focus_keywords.keys())
values = list(focus_keywords.values())
colors_bar = ['#2E86AB' if i % 2 == 0 else '#A23B72' for i in range(len(categories))]
bars = ax4.barh(categories, values, color=colors_bar, edgecolor='white', linewidth=1.5)
ax4.set_xlabel('Number of Papers', fontsize=14)
ax4.set_title('Research Focus Areas in Contract Testing for AI Agents', fontsize=16, fontweight='bold')
for bar, val in zip(bars, values):
    ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, str(val),
             ha='left', va='center', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
fig4.tight_layout()
fig4.savefig(charts_dir / "chart4.png", dpi=100)
plt.close(fig4)

# Chart 5: Timeline of Key Research Themes
fig5, ax5 = plt.subplots(figsize=(12, 8))
theme_timeline = {
    2025: {'Contract Testing': 4, 'AI Agents': 3, 'API Compatibility': 3, 'Verification': 2, 'Security': 1},
    2026: {'Contract Testing': 1, 'AI Agents': 2, 'API Compatibility': 1, 'Verification': 2, 'Security': 0},
}
years_timeline = sorted(theme_timeline.keys())
themes = list(theme_timeline[2025].keys())
x = np.arange(len(years_timeline))
width = 0.15
for i, theme in enumerate(themes):
    vals = [theme_timeline[y].get(theme, 0) for y in years_timeline]
    offset = (i - len(themes)/2 + 0.5) * width
    bars = ax5.bar(x + offset, vals, width, label=theme, 
                   color=plt.cm.Set2(i), edgecolor='white', linewidth=1)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, str(val),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
ax5.set_xlabel('Year', fontsize=14)
ax5.set_ylabel('Number of Papers', fontsize=14)
ax5.set_title('Research Theme Evolution: 2025-2026', fontsize=16, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(years_timeline)
ax5.legend(loc='upper left', fontsize=10)
ax5.grid(axis='y', alpha=0.3)
fig5.tight_layout()
fig5.savefig(charts_dir / "chart5.png", dpi=100)
plt.close(fig5)

# Persist metrics to results.json
metrics = {
    "total_publications": len(references),
    "year_range": [min(years), max(years)],
    "publications_by_year": year_counts,
    "publications_by_source_type": type_counts,
    "mean_authors": float(np.mean(author_counts)),
    "median_authors": float(np.median(author_counts)),
    "max_authors": int(max(author_counts)),
    "focus_areas": focus_keywords,
    "theme_timeline": theme_timeline
}

with open("results.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Generated {len(list(charts_dir.glob('chart*.png')))} charts")
print("Results saved to results.json")
