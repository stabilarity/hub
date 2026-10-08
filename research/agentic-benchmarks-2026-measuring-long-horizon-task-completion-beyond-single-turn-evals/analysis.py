"""
Agentic Benchmarks 2026 Analysis Script
Analyzes multi-step agent benchmarks and their properties.

Reproduce: python3 analysis.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Reference data from the article (public knowledge)
BENCHMARKS = [
    {"name": "AgenticRAGTracer", "year": 2026, "venue": "ACL", "type": "conference", "domain": "RAG/Retrieval", "steps": "multi-hop"},
    {"name": "EpiBench", "year": 2026, "venue": "LNCS", "type": "conference", "domain": "Research Workflows", "steps": "multi-turn"},
    {"name": "Distributed Sorting", "year": 2026, "venue": "ACL", "type": "conference", "domain": "Multi-Agent", "steps": "multi-agent"},
    {"name": "Proto-AGI Memory", "year": 2026, "venue": "LNCS", "type": "conference", "domain": "Cognitive Architecture", "steps": "long-horizon"},
    {"name": "ColorBench", "year": 2026, "venue": "WWW", "type": "conference", "domain": "Mobile Agents", "steps": "graph-structured"},
    {"name": "LeSkill", "year": 2025, "venue": "IEEE TSMC", "type": "journal", "domain": "Robotics", "steps": "skill-learning"},
    {"name": "ABC-Bench", "year": 2026, "venue": "ACL", "type": "conference", "domain": "Code Generation", "steps": "repository-level"},
    {"name": "ShoppingBench", "year": 2026, "venue": "AAAI", "type": "conference", "domain": "Shopping", "steps": "intent-grounded"},
    {"name": "Repo-Level Code", "year": 2026, "venue": "ACL", "type": "conference", "domain": "Code Reasoning", "steps": "repository"},
    {"name": "AgentGym2", "year": 2026, "venue": "ACL", "type": "conference", "domain": "General Agents", "steps": "de-idealized"},
    {"name": "Agent-RewardBench", "year": 2025, "venue": "ACL", "type": "conference", "domain": "Reward Modeling", "steps": "multimodal"},
    {"name": "LLM Agent Survey", "year": 2025, "venue": "KDD", "type": "conference", "domain": "Survey", "steps": "overview"},
    {"name": "DABstep", "year": 2025, "venue": "arXiv", "type": "preprint", "domain": "Data Agents", "steps": "multi-step"},
    {"name": "AgenticRAGTracer (arXiv)", "year": 2026, "venue": "arXiv", "type": "preprint", "domain": "RAG/Retrieval", "steps": "multi-hop"},
    {"name": "Sci-MMR", "year": 2026, "venue": "arXiv", "type": "preprint", "domain": "Scientific Reasoning", "steps": "multi-step"},
    {"name": "SciAgentGym", "year": 2026, "venue": "arXiv", "type": "preprint", "domain": "Scientific Tool-use", "steps": "multi-step"},
    {"name": "E-Bench", "year": 2026, "venue": "arXiv", "type": "preprint", "domain": "Product Scenarios", "steps": "multi-step"},
    {"name": "StepJack", "year": 2026, "venue": "arXiv", "type": "preprint", "domain": "Safety", "steps": "multi-step"},
    {"name": "Multi-Agent Coord", "year": 2026, "venue": "arXiv", "type": "preprint", "domain": "Coordination", "steps": "open-ended"},
    {"name": "StarCraft+", "year": 2025, "venue": "arXiv", "type": "preprint", "domain": "Adversarial", "steps": "multi-agent"},
]

def create_charts():
    """Generate all required charts."""
    df = pd.DataFrame(BENCHMARKS)
    charts_dir = Path("charts")
    charts_dir.mkdir(exist_ok=True)
    
    # Chart 1: Benchmarks by Year and Venue Type
    fig, ax = plt.subplots(figsize=(12, 8))
    year_venue = df.groupby(['year', 'type']).size().unstack(fill_value=0)
    year_venue.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_title('Agentic Benchmarks by Year and Publication Type (2025-2026)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Benchmarks', fontsize=12)
    ax.legend(title='Publication Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(charts_dir / "chart1.png", dpi=100)
    plt.close()
    
    # Chart 2: Domain Distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    domain_counts = df['domain'].value_counts()
    colors = plt.cm.Set3(np.arange(len(domain_counts)))
    bars = ax.barh(range(len(domain_counts)), domain_counts.values, color=colors)
    ax.set_yticks(range(len(domain_counts)))
    ax.set_yticklabels(domain_counts.index, fontsize=11)
    ax.set_xlabel('Number of Benchmarks', fontsize=12)
    ax.set_title('Benchmark Domains Distribution', fontsize=14, fontweight='bold')
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, domain_counts.values)):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=10)
    ax.set_xlim(0, max(domain_counts.values) + 1.5)
    plt.tight_layout()
    plt.savefig(charts_dir / "chart2.png", dpi=100)
    plt.close()
    
    # Chart 3: Publication Venue Distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    venue_counts = df['venue'].value_counts()
    colors = plt.cm.tab20(np.arange(len(venue_counts)))
    wedges, texts, autotexts = ax.pie(venue_counts.values, labels=venue_counts.index, autopct='%1.0f%%',
                                       colors=colors, startangle=90)
    ax.set_title('Publication Venue Distribution', fontsize=14, fontweight='bold')
    plt.setp(autotexts, size=10, weight='bold')
    plt.setp(texts, size=10)
    plt.tight_layout()
    plt.savefig(charts_dir / "chart3.png", dpi=100)
    plt.close()
    
    # Chart 4: Step Complexity Categories
    fig, ax = plt.subplots(figsize=(12, 8))
    step_counts = df['steps'].value_counts()
    colors = plt.cm.Pastel1(np.arange(len(step_counts)))
    bars = ax.bar(range(len(step_counts)), step_counts.values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(step_counts)))
    ax.set_xticklabels(step_counts.index, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Number of Benchmarks', fontsize=12)
    ax.set_title('Step Complexity Categories Across Benchmarks', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, step_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(val), 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(step_counts.values) + 1.5)
    plt.tight_layout()
    plt.savefig(charts_dir / "chart4.png", dpi=100)
    plt.close()
    
    # Chart 5: Year-over-Year Growth
    fig, ax = plt.subplots(figsize=(12, 8))
    yearly = df.groupby('year').size()
    ax.plot(yearly.index, yearly.values, 'o-', linewidth=3, markersize=10, color='#2E86AB')
    ax.fill_between(yearly.index, yearly.values, alpha=0.3, color='#2E86AB')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of New Benchmarks', fontsize=12)
    ax.set_title('Year-over-Year Growth in Agentic Benchmarks', fontsize=14, fontweight='bold')
    ax.set_xticks(yearly.index)
    ax.grid(True, alpha=0.3)
    for x, y in zip(yearly.index, yearly.values):
        ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(charts_dir / "chart5.png", dpi=100)
    plt.close()
    
    return df

def compute_metrics(df):
    """Compute summary metrics."""
    metrics = {
        "total_benchmarks": int(len(df)),
        "by_year": df['year'].value_counts().to_dict(),
        "by_type": df['type'].value_counts().to_dict(),
        "by_venue": df['venue'].value_counts().to_dict(),
        "by_domain": df['domain'].value_counts().to_dict(),
        "by_steps": df['steps'].value_counts().to_dict(),
        "conference_count": int(len(df[df['type'] == 'conference'])),
        "journal_count": int(len(df[df['type'] == 'journal'])),
        "preprint_count": int(len(df[df['type'] == 'preprint'])),
        "acl_count": int(len(df[df['venue'] == 'ACL'])),
        "arxiv_count": int(len(df[df['venue'] == 'arXiv'])),
        "unique_domains": int(df['domain'].nunique()),
    }
    return metrics

def main():
    df = create_charts()
    metrics = compute_metrics(df)
    
    # Save results
    with open("results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Generated {len(list(Path('charts').glob('*.png')))} charts")
    print("Saved metrics to results.json")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
