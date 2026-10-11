"""
Interoperable Agent Protocols: Evaluating MCP and Successor Standards for Cross-Vendor Tool Use
Analysis script generating charts from planned references.

Reproduce: python3 analysis.py
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Planned references data
references = [
    {"title": "The New Interoperability Paradigm: Model Context Protocol (MCP), APIs, and the Future of Agentic AI", "url": "https://doi.org/10.52710/cfs.817", "doi": "10.52710/cfs.817", "year": 2025, "authors": "Padmanabhan Venkiteela", "source_type": "journal", "venue": "Computer Fraud & Security"},
    {"title": "Model context protocol for MBSE: a unified interoperable agentic framework for systems modelling", "url": "https://doi.org/10.1017/pds.2026.10630", "doi": "10.1017/pds.2026.10630", "year": 2026, "authors": "Mahmoud Bazzal, Simon Schleifer, Adriana Lungu", "source_type": "journal", "venue": "Proceedings of the Design Society"},
    {"title": "Operationalizing Multi-Agent Interoperability via Contract-Driven Model Context Protocols", "url": "https://doi.org/10.1109/southeastcon63549.2026.11476405", "doi": "10.1109/southeastcon63549.2026.11476405", "year": 2026, "authors": "Aswathnarayan Muthukrishnan Kirubakaran, Vinoth Punniyamoorthy, Adithya Parthasarathy", "source_type": "conference-paper", "venue": "Proceedings of IEEE Southeastcon"},
    {"title": "Model Context Protocol for Agentic AI: Enabling Contextual Interoperability Across Systems", "url": "https://doi.org/10.22399/ijcesen.3678", "doi": "10.22399/ijcesen.3678", "year": 2025, "authors": "Vallikranth Ayyagari", "source_type": "journal", "venue": "International Journal of Computational and Experimental Science and Engineering"},
    {"title": "Leveraging MCP and Corrective RAG for Scalable and Interoperable Multi-Agent Healthcare Systems", "url": "https://doi.org/10.3390/electronics15040888", "doi": "10.3390/electronics15040888", "year": 2026, "authors": "Dimitrios Kalathas, Andreas Menychtas, Panayiotis D. Tsanakas", "source_type": "journal", "venue": "Electronics"},
    {"title": "The MCP Standard", "url": "https://doi.org/10.1007/979-8-8688-2364-0", "doi": "10.1007/979-8-8688-2364-0", "year": 2026, "authors": "Srinivasan Sekar", "source_type": "book", "venue": "Apress eBooks"},
    {"title": "Model Context Protocol (MCP) at First Glance: Studying the Security and Maintainability of MCP Servers", "url": "https://doi.org/10.1145/3814959", "doi": "10.1145/3814959", "year": 2026, "authors": "Mohammed Mehedi Hasan, Hao Li, Emad Fallahzadeh", "source_type": "journal", "venue": "ACM Transactions on Software Engineering and Methodology"},
    {"title": "MCP Guardian: A Security-First Layer for Safeguarding MCP-Based AI System", "url": "https://doi.org/10.5121/csit.2025.150908", "doi": "10.5121/csit.2025.150908", "year": 2025, "authors": "Sonu Kumar, Anubhav Girdhar, Ritesh Patil", "source_type": "conference-paper", "venue": ""},
    {"title": "A systematic mapping study of AI agent–Tool communication: protocols, security, and orchestration", "url": "https://doi.org/10.1016/j.iswa.2026.200733", "doi": "10.1016/j.iswa.2026.200733", "year": 2026, "authors": "HAFIZ SIDDIQ UMER, Naeem Ahmed Haq Nawaz, Kashif Ishaq", "source_type": "journal", "venue": "Intelligent Systems with Applications"},
    {"title": "Standardizing AI Agent Protocols: Definitions, Layered Architectures, and Future Research Directions", "url": "https://doi.org/10.63282/3050-9262.ijaidsml-v7i1p109", "doi": "10.63282/3050-9262.ijaidsml-v7i1p109", "year": 2026, "authors": "Saurabh Atri", "source_type": "journal", "venue": "International Journal of Artificial Intelligence Data Science and Machine Learning"},
    {"title": "Agentic AI Communication Protocols and Security", "url": "https://doi.org/10.1007/978-3-032-02130-4_4", "doi": "10.1007/978-3-032-02130-4_4", "year": 2025, "authors": "Ken Huang, Chris Hughes", "source_type": "book-chapter", "venue": "Springer eBooks"},
    {"title": "Advancing Agentic AI through Communication Protocols", "url": "https://doi.org/10.32628/ijsrst25125127", "doi": "10.32628/ijsrst25125127", "year": 2025, "authors": "Aniket P. Kakde, Karan M. Bhoyar, Muhammad Aiman Shad", "source_type": "journal", "venue": "International Journal of Scientific Research in Science and Technology"},
    {"title": "Toward the Internet of Agentic AI: Protocols, Architecture, and Challenges", "url": "https://doi.org/10.1109/mcom.001.2500309", "doi": "10.1109/mcom.001.2500309", "year": 2026, "authors": "Yuzheng Ren, Jiahui Yang, Haijun Zhang", "source_type": "journal", "venue": "IEEE Communications Magazine"},
    {"title": "Securing the Agent: Vendor-Neutral, Multitenant Enterprise Retrieval and Tool Use", "url": "https://doi.org/10.1145/3786335.3813145", "doi": "10.1145/3786335.3813145", "year": 2026, "authors": "Francisco Javier Arceo, Varsha Prasad Narsing", "source_type": "conference-paper", "venue": ""},
    {"title": "Cross-vendor repeatability and reproducibility of quantitative MRI for prostate cancer applications", "url": "https://doi.org/10.58530/2025/1668", "doi": "10.58530/2025/1668", "year": 2025, "authors": "Useong Yang, Patrick J. Bolan, Susan Moyher Noworolski", "source_type": "conference-paper", "venue": "Proceedings on CD-ROM - International Society for Magnetic Resonance in Medicine"},
    {"title": "Cross-Vendor Variability Management for Cloud Systems Using the TOSCA DSL", "url": "https://doi.org/10.1145/3715340.3715433", "doi": "10.1145/3715340.3715433", "year": 2025, "authors": "Tobias Fellner, Paul Grunbacher", "source_type": "conference-paper", "venue": "International Working Conference on Variability Modelling of Software-Intensive Systems (VaMoS)"},
    {"title": "Cross-Vendor GPU Programming: Extending CUDA Beyond NVIDIA", "url": "https://doi.org/10.1145/3723851.3723860", "doi": "10.1145/3723851.3723860", "year": 2025, "authors": "Manos Pavlidakis, Chris Kitching, Nicholas Tomlinson", "source_type": "conference-paper", "venue": ""},
    {"title": "Prevalence and impact of frailty assessed using standard geriatrics tools in older adults with blood cancers", "url": "https://doi.org/10.3324/haematol.2025.300233", "doi": "10.3324/haematol.2025.300233", "year": 2026, "authors": "Clark DuMontier, Angel M. Cronin, Tammy T. Hshieh", "source_type": "journal", "venue": "Haematologica"},
    {"title": "Ethical Compliance Audit of LLMs Used as Computing Education Tools: An Evaluation Under ACM Standards", "url": "https://doi.org/10.1145/3803401.3812005", "doi": "10.1145/3803401.3812005", "year": 2026, "authors": "Francisco Jose Anillo Carrasco, Andreea Madalina Oprescu, Maria del Carmen Romero Ternero", "source_type": "conference-paper", "venue": "ACM Conference on Innovation and Technology in Computer Science Education (ITiCSE)"},
    {"title": "Benchmarking web-based in-silico toxicity prediction tools using gold-standard datasets across five key endpoints", "url": "https://doi.org/10.1007/s42452-026-08308-7", "doi": "10.1007/s42452-026-08308-7", "year": 2026, "authors": "Anirudh R Urs, Varshini Ganesan Selvi, Ananya Sreekumar", "source_type": "journal", "venue": "Discover Applied Sciences"}
]

def compute_metrics(refs):
    """Compute metrics from references."""
    years = [r["year"] for r in refs]
    source_types = [r["source_type"] for r in refs]
    venues = [r["venue"] for r in refs if r["venue"]]
    
    year_counts = Counter(years)
    type_counts = Counter(source_types)
    venue_counts = Counter(venues)
    
    # MCP-related papers
    mcp_papers = [r for r in refs if "MCP" in r["title"] or "Model Context Protocol" in r["title"]]
    cross_vendor_papers = [r for r in refs if "Cross-Vendor" in r["title"] or "Cross-vendor" in r["title"] or "vendor" in r["title"].lower()]
    
    # Count authors per paper
    author_counts = [len(r["authors"].split(",")) for r in refs]
    
    return {
        "total_papers": len(refs),
        "year_distribution": dict(year_counts),
        "source_type_distribution": dict(type_counts),
        "top_venues": dict(venue_counts.most_common(10)),
        "mcp_papers": len(mcp_papers),
        "cross_vendor_papers": len(cross_vendor_papers),
        "avg_authors": np.mean(author_counts),
        "author_distribution": dict(Counter(author_counts))
    }

def generate_chart1_year_distribution(metrics):
    """Chart 1: Publications by Year"""
    fig, ax = plt.subplots(figsize=(12, 8))
    years = sorted(metrics["year_distribution"].keys())
    counts = [metrics["year_distribution"][y] for y in years]
    
    bars = ax.bar(years, counts, color=['#2E86C1' if y == 2025 else '#E74C3C' for y in years], edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Number of Publications', fontsize=14)
    ax.set_title('Publications by Year (Planned References)', fontsize=16, fontweight='bold')
    ax.set_xticks(years)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(count), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./charts/chart1.png', dpi=100)
    plt.close()

def generate_chart2_source_types(metrics):
    """Chart 2: Source Type Distribution"""
    fig, ax = plt.subplots(figsize=(12, 8))
    types = list(metrics["source_type_distribution"].keys())
    counts = list(metrics["source_type_distribution"].values())
    
    colors = ['#2E86C1', '#E74C3C', '#27AE60', '#F39C12', '#8E44AD']
    wedges, texts, autotexts = ax.pie(counts, labels=types, autopct='%1.1f%%', 
                                       colors=colors[:len(types)], startangle=90,
                                       textprops={'fontsize': 12})
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax.set_title('Source Type Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./charts/chart2.png', dpi=100)
    plt.close()

def generate_chart3_mcp_vs_other(metrics):
    """Chart 3: MCP-related vs Cross-vendor vs Other Papers"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    categories = ['MCP-Related', 'Cross-Vendor', 'Other']
    mcp_count = metrics["mcp_papers"]
    cross_vendor_count = metrics["cross_vendor_papers"]
    other_count = metrics["total_papers"] - mcp_count - cross_vendor_count
    # Note: some papers may be in both categories, so we need to handle overlap
    mcp_titles = [r["title"] for r in references if "MCP" in r["title"] or "Model Context Protocol" in r["title"]]
    cv_titles = [r["title"] for r in references if "Cross-Vendor" in r["title"] or "Cross-vendor" in r["title"] or "vendor" in r["title"].lower()]
    overlap = set(mcp_titles) & set(cv_titles)
    overlap_count = len(overlap)
    
    # Unique counts
    mcp_only = mcp_count - overlap_count
    cv_only = cross_vendor_count - overlap_count
    other = metrics["total_papers"] - mcp_only - cv_only - overlap_count
    
    counts = [mcp_only, cv_only, overlap_count, other]
    categories = ['MCP Only', 'Cross-Vendor Only', 'Both', 'Other']
    colors = ['#2E86C1', '#E74C3C', '#F39C12', '#95A5A6']
    
    bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Number of Papers', fontsize=14)
    ax.set_title('Paper Categorization: MCP & Cross-Vendor Focus', fontsize=16, fontweight='bold')
    
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(count), 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./charts/chart3.png', dpi=100)
    plt.close()

def generate_chart4_author_distribution(metrics):
    """Chart 4: Authors per Paper Distribution"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    auth_dist = metrics["author_distribution"]
    authors = sorted(auth_dist.keys())
    counts = [auth_dist[a] for a in authors]
    
    bars = ax.bar([str(a) for a in authors], counts, color='#27AE60', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Number of Authors', fontsize=14)
    ax.set_ylabel('Number of Papers', fontsize=14)
    ax.set_title('Distribution of Authors per Paper', fontsize=16, fontweight='bold')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(count), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./charts/chart4.png', dpi=100)
    plt.close()

def generate_chart5_top_venues(metrics):
    """Chart 5: Top Venues"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    venues = metrics["top_venues"]
    # Filter to top 8
    top_venues = dict(list(venues.items())[:8])
    venue_names = list(top_venues.keys())
    venue_counts = list(top_venues.values())
    
    # Truncate long venue names
    short_names = [v[:40] + '...' if len(v) > 40 else v for v in venue_names]
    
    bars = ax.barh(short_names, venue_counts, color='#8E44AD', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Number of Papers', fontsize=14)
    ax.set_title('Top Publication Venues', fontsize=16, fontweight='bold')
    
    for bar, count in zip(bars, venue_counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, str(count), 
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./charts/chart5.png', dpi=100)
    plt.close()

def main():
    print("Computing metrics from planned references...")
    metrics = compute_metrics(references)
    
    print(f"Total papers: {metrics['total_papers']}")
    print(f"Year distribution: {metrics['year_distribution']}")
    print(f"Source type distribution: {metrics['source_type_distribution']}")
    print(f"MCP papers: {metrics['mcp_papers']}")
    print(f"Cross-vendor papers: {metrics['cross_vendor_papers']}")
    print(f"Average authors per paper: {metrics['avg_authors']:.2f}")
    
    print("\nGenerating charts...")
    generate_chart1_year_distribution(metrics)
    print("  chart1.png - Year distribution")
    generate_chart2_source_types(metrics)
    print("  chart2.png - Source type distribution")
    generate_chart3_mcp_vs_other(metrics)
    print("  chart3.png - MCP vs Cross-vendor categorization")
    generate_chart4_author_distribution(metrics)
    print("  chart4.png - Author distribution")
    generate_chart5_top_venues(metrics)
    print("  chart5.png - Top venues")
    
    # Save metrics to results.json
    with open('./results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved metrics to results.json")
    print("Done!")

if __name__ == "__main__":
    main()
