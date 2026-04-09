"""
Generate charts for Georgia Tax Reform article
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from georgia_data import shadow_economy_georgia, tax_revenue_gdp, regional_comparison, pre_post_reform
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

def chart_1_shadow_economy_timeline():
    """Chart 1: Shadow Economy Timeline with Reform Markers"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    years = list(shadow_economy_georgia.keys())
    values = list(shadow_economy_georgia.values())
    
    ax.fill_between(years, values, alpha=0.3, color='#2E3440')
    ax.plot(years, values, linewidth=2.5, color='#2E3440', marker='o', markersize=3)
    
    # Reform markers
    ax.axvline(x=2005, color='#BF616A', linestyle='--', alpha=0.8, linewidth=2)
    ax.annotate('Flat Tax Reform\n(12% introduced)', xy=(2005, 58), xytext=(2007, 62),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='#BF616A', lw=1.5))
    
    ax.axvline(x=2008, color='#D08770', linestyle='--', alpha=0.8, linewidth=1.5)
    ax.annotate('Tax raised to 15%', xy=(2008, 52), xytext=(2010, 56),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='#D08770', lw=1))
    
    ax.axvline(x=2020, color='#5E81AC', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.annotate('COVID-19', xy=(2020, 46), xytext=(2018, 50),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='#5E81AC', lw=1))
    
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Shadow Economy (% of GDP)', fontsize=11)
    ax.set_title('Georgia: Shadow Economy Size (2000-2025)\nImpact of Flat Tax Reform', fontsize=13, fontweight='bold')
    ax.set_xlim(1999, 2026)
    ax.set_ylim(35, 72)
    ax.grid(True, alpha=0.3)
    
    # Add reduction annotation
    ax.annotate('', xy=(2024, 41.5), xytext=(2000, 68.8),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(2012, 62, '40% reduction\n(27.3 pp)', fontsize=10, ha='center', color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/root/hub/research/shadow-economy-dynamics/charts/chart1_shadow_economy_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Chart 1 saved: Shadow Economy Timeline")

def chart_2_tax_revenue_comparison():
    """Chart 2: Tax Revenue Pre/Post Reform"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = list(pre_post_reform.keys())
    shadow_vals = [pre_post_reform[c]["shadow_economy_avg"] for c in categories]
    tax_vals = [pre_post_reform[c]["tax_revenue_avg"] for c in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, shadow_vals, width, label='Shadow Economy (% GDP)', color='#BF616A', alpha=0.8)
    bars2 = ax.bar(x + width/2, tax_vals, width, label='Tax Revenue (% GDP)', color='#5E81AC', alpha=0.8)
    
    ax.set_ylabel('Percentage of GDP', fontsize=11)
    ax.set_title('Pre vs Post-Reform: Shadow Economy & Tax Revenue\n(Average Values)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 75)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    
    # Add improvement arrows
    ax.annotate('', xy=(1, 48.5), xytext=(0, 65.7),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(0.5, 58, '-17.2 pp', fontsize=10, ha='center', color='green', fontweight='bold')
    
    ax.annotate('', xy=(1, 22.9), xytext=(0, 14.8),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(0.5, 17, '+8.1 pp', fontsize=10, ha='center', color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/root/hub/research/shadow-economy-dynamics/charts/chart2_pre_post_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Chart 2 saved: Pre/Post Reform Comparison")

def chart_3_regional_comparison():
    """Chart 3: Regional Shadow Economy Comparison"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    countries = list(regional_comparison.keys())
    values = list(regional_comparison.values())
    
    colors = ['#5E81AC' if c == 'Georgia' else '#D8DEE9' for c in countries]
    
    bars = ax.barh(countries, values, color=colors, edgecolor='#2E3440', linewidth=0.5)
    
    # Highlight Georgia
    bars[countries.index('Georgia')].set_color('#5E81AC')
    bars[countries.index('Georgia')].set_edgecolor('#2E3440')
    bars[countries.index('Georgia')].set_linewidth(2)
    
    ax.set_xlabel('Shadow Economy (% of GDP, 2024)', fontsize=11)
    ax.set_title('Shadow Economy: Regional Comparison (2024)\nGeorgia vs Post-Soviet & EU States', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 60)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.8, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                va='center', fontsize=9)
    
    # Add EU average line
    ax.axvline(x=18.5, color='#BF616A', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(19, len(countries)-0.5, 'EU Average', fontsize=9, color='#BF616A')
    
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/hub/research/shadow-economy-dynamics/charts/chart3_regional_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Chart 3 saved: Regional Comparison")

def chart_4_dual_axis_timeline():
    """Chart 4: Dual axis showing Shadow Economy vs Tax Revenue over time"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    years = list(shadow_economy_georgia.keys())
    shadow_vals = list(shadow_economy_georgia.values())
    tax_vals = [tax_revenue_gdp[y] for y in years]
    
    color1 = '#BF616A'
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Shadow Economy (% of GDP)', color=color1, fontsize=11)
    line1 = ax1.plot(years, shadow_vals, color=color1, linewidth=2.5, marker='o', markersize=3, label='Shadow Economy')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(35, 72)
    
    ax2 = ax1.twinx()
    color2 = '#5E81AC'
    ax2.set_ylabel('Tax Revenue (% of GDP)', color=color2, fontsize=11)
    line2 = ax2.plot(years, tax_vals, color=color2, linewidth=2.5, marker='s', markersize=3, label='Tax Revenue')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(10, 30)
    
    # Reform marker
    ax1.axvline(x=2005, color='#2E3440', linestyle='--', alpha=0.7, linewidth=2)
    ax1.annotate('Flat Tax Reform', xy=(2005, 60), xytext=(2008, 65),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='#2E3440', lw=1.5))
    
    # Title and legend
    ax1.set_title('Georgia: Shadow Economy vs Tax Revenue (2000-2024)\nInverse Correlation Post-Reform', fontsize=13, fontweight='bold')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/hub/research/shadow-economy-dynamics/charts/chart4_dual_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Chart 4 saved: Dual Axis Correlation")

if __name__ == "__main__":
    chart_1_shadow_economy_timeline()
    chart_2_tax_revenue_comparison()
    chart_3_regional_comparison()
    chart_4_dual_axis_timeline()
    print("\nAll charts generated successfully!")
