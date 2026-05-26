"""
Create simple chart data files for AI transparency analysis
"""

import numpy as np
import json
from datetime import datetime

# Set random seed for reproducibility
np.random.default_rng(42)

def generate_ai_transparency_data():
    """Generate synthetic data for AI transparency analysis"""
    years = np.arange(2020, 2026)
    
    # Market size growth for transparent AI solutions
    transparent_ai_market = np.array([2.1, 3.5, 5.2, 7.8, 12.1, 18.5])  # Billions USD
    
    # Regulatory compliance costs for opaque vs transparent AI
    opaque_costs = np.array([0.8, 1.2, 1.9, 3.1, 4.8, 7.2])
    transparent_costs = np.array([0.3, 0.5, 0.7, 1.0, 1.4, 1.9])
    
    # Customer trust scores (0-100 scale)
    customer_trust_opaque = np.array([45, 42, 38, 35, 33, 30])
    customer_trust_transparent = np.array([55, 62, 70, 78, 85, 92])
    
    # Competitive advantage scores (0-100 scale)
    competitive_advantage = np.array([20, 35, 50, 68, 82, 95])
    
    return {
        'years': years,
        'transparent_ai_market': transparent_ai_market,
        'opaque_costs': opaque_costs,
        'transparent_costs': transparent_costs,
        'customer_trust_opaque': customer_trust_opaque,
        'customer_trust_transparent': customer_trust_transparent,
        'competitive_advantage': competitive_advantage
    }

def create_simple_chart_file(data, filename, title, xlabel, ylabel, data_series, series2=None, labels=None):
    """Create a simple chart file with ASCII art and data"""
    chart_data = {
        'title': title,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'data': []
    }
    
    for i, year in enumerate(data['years']):
        if series2 is not None and labels is not None:
            chart_data['data'].append({
                'year': int(year),
                labels[0]: float(data_series[i]),
                labels[1]: float(series2[i])
            })
        else:
            chart_data['data'].append({
                'year': int(year),
                'value': float(data_series[i])
            })
    
    # Create ASCII visualization
    ascii_chart = f"\n{title}\n{xlabel} vs {ylabel}\n" + "=" * 60 + "\n"
    
    if series2 is not None and labels is not None:
        max_val = max(max(data_series), max(series2))
        for i, year in enumerate(data['years']):
            val1 = data_series[i]
            val2 = series2[i]
            bar1 = "█" * int(val1 / max_val * 30)
            bar2 = "█" * int(val2 / max_val * 30)
            ascii_chart += f"{year}: {labels[0]} {bar1} {val1:.1f}  {labels[1]} {bar2} {val2:.1f}\n"
    else:
        max_val = max(data_series)
        for i, year in enumerate(data['years']):
            val = data_series[i]
            bar = "█" * int(val / max_val * 40)
            ascii_chart += f"{year}: {bar} {val:.1f}\n"
    
    ascii_chart += "=" * 60 + "\n"
    
    # Save both JSON and ASCII versions
    with open(f'charts/{filename}.json', 'w') as f:
        json.dump(chart_data, f, indent=2)
    
    with open(f'charts/{filename}.txt', 'w') as f:
        f.write(ascii_chart)
    
    return ascii_chart

def calculate_metrics(data):
    """Calculate key metrics for the analysis"""
    market_growth_rate = (data['transparent_ai_market'][-1] - data['transparent_ai_market'][0]) / data['transparent_ai_market'][0] * 100
    cost_savings = np.mean(data['opaque_costs'] - data['transparent_costs'])
    trust_improvement = data['customer_trust_transparent'][-1] - data['customer_trust_opaque'][-1]
    advantage_score = data['competitive_advantage'][-1]
    
    return {
        'market_growth_rate': float(market_growth_rate),
        'avg_cost_savings_millions': float(cost_savings),
        'trust_improvement_points': float(trust_improvement),
        'final_competitive_advantage': float(advantage_score),
        'analysis_timestamp': datetime.now().isoformat(),
        'years_analyzed': 2020,
        'years_analyzed_end': 2025
    }

def main():
    """Main analysis function"""
    print("Starting AI Transparency Chart Creation...")
    
    # Ensure charts directory exists
    import os
    os.makedirs('charts', exist_ok=True)
    
    # Generate data
    data = generate_ai_transparency_data()
    
    # Create chart files
    print("Creating chart1: Market Growth of Transparent AI Solutions")
    create_simple_chart_file(data, 'chart1', 
                            'Market Growth of Transparent AI Solutions',
                            'Year', 'Market Size ($B)',
                            data['transparent_ai_market'])
    
    print("Creating chart2: Regulatory Compliance Costs Comparison")
    create_simple_chart_file(data, 'chart2',
                            'Regulatory Compliance Costs Comparison',
                            'Year', 'Cost ($M)',
                            data['opaque_costs'], data['transparent_costs'],
                            ['Opaque AI', 'Transparent AI'])
    
    print("Creating chart3: Customer Trust Evolution")
    create_simple_chart_file(data, 'chart3',
                            'Customer Trust Evolution',
                            'Year', 'Trust Score',
                            data['customer_trust_opaque'], data['customer_trust_transparent'],
                            ['Opaque AI', 'Transparent AI'])
    
    print("Creating chart4: Competitive Advantage Growth")
    create_simple_chart_file(data, 'chart4',
                            'Competitive Advantage Through Transparency',
                            'Year', 'Advantage Score',
                            data['competitive_advantage'])
    
    # Create additional chart
    print("Creating chart5: Market Potential Index")
    market_potential = data['transparent_ai_market'] * (data['competitive_advantage'] / 100)
    create_simple_chart_file(data, 'chart5',
                            'AI Transparency Market Potential Index',
                            'Year', 'Market Potential Index',
                            market_potential)
    
    # Calculate metrics
    metrics = calculate_metrics(data)
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nChart creation complete!")
    print("Files created:")
    print("- charts/chart1.json, charts/chart1.txt")
    print("- charts/chart2.json, charts/chart2.txt")
    print("- charts/chart3.json, charts/chart3.txt")
    print("- charts/chart4.json, charts/chart4.txt")
    print("- charts/chart5.json, charts/chart5.txt")
    print("- results.json")
    
    print("\nKey Metrics:")
    print(f"- Market Growth Rate: {metrics['market_growth_rate']:.1f}%")
    print(f"- Avg Cost Savings: ${metrics['avg_cost_savings_millions']:.1f}M")
    print(f"- Trust Improvement: +{metrics['trust_improvement_points']:.1f} points")
    print(f"- Final Competitive Advantage: {metrics['final_competitive_advantage']:.1f}/100")

if __name__ == "__main__":
    main()
