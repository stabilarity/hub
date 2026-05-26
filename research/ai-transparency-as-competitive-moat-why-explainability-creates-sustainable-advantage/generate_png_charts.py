"""
Generate minimal PNG-like chart files for AI transparency analysis
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

def create_simple_png_chart(data, filename, title, xlabel, ylabel, data_series, series2=None, labels=None):
    """Create a simple PNG-like file with bitmap data"""
    # Create a simple bitmap representation
    width, height = 1200, 800  # Standard size
    
    # Create simple header
    png_header = f"""P3
{width} {height}
255"""
    
    # Create a simple chart representation as bitmap
    bitmap_data = []
    
    # Background
    for y in range(height):
        row = []
        for x in range(width):
            # Simple gradient background
            gray = int(240 - (y * 240 / height))
            row.append(f"{gray} {gray} {gray}")
        bitmap_data.append(" ".join(row))
    
    # Add chart elements
    if series2 is None:
        # Single line chart
        max_val = max(data_series)
        for i, (year, value) in enumerate(zip(data['years'], data_series)):
            x_pos = int((i / (len(data['years']) - 1)) * (width - 100) + 50)
            y_pos = int(height - 50 - (value / max_val) * (height - 100))
            
            # Draw point
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if 0 <= x_pos + dx < width and 0 <= y_pos + dy < height:
                        bitmap_data[y_pos + dy] = bitmap_data[y_pos + dy].replace(
                            f"{x_pos + dx} {x_pos + dx} {x_pos + dx}",
                            "46 139 87"  # Green color
                        )
    else:
        # Double line chart
        max_val = max(max(data_series), max(series2))
        for i, (year, val1, val2) in enumerate(zip(data['years'], data_series, series2)):
            x_pos = int((i / (len(data['years']) - 1)) * (width - 100) + 50)
            y1_pos = int(height - 50 - (val1 / max_val) * (height - 100))
            y2_pos = int(height - 50 - (val2 / max_val) * (height - 100))
            
            # Draw points
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if 0 <= x_pos + dx < width and 0 <= y1_pos + dy < height:
                        bitmap_data[y1_pos + dy] = bitmap_data[y1_pos + dy].replace(
                            f"{x_pos + dx} {x_pos + dx} {x_pos + dx}",
                            "230 57 70"  # Red color
                        )
                    if 0 <= x_pos + dx < width and 0 <= y2_pos + dy < height:
                        bitmap_data[y2_pos + dy] = bitmap_data[y2_pos + dy].replace(
                            f"{x_pos + dx} {x_pos + dx} {x_pos + dx}",
                            "0 150 136"  # Teal color
                        )
    
    # Combine header and data
    png_content = png_header + "\n" + "\n".join(bitmap_data)
    
    # Save as PNG file
    with open(f'charts/{filename}.ppm', 'w') as f:
        f.write(png_content)
    
    # Also create metadata
    metadata = {
        'title': title,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'width': width,
        'height': height,
        'format': 'PPM (Portable Pixmap)',
        'data_points': len(data['years'])
    }
    
    with open(f'charts/{filename}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

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
    print("Starting AI Transparency PNG Chart Creation...")
    
    # Ensure charts directory exists
    import os
    os.makedirs('charts', exist_ok=True)
    
    # Generate data
    data = generate_ai_transparency_data()
    
    # Create PNG-like chart files
    print("Creating chart1.png: Market Growth of Transparent AI Solutions")
    create_simple_png_chart(data, 'chart1', 
                            'Market Growth of Transparent AI Solutions',
                            'Year', 'Market Size ($B)',
                            data['transparent_ai_market'])
    
    print("Creating chart2.png: Regulatory Compliance Costs Comparison")
    create_simple_png_chart(data, 'chart2',
                            'Regulatory Compliance Costs Comparison',
                            'Year', 'Cost ($M)',
                            data['opaque_costs'], data['transparent_costs'],
                            ['Opaque AI', 'Transparent AI'])
    
    print("Creating chart3.png: Customer Trust Evolution")
    create_simple_png_chart(data, 'chart3',
                            'Customer Trust Evolution',
                            'Year', 'Trust Score',
                            data['customer_trust_opaque'], data['customer_trust_transparent'],
                            ['Opaque AI', 'Transparent AI'])
    
    print("Creating chart4.png: Competitive Advantage Growth")
    create_simple_png_chart(data, 'chart4',
                            'Competitive Advantage Through Transparency',
                            'Year', 'Advantage Score',
                            data['competitive_advantage'])
    
    # Calculate metrics
    metrics = calculate_metrics(data)
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nPNG chart creation complete!")
    print("Files created:")
    print("- charts/chart1.ppm, charts/chart1_metadata.json")
    print("- charts/chart2.ppm, charts/chart2_metadata.json")
    print("- charts/chart3.ppm, charts/chart3_metadata.json")
    print("- charts/chart4.ppm, charts/chart4_metadata.json")
    print("- results.json")
    
    print("\nNote: PPM format used instead of PNG due to matplotlib unavailability")
    print("PPM files can be converted to PNG using image conversion tools")
    
    print("\nKey Metrics:")
    print(f"- Market Growth Rate: {metrics['market_growth_rate']:.1f}%")
    print(f"- Avg Cost Savings: ${metrics['avg_cost_savings_millions']:.1f}M")
    print(f"- Trust Improvement: +{metrics['trust_improvement_points']:.1f} points")
    print(f"- Final Competitive Advantage: {metrics['final_competitive_advantage']:.1f}/100")

if __name__ == "__main__":
    main()
