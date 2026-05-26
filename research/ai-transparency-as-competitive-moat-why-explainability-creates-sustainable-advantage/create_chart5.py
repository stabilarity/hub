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
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    if 0 <= x_pos + dx < width and 0 <= y_pos + dy < height:
                        bitmap_data[y_pos + dy] = bitmap_data[y_pos + dy].replace(
                            f"{x_pos + dx} {x_pos + dx} {x_pos + dx}",
                            "255 140 0"  # Orange color for market potential
                        )
    
    # Combine header and data
    png_content = png_header + "\n" + "\n".join(bitmap_data)
    
    # Save as PPM file
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

def main():
    """Main analysis function"""
    print("Creating chart5.png: Market Potential Index...")
    
    # Ensure charts directory exists
    import os
    os.makedirs('charts', exist_ok=True)
    
    # Generate data
    data = generate_ai_transparency_data()
    
    # Create market potential chart
    market_potential = data['transparent_ai_market'] * (data['competitive_advantage'] / 100)
    create_simple_png_chart(data, 'chart5',
                            'AI Transparency Market Potential Index',
                            'Year', 'Market Potential Index',
                            market_potential)
    
    print("chart5.ppm created successfully!")

if __name__ == "__main__":
    main()
