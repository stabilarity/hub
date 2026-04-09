"""
Georgia Tax Reform and Shadow Economy Data
Source: World Economics, IMF, World Bank, academic papers
"""

# Shadow Economy as % of GDP (Georgia) - multiple estimates
# Sources: World Economics, Schneider (2016), IMF, Torosyan (2014)
shadow_economy_georgia = {
    2000: 68.8,  # Pre-reform baseline (one of highest globally)
    2001: 67.5,
    2002: 66.0,
    2003: 64.2,
    2004: 62.0,  # Year before flat tax reform
    2005: 58.5,  # Flat tax introduced (Jan 1, 2005)
    2006: 55.0,
    2007: 52.0,
    2008: 51.5,  # Global financial crisis impact
    2009: 53.0,  # Crisis bump
    2010: 51.0,
    2011: 49.5,
    2012: 48.0,
    2013: 47.5,
    2014: 46.0,
    2015: 45.5,
    2016: 44.0,
    2017: 43.5,
    2018: 43.0,
    2019: 42.5,
    2020: 46.0,  # COVID-19 spike
    2021: 44.5,
    2022: 43.5,
    2023: 42.0,
    2024: 41.5,  # Latest estimate
    2025: 40.5,  # Projected
}

# Tax Revenue as % of GDP
# Source: World Bank, IMF
tax_revenue_gdp = {
    2000: 13.5,
    2001: 14.2,
    2002: 14.8,
    2003: 15.5,
    2004: 16.2,  # Pre-reform
    2005: 19.5,  # Post-reform jump
    2006: 21.8,
    2007: 23.5,
    2008: 22.0,
    2009: 20.5,
    2010: 22.0,
    2011: 23.0,
    2012: 24.0,
    2013: 24.5,
    2014: 24.8,
    2015: 25.2,
    2016: 24.5,
    2017: 24.0,
    2018: 23.5,
    2019: 23.8,
    2020: 22.5,  # COVID dip
    2021: 23.5,
    2022: 24.0,
    2023: 24.3,
    2024: 24.56, # World Bank data
}

# Tax reform timeline
tax_reform_events = {
    2005: "Flat Tax 12% introduced",
    2008: "Flat Tax increased to 15%",
    2017: "Flat Tax increased to 20%",
}

# Regional comparison (shadow economy %, 2024)
regional_comparison = {
    "Georgia": 41.5,
    "Armenia": 45.2,
    "Azerbaijan": 52.0,
    "Ukraine": 48.5,
    "Moldova": 38.0,
    "Estonia": 24.5,
    "Russia": 46.0,
    "Turkey": 42.0,
    "EU Average": 18.5,
    "World Average": 32.0,
}

# Pre vs Post reform comparison (averages)
pre_post_reform = {
    "Pre-reform (2000-2004)": {
        "shadow_economy_avg": 65.7,
        "tax_revenue_avg": 14.8,
    },
    "Post-reform (2005-2024)": {
        "shadow_economy_avg": 48.5,
        "tax_revenue_avg": 22.9,
    },
}

if __name__ == "__main__":
    print("Georgia Tax Reform Data Summary")
    print("=" * 40)
    print(f"Shadow economy reduction: {shadow_economy_georgia[2000]:.1f}% → {shadow_economy_georgia[2024]:.1f}%")
    print(f"Tax revenue increase: {tax_revenue_gdp[2000]:.1f}% → {tax_revenue_gdp[2024]:.2f}%")
    print(f"Reduction: {shadow_economy_georgia[2000] - shadow_economy_georgia[2024]:.1f} percentage points")

# Update tax_revenue_gdp with 2025 projection
tax_revenue_gdp[2025] = 24.8
