"""
Shadow Real Estate Markets: Using Satellite and Utility Data to Estimate Undeclared Property Value
Reproduce: python3 analysis.py

This analysis extends satellite-based informal economy measurement to property valuation,
cross-referencing utility usage as a proxy for occupancy to estimate undeclared property value.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set seed for reproducibility
rng = np.random.default_rng(42)

# Create output directories
charts_dir = Path("./charts")
charts_dir.mkdir(exist_ok=True)

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

# Simulate 100 neighborhoods with satellite and utility data
n_neighborhoods = 100

# Satellite-derived features (nighttime lights, building density, roof area)
satellite_lights = rng.lognormal(mean=2.5, sigma=0.8, size=n_neighborhoods)
building_density = rng.beta(a=2, b=5, size=n_neighborhoods) * 100  # buildings per hectare
roof_area = rng.gamma(shape=2, scale=500, size=n_neighborhoods)  # sq meters

# Utility usage as proxy for occupancy
electricity_per_capita = rng.lognormal(mean=4.0, sigma=0.5, size=n_neighborhoods)
water_consumption = rng.lognormal(mean=3.5, sigma=0.6, size=n_neighborhoods)

# Declared property values (official records)
declared_value = rng.lognormal(mean=12.0, sigma=0.7, size=n_neighborhoods)  # USD

# True property value (latent) - higher where satellite/utility indicate more activity
# True value correlates with satellite lights, building density, and utility usage
true_value = (declared_value * 
              (1 + 0.3 * (satellite_lights / satellite_lights.max()) + 
               0.2 * (building_density / building_density.max()) + 
               0.25 * (electricity_per_capita / electricity_per_capita.max()) +
               0.15 * (water_consumption / water_consumption.max())) *
              rng.lognormal(mean=0, sigma=0.15, size=n_neighborhoods))

# Undeclared value = true value - declared value
undeclared_value = true_value - declared_value
undeclared_pct = (undeclared_value / true_value) * 100

# Model predictions using satellite + utility features
# Simple linear model for demonstration
X = np.column_stack([
    satellite_lights,
    building_density,
    roof_area,
    electricity_per_capita,
    water_consumption
])
X = (X - X.mean(axis=0)) / X.std(axis=0)  # Standardize

# True coefficients for simulation
true_coeffs = np.array([0.35, 0.20, 0.15, 0.25, 0.15])
log_true_value = np.log(true_value)
predicted_log_value = X @ true_coeffs + rng.normal(0, 0.1, n_neighborhoods)
predicted_value = np.exp(predicted_log_value)

# Model residuals
residuals = true_value - predicted_value
mape = np.mean(np.abs(residuals / true_value)) * 100
rmse = np.sqrt(np.mean(residuals**2))

# ============================================================
# CHART 1: Satellite vs Utility Indicators vs Undeclared Value
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Nighttime lights vs undeclared %
sc = axes[0].scatter(satellite_lights, undeclared_pct, c=building_density, 
                      cmap='viridis', alpha=0.7, s=50)
axes[0].set_xlabel('Nighttime Lights Index')
axes[0].set_ylabel('Undeclared Value (%)')
axes[0].set_title('Satellite Nighttime Lights vs Undeclared Property Value')
plt.colorbar(sc, ax=axes[0], label='Building Density')

# Electricity usage vs undeclared %
sc = axes[1].scatter(electricity_per_capita, undeclared_pct, c=water_consumption,
                      cmap='plasma', alpha=0.7, s=50)
axes[1].set_xlabel('Electricity per Capita (kWh)')
axes[1].set_ylabel('Undeclared Value (%)')
axes[1].set_title('Utility Usage (Electricity) vs Undeclared Value')
plt.colorbar(sc, ax=axes[1], label='Water Consumption')

# Building density vs undeclared %
sc = axes[2].scatter(building_density, undeclared_pct, c=roof_area,
                      cmap='inferno', alpha=0.7, s=50)
axes[2].set_xlabel('Building Density (buildings/hectare)')
axes[2].set_ylabel('Undeclared Value (%)')
axes[2].set_title('Building Density vs Undeclared Property Value')
plt.colorbar(sc, ax=axes[2], label='Roof Area (sq m)')

# Water vs Electricity (utility correlation)
sc = axes[3].scatter(water_consumption, electricity_per_capita, c=undeclared_pct,
                      cmap='coolwarm', alpha=0.7, s=50)
axes[3].set_xlabel('Water Consumption (m³)')
axes[3].set_ylabel('Electricity per Capita (kWh)')
axes[3].set_title('Water vs Electricity Consumption (colored by Undeclared %)')
plt.colorbar(sc, ax=axes[3], label='Undeclared Value (%)')

plt.tight_layout()
plt.savefig(charts_dir / "chart1.png", dpi=100, bbox_inches='tight')
plt.close()

# ============================================================
# CHART 2: Declared vs True vs Predicted Property Values
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 8))

# Declared vs True
axes[0].scatter(declared_value, true_value, alpha=0.6, s=50, c=undeclared_pct, cmap='RdYlGn_r')
axes[0].plot([declared_value.min(), declared_value.max()], 
             [declared_value.min(), declared_value.max()], 'k--', alpha=0.5, label='y=x')
axes[0].set_xlabel('Declared Value (USD)')
axes[0].set_ylabel('True Value (USD)')
axes[0].set_title('Declared vs True Property Value')
axes[0].legend()
plt.colorbar(axes[0].collections[0], ax=axes[0], label='Undeclared %')

# Predicted vs True
axes[1].scatter(predicted_value, true_value, alpha=0.6, s=50, c='steelblue')
axes[1].plot([true_value.min(), true_value.max()], 
             [true_value.min(), true_value.max()], 'k--', alpha=0.5, label='y=x')
axes[1].set_xlabel('Predicted Value (USD)')
axes[1].set_ylabel('True Value (USD)')
axes[1].set_title(f'Satellite+Utility Model: Predicted vs True\nMAPE={mape:.1f}%, RMSE={rmse:,.0f}')
axes[1].legend()

# Residuals
axes[2].scatter(true_value, residuals, alpha=0.6, s=50, c='coral')
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[2].set_xlabel('True Value (USD)')
axes[2].set_ylabel('Residual (True - Predicted)')
axes[2].set_title('Model Residuals')

plt.tight_layout()
plt.savefig(charts_dir / "chart2.png", dpi=100, bbox_inches='tight')
plt.close()

# ============================================================
# CHART 3: Feature Importance and Undeclared Value Distribution
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 8))

# Feature importance (from true coefficients)
features = ['Nighttime Lights', 'Building Density', 'Roof Area', 'Electricity', 'Water']
importance = true_coeffs / true_coeffs.sum() * 100
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
bars = axes[0].barh(features, importance, color=colors)
axes[0].set_xlabel('Relative Importance (%)')
axes[0].set_title('Feature Importance for Property Valuation\n(Satellite + Utility Features)')
for bar, val in zip(bars, importance):
    axes[0].text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center')

# Undeclared value distribution
axes[1].hist(undeclared_pct, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[1].axvline(undeclared_pct.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {undeclared_pct.mean():.1f}%')
axes[1].axvline(np.median(undeclared_pct), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(undeclared_pct):.1f}%')
axes[1].set_xlabel('Undeclared Value (%)')
axes[1].set_ylabel('Number of Neighborhoods')
axes[1].set_title('Distribution of Undeclared Property Value')
axes[1].legend()

# Cumulative undeclared value by neighborhood rank
sorted_idx = np.argsort(undeclared_value)[::-1]
cumsum = np.cumsum(undeclared_value[sorted_idx]) / undeclared_value.sum() * 100
axes[2].plot(range(1, n_neighborhoods + 1), cumsum, 'b-', linewidth=2)
axes[2].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
axes[2].axvline(x=np.where(cumsum >= 80)[0][0] + 1, color='red', linestyle='--', alpha=0.7)
axes[2].set_xlabel('Neighborhood Rank (by undeclared value)')
axes[2].set_ylabel('Cumulative Share of Undeclared Value (%)')
axes[2].set_title('Concentration of Undeclared Value\n(Top neighborhoods capture majority)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(charts_dir / "chart3.png", dpi=100, bbox_inches='tight')
plt.close()

# ============================================================
# CHART 4: Regional Aggregation (bonus chart)
# ============================================================
# Simulate regional grouping
regions = ['North', 'South', 'East', 'West', 'Central']
region_labels = rng.choice(regions, size=n_neighborhoods, p=[0.25, 0.2, 0.2, 0.2, 0.15])

fig, axes = plt.subplots(1, 2, figsize=(12, 8))

# Box plot by region
region_data = [undeclared_pct[region_labels == r] for r in regions]
bp = axes[0].boxplot(region_data, labels=regions, patch_artist=True)
for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(regions)))):
    patch.set_facecolor(color)
axes[0].set_ylabel('Undeclared Value (%)')
axes[0].set_title('Undeclared Property Value by Region')

# Regional summary bar chart
region_means = [d.mean() for d in region_data]
region_totals = [undeclared_value[region_labels == r].sum() / 1e9 for r in regions]
ax2 = axes[1].twinx()
bars1 = axes[1].bar(regions, region_means, alpha=0.7, color='steelblue', label='Mean Undeclared %')
bars2 = ax2.bar([r + 0.4 for r in range(len(regions))], region_totals, width=0.4, 
                 alpha=0.7, color='coral', label='Total Undeclared (B USD)')
axes[1].set_ylabel('Mean Undeclared %', color='steelblue')
ax2.set_ylabel('Total Undeclared (Billion USD)', color='coral')
axes[1].set_title('Regional Summary: Mean % and Total Undeclared Value')
# Combined legend
lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(charts_dir / "chart4.png", dpi=100, bbox_inches='tight')
plt.close()

# ============================================================
# CHART 5: Model Performance by Neighborhood Characteristics
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Absolute error vs building density
abs_error = np.abs(residuals)
sc = axes[0].scatter(building_density, abs_error, c=satellite_lights, cmap='viridis', alpha=0.7, s=50)
axes[0].set_xlabel('Building Density')
axes[0].set_ylabel('Absolute Prediction Error (USD)')
axes[0].set_title('Prediction Error vs Building Density')
plt.colorbar(sc, ax=axes[0], label='Nighttime Lights')

# Absolute error vs electricity
sc = axes[1].scatter(electricity_per_capita, abs_error, c=water_consumption, cmap='plasma', alpha=0.7, s=50)
axes[1].set_xlabel('Electricity per Capita')
axes[1].set_ylabel('Absolute Prediction Error (USD)')
axes[1].set_title('Prediction Error vs Electricity Usage')
plt.colorbar(sc, ax=axes[1], label='Water Consumption')

# Error distribution by declared value quantiles
quantiles = pd.qcut(declared_value, q=5, labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'])
error_by_quantile = [abs_error[quantiles == q] for q in ['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)']]
bp = axes[2].boxplot(error_by_quantile, labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.7)
axes[2].set_ylabel('Absolute Error (USD)')
axes[2].set_title('Prediction Error by Declared Value Quintile')

# R² by region
from sklearn.metrics import r2_score
region_r2 = []
for r in regions:
    mask = region_labels == r
    if mask.sum() > 5:
        r2 = r2_score(true_value[mask], predicted_value[mask])
        region_r2.append(r2)
    else:
        region_r2.append(0)
axes[3].bar(regions, region_r2, color=plt.cm.Set3(np.linspace(0, 1, len(regions))), alpha=0.8)
axes[3].axhline(y=np.mean(region_r2), color='red', linestyle='--', label=f'Mean R²: {np.mean(region_r2):.3f}')
axes[3].set_ylabel('R² Score')
axes[3].set_title('Model R² by Region')
axes[3].legend()
axes[3].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(charts_dir / "chart5.png", dpi=100, bbox_inches='tight')
plt.close()

# ============================================================
# COMPUTE AND SAVE METRICS
# ============================================================
metrics = {
    "n_neighborhoods": int(n_neighborhoods),
    "mean_undeclared_pct": float(undeclared_pct.mean()),
    "median_undeclared_pct": float(np.median(undeclared_pct)),
    "total_declared_value_usd": float(declared_value.sum()),
    "total_true_value_usd": float(true_value.sum()),
    "total_undeclared_value_usd": float(undeclared_value.sum()),
    "undeclared_share_of_true_pct": float(undeclared_value.sum() / true_value.sum() * 100),
    "model_mape_pct": float(mape),
    "model_rmse_usd": float(rmse),
    "model_r2_overall": float(r2_score(true_value, predicted_value)),
    "feature_importance": dict(zip(features, (true_coeffs / true_coeffs.sum() * 100).tolist())),
    "region_summary": {
        r: {
            "mean_undeclared_pct": float(undeclared_pct[region_labels == r].mean()),
            "total_undeclared_usd": float(undeclared_value[region_labels == r].sum()),
            "n_neighborhoods": int((region_labels == r).sum())
        } for r in regions
    }
}

with open("./results.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Analysis complete. Charts saved to ./charts/")
print(f"Results saved to ./results.json")
print(f"Mean undeclared value: {metrics['mean_undeclared_pct']:.1f}%")
print(f"Model MAPE: {metrics['model_mape_pct']:.1f}%")
print(f"Model R²: {metrics['model_r2_overall']:.3f}")
