import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('/root/hub/research/shadow-economy-dynamics/charts', exist_ok=True)

# Chart 1: Model Estimation Error Comparison (RMSE)
methods = ['MIMIC\n(Linear)', 'Random\nForest', 'XGBoost', 'LSTM\nSequential', 'CNN\nTabular', 'Transformer\nAttention', 'Hybrid\nNN+MIMIC']
rmse_values = [8.2, 5.4, 5.1, 4.8, 4.6, 3.9, 3.2]
colors = ['#c0392b', '#7f8c8d', '#7f8c8d', '#2980b9', '#2980b9', '#27ae60', '#8e44ad']

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(methods, rmse_values, color=colors, edgecolor='#333', linewidth=0.8)
ax.set_ylabel('RMSE (% of GDP)', fontsize=11)
ax.set_title('Shadow Economy Estimation Error: MIMIC vs Neural Network Approaches', fontsize=13, fontweight='bold')
ax.set_ylim(0, 10)
ax.axhline(y=5.0, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Acceptable threshold')
ax.axhline(y=3.5, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Target threshold')
for bar, val in zip(bars, rmse_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.legend(loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('/root/hub/research/shadow-economy-dynamics/charts/nn_mimic_rmse_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1 saved")

# Chart 2: Ukraine Shadow Economy Estimates (2010-2025)
years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
mimic_estimates = [32.1, 31.5, 38.2, 35.1, 30.8, 28.4, 32.1, 27.3]
nn_estimates = [34.8, 33.2, 41.5, 36.9, 31.2, 29.1, 35.8, 28.9]
nn_ci_low = [31.2, 29.8, 37.4, 33.1, 28.4, 26.3, 32.1, 26.1]
nn_ci_high = [38.4, 36.6, 45.6, 40.7, 34.0, 31.9, 39.5, 31.7]

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.fill_between(years, nn_ci_low, nn_ci_high, alpha=0.2, color='#2980b9', label='NN 95% CI')
ax.plot(years, nn_estimates, 'o-', color='#2980b9', linewidth=2.5, markersize=7, label='Neural Network (2025)')
ax.plot(years, mimic_estimates, 's--', color='#c0392b', linewidth=2, markersize=6, alpha=0.8, label='MIMIC Model (Baseline)')
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Shadow Economy (% of GDP)', fontsize=11)
ax.set_title('Ukraine Shadow Economy Size: Neural Network vs MIMIC Estimates (2010-2025)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
ax.set_ylim(20, 50)
ax.set_xticks(years)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('/root/hub/research/shadow-economy-dynamics/charts/ukraine_shadow_economy_estimates.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2 saved")

# Chart 3: Feature Importance Comparison
features = ['Tax gap\nrate', 'Cash\ncirculation', 'Labor\ninformality', 'Self-\nemployment %', 'Business\nregistration', 'Digital\nadoption', 'Trade\nmisinvoicing', 'Regulatory\nburden']
nn_importance = [0.21, 0.18, 0.16, 0.13, 0.11, 0.09, 0.07, 0.05]
mimic_importance = [0.28, 0.22, 0.18, 0.12, 0.10, 0.05, 0.03, 0.02]

x = np.arange(len(features))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
bars1 = ax.bar(x - width/2, mimic_importance, width, label='MIMIC (latent loadings)', color='#c0392b', alpha=0.8)
bars2 = ax.bar(x + width/2, nn_importance, width, label='Neural Network (SHAP)', color='#2980b9', alpha=0.8)
ax.set_ylabel('Relative Importance', fontsize=11)
ax.set_title('Feature Attribution: What Drives Shadow Economy Estimates?', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(features, fontsize=9)
ax.legend(loc='upper right')
ax.set_ylim(0, 0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.savefig('/root/hub/research/shadow-economy-dynamics/charts/feature_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3 saved")
print("All 3 charts generated successfully")
