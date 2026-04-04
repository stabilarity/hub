"""
Real-Time DRI Monitoring: Continuous Decision Readiness Assessment
HPF-P Framework Research Series - Article #13
Charts based on synthetic simulation data representing typical pharmaceutical portfolio patterns
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from datetime import datetime, timedelta

# ---- Chart 1: DRI Signal Drift Over Time (time-series monitoring) ----
np.random.seed(42)
days = 90
time_points = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(days)]
dates_str = [d.strftime('%b %d') for d in time_points]

# Simulate DRI trajectory for 3 portfolio assets
def dri_trajectory(base, noise_scale, drift_events):
    vals = []
    v = base
    for i in range(days):
        v += np.random.normal(0, noise_scale)
        for day, delta in drift_events:
            if i == day:
                v += delta
        v = max(0.1, min(1.0, v))
        vals.append(v)
    return vals

asset_a = dri_trajectory(0.72, 0.015, [(25, -0.12), (60, +0.08)])  # mild drop, recovery
asset_b = dri_trajectory(0.55, 0.020, [(40, -0.20), (55, -0.10)])  # two drops → critical
asset_c = dri_trajectory(0.81, 0.010, [(70, +0.06)])              # stable then improvement

threshold_yellow = 0.60
threshold_red    = 0.45

fig, ax = plt.subplots(figsize=(12, 5))
tick_indices = list(range(0, days, 10))
ax.fill_between(range(days), 0, threshold_red, alpha=0.08, color='red', label='Critical zone (<0.45)')
ax.fill_between(range(days), threshold_red, threshold_yellow, alpha=0.06, color='orange', label='Warning zone (0.45–0.60)')
ax.plot(asset_a, color='#1a73e8', linewidth=2, label='Asset A — Moderate volatility')
ax.plot(asset_b, color='#e53935', linewidth=2, label='Asset B — Progressive decline')
ax.plot(asset_c, color='#2e7d32', linewidth=2, label='Asset C — High stability')
ax.axhline(threshold_yellow, color='orange', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(threshold_red, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xticks(tick_indices)
ax.set_xticklabels([dates_str[i] for i in tick_indices], rotation=30, ha='right', fontsize=9)
ax.set_ylabel('DRI Score', fontsize=11)
ax.set_xlabel('Observation Window (Q1 2026)', fontsize=11)
ax.set_title('DRI Signal Drift Detection: 3-Asset Pharmaceutical Portfolio (90-Day Window)', fontsize=12, fontweight='bold')
ax.set_ylim(0.1, 1.05)
ax.legend(loc='lower left', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/root/hub/research/hpfp-realtime-dri-monitoring/charts/dri_drift_monitoring.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1 saved.")

# ---- Chart 2: Alert Latency vs Detection Accuracy across monitoring frequencies ----
frequencies = ['Every 6h', 'Daily', 'Weekly', 'Bi-weekly', 'Monthly']
latency_hrs  = [6, 24, 168, 336, 720]          # avg hours until alert triggered
accuracy_pct = [94.1, 91.5, 83.2, 74.6, 61.3] # % of true drift events detected

fig, ax1 = plt.subplots(figsize=(9, 5))
x = np.arange(len(frequencies))
bars = ax1.bar(x, accuracy_pct, color=['#2e7d32','#1a73e8','#f9a825','#e65100','#c62828'], alpha=0.85, width=0.5)
ax1.set_ylabel('Detection Accuracy (%)', fontsize=11)
ax1.set_ylim(50, 100)
ax1.set_xticks(x)
ax1.set_xticklabels(frequencies, fontsize=10)
ax1.set_xlabel('Monitoring Frequency', fontsize=11)

ax2 = ax1.twinx()
ax2.plot(x, latency_hrs, marker='o', color='black', linewidth=2, markersize=7, label='Alert Latency (h)')
ax2.set_ylabel('Alert Latency (hours, log scale)', fontsize=11)
ax2.set_yscale('log')

for bar, acc in zip(bars, accuracy_pct):
    ax1.text(bar.get_x() + bar.get_width()/2, acc + 0.5, f'{acc}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_title('Monitoring Frequency vs. Detection Accuracy and Alert Latency', fontsize=12, fontweight='bold')
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend([mpatches.Patch(color='gray', alpha=0.6, label='Detection Accuracy')], ['Detection Accuracy'], loc='upper right', fontsize=9)
ax2.legend(loc='lower right', fontsize=9)
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/root/hub/research/hpfp-realtime-dri-monitoring/charts/monitoring_frequency_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2 saved.")

# ---- Chart 3: DRI Component Breakdown (Radar) ----
categories = ['Data\nCompleteness', 'Source\nTimeliness', 'Signal\nConsistency', 'Expert\nConfidence', 'Regulatory\nAlignment']
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

assets_radar = {
    'Asset A (DRI=0.71)': [0.82, 0.65, 0.74, 0.68, 0.72],
    'Asset B (DRI=0.38)': [0.45, 0.30, 0.42, 0.35, 0.28],
    'Asset C (DRI=0.86)': [0.90, 0.88, 0.85, 0.82, 0.84],
}
colors_radar = ['#1a73e8', '#e53935', '#2e7d32']

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for (label, vals), color in zip(assets_radar.items(), colors_radar):
    vals_c = vals + vals[:1]
    ax.plot(angles, vals_c, 'o-', linewidth=2, color=color, label=label)
    ax.fill(angles, vals_c, alpha=0.10, color=color)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=8)
ax.axhline(0.60, color='orange', linestyle='--', linewidth=1, alpha=0.6)
ax.set_title('DRI Component Profile: 3 Portfolio Assets', fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig('/root/hub/research/hpfp-realtime-dri-monitoring/charts/dri_component_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3 saved.")

# ---- Chart 4: Response Time Distribution (intervention speed after alert) ----
np.random.seed(7)
automated_resp = np.random.exponential(scale=1.5, size=200).clip(0.1, 10)
manual_resp    = np.random.exponential(scale=8.0, size=200).clip(0.5, 48)

fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, 48, 30)
ax.hist(automated_resp, bins=bins, alpha=0.7, color='#2e7d32', label=f'Automated response (median={np.median(automated_resp):.1f}h)')
ax.hist(manual_resp,    bins=bins, alpha=0.7, color='#1a73e8', label=f'Manual response (median={np.median(manual_resp):.1f}h)')
ax.axvline(np.median(automated_resp), color='#2e7d32', linestyle='--', linewidth=2)
ax.axvline(np.median(manual_resp),    color='#1a73e8', linestyle='--', linewidth=2)
ax.set_xlabel('Response Time (hours after alert)', fontsize=11)
ax.set_ylabel('Frequency (portfolio events)', fontsize=11)
ax.set_title('Automated vs Manual DRI Alert Response Time Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/root/hub/research/hpfp-realtime-dri-monitoring/charts/alert_response_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 4 saved.")

print("All 4 charts generated.")
