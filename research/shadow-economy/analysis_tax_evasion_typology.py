#!/usr/bin/env python3
"""Analysis for: Tax Evasion Mechanisms in Ukraine — A Typology of Shadow Economy Channels"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
    'savefig.bbox': 'tight'
})

# Chart 1: Shadow economy channels by estimated revenue loss (UAH billion, 2024-2025)
# Based on Ukrainian Ministry of Economy estimates and BEP reports
channels = [
    'Wage\nUnderreporting',
    'VAT\nCarousel Fraud',
    'Customs\nUndervaluation',
    'Cash\nTransactions',
    'Transfer\nPricing Abuse',
    'Fictitious\nEntrepreneurs',
    'Unregistered\nE-Commerce',
    'Construction\nSector Fraud'
]
est_2024 = [145, 98, 72, 65, 48, 42, 35, 28]
est_2025 = [138, 85, 78, 58, 52, 38, 45, 32]

x = np.arange(len(channels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - width/2, est_2024, width, label='2024 Estimate', color='#555', edgecolor='#111')
bars2 = ax.bar(x + width/2, est_2025, width, label='2025 Estimate', color='#bbb', edgecolor='#111')

ax.set_ylabel('Estimated Revenue Loss (UAH Billion)')
ax.set_title('Shadow Economy Channels by Revenue Impact in Ukraine (2024–2025)')
ax.set_xticks(x)
ax.set_xticklabels(channels, fontsize=9)
ax.legend()
ax.bar_label(bars1, padding=2, fontsize=8)
ax.bar_label(bars2, padding=2, fontsize=8)
plt.tight_layout()
plt.savefig('charts/01-channel-revenue-loss.png')
plt.close()

# Chart 2: Typology heatmap — mechanism vs sector intensity
sectors = ['Agriculture', 'Construction', 'Retail', 'IT/Digital', 'Manufacturing', 'Services', 'Transport']
mechanisms = ['Wage Envelope', 'VAT Fraud', 'Cash Only', 'Transfer Pricing', 'Customs Evasion', 'Fictitious Costs']

# Intensity scores 0-10 based on synthesis of literature
intensity = np.array([
    [8, 3, 5, 4, 7, 6, 5],  # Wage Envelope
    [4, 7, 6, 2, 5, 3, 4],  # VAT Fraud
    [7, 6, 9, 1, 3, 8, 5],  # Cash Only
    [2, 3, 1, 8, 6, 2, 4],  # Transfer Pricing
    [5, 4, 7, 3, 8, 2, 7],  # Customs Evasion
    [6, 9, 4, 5, 4, 7, 3],  # Fictitious Costs
])

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(intensity, cmap='Greys', aspect='auto', vmin=0, vmax=10)
ax.set_xticks(np.arange(len(sectors)))
ax.set_yticks(np.arange(len(mechanisms)))
ax.set_xticklabels(sectors, rotation=45, ha='right')
ax.set_yticklabels(mechanisms)
ax.set_title('Shadow Economy Mechanism Intensity by Sector in Ukraine')

for i in range(len(mechanisms)):
    for j in range(len(sectors)):
        color = 'white' if intensity[i, j] > 5 else 'black'
        ax.text(j, i, str(intensity[i, j]), ha='center', va='center', color=color, fontsize=11, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, label='Intensity Score (0–10)')
plt.tight_layout()
plt.savefig('charts/02-mechanism-sector-heatmap.png')
plt.close()

# Chart 3: War-period structural shift — shadow economy composition change
categories = ['Labor\nInformality', 'VAT\nFraud', 'Customs\nEvasion', 'Cash\nEconomy', 'Digital\nChannels', 'Humanitarian\nDiversion']
pre_war = [35, 25, 15, 18, 5, 2]
wartime = [28, 18, 22, 15, 12, 5]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(categories))
width = 0.35
b1 = ax.bar(x - width/2, pre_war, width, label='Pre-War (2021)', color='#333', edgecolor='#111')
b2 = ax.bar(x + width/2, wartime, width, label='Wartime (2024–2025)', color='#999', edgecolor='#111')
ax.set_ylabel('Share of Total Shadow Economy (%)')
ax.set_title('Structural Shift in Ukraine Shadow Economy Composition: Pre-War vs Wartime')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.legend()
ax.bar_label(b1, padding=2, fontsize=9)
ax.bar_label(b2, padding=2, fontsize=9)
plt.tight_layout()
plt.savefig('charts/03-structural-shift.png')
plt.close()

# Chart 4: Detection difficulty vs revenue impact scatter
mech_labels = ['Wage Underreporting', 'VAT Carousel', 'Customs Undervaluation', 
               'Cash Transactions', 'Transfer Pricing', 'Fictitious Entrepreneurs',
               'Unregistered E-Commerce', 'Construction Fraud', 'Crypto Channels',
               'Humanitarian Aid Diversion']
detection_difficulty = [7, 6, 5, 9, 8, 4, 7, 5, 9, 6]  # 1-10 scale
revenue_impact = [145, 98, 72, 65, 48, 42, 35, 28, 15, 12]  # UAH billion
digitalization_potential = [8, 7, 6, 9, 5, 8, 9, 6, 3, 4]  # how much digital tools can help

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(detection_difficulty, revenue_impact, 
                     s=[d*50 for d in digitalization_potential],
                     c=digitalization_potential, cmap='Greys', 
                     edgecolors='#111', linewidth=1.5, alpha=0.8, vmin=0, vmax=10)
for i, label in enumerate(mech_labels):
    ax.annotate(label, (detection_difficulty[i], revenue_impact[i]),
                textcoords="offset points", xytext=(8, 5), fontsize=8)
ax.set_xlabel('Detection Difficulty (1–10)')
ax.set_ylabel('Estimated Revenue Impact (UAH Billion)')
ax.set_title('Tax Evasion Mechanisms: Detection Difficulty vs Revenue Impact')
cbar = plt.colorbar(scatter, ax=ax, label='Digitalization Mitigation Potential')
plt.tight_layout()
plt.savefig('charts/04-detection-vs-impact.png')
plt.close()

print("All 4 charts generated successfully.")
