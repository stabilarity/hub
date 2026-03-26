#!/usr/bin/env python3
"""Analysis for: Digital Payment Adoption and Shadow Economy Reduction"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.family': 'serif',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
    'savefig.bbox': 'tight'
})

# Chart 1: Ukraine digital payment adoption vs shadow economy (2015-2025)
years = np.arange(2015, 2026)
# Cashless transaction share of total payments (%)
cashless_share = [25, 29, 34, 39, 45, 50, 54, 48, 55, 62, 68]
# Shadow economy as % of GDP (Ministry of Economy estimates)
shadow_pct = [40, 38, 37, 35, 33, 32, 30, 40, 36, 33, 30]

fig, ax1 = plt.subplots()
color1 = '#333333'
color2 = '#888888'
ax1.set_xlabel('Year')
ax1.set_ylabel('Cashless Transactions (% of total)', color=color1)
line1 = ax1.plot(years, cashless_share, 'o-', color=color1, linewidth=2, label='Cashless Share (%)')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
ax2.set_ylabel('Shadow Economy (% of GDP)', color=color2)
line2 = ax2.plot(years, shadow_pct, 's--', color=color2, linewidth=2, label='Shadow Economy (%)')
ax2.tick_params(axis='y', labelcolor=color2)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')
plt.title('Ukraine: Digital Payment Adoption vs Shadow Economy Size (2015–2025)')
ax1.annotate('Full-scale\ninvasion', xy=(2022, 48), xytext=(2020, 42),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)
plt.savefig('charts/01-cashless-vs-shadow.png')
plt.close()

# Chart 2: Diia platform user growth
diia_years = [2020, 2021, 2022, 2023, 2024, 2025]
diia_users_m = [1.5, 6.0, 14.0, 18.5, 20.0, 23.0]
diia_services = [9, 25, 50, 90, 120, 140]

fig, ax1 = plt.subplots()
bars = ax1.bar(diia_years, diia_users_m, color='#555555', width=0.6, label='Users (millions)')
ax1.set_xlabel('Year')
ax1.set_ylabel('Registered Users (millions)')
ax1.set_ylim(0, 28)

ax2 = ax1.twinx()
ax2.plot(diia_years, diia_services, 'D-', color='#111111', linewidth=2, label='Digital Services Available')
ax2.set_ylabel('Number of Digital Services')
ax2.set_ylim(0, 170)

ax1.legend(loc='upper left')
ax2.legend(loc='center right')
plt.title('Diia Platform Growth: Users and Services (2020–2025)')
plt.savefig('charts/02-diia-growth.png')
plt.close()

# Chart 3: Cross-country comparison - digital payments vs shadow economy
countries = ['Sweden', 'Estonia', 'Poland', 'Georgia', 'Ukraine', 'Romania', 'Turkey', 'India']
cashless_pct = [98, 92, 78, 65, 68, 55, 52, 40]
shadow_pct_c = [8, 15, 18, 22, 30, 28, 27, 42]

fig, ax = plt.subplots()
ax.scatter(cashless_pct, shadow_pct_c, s=120, c='#333333', zorder=5)
for i, country in enumerate(countries):
    ax.annotate(country, (cashless_pct[i], shadow_pct_c[i]),
               textcoords="offset points", xytext=(8, 5), fontsize=9)

# Trend line
z = np.polyfit(cashless_pct, shadow_pct_c, 1)
p = np.poly1d(z)
x_line = np.linspace(35, 100, 100)
ax.plot(x_line, p(x_line), '--', color='#999999', linewidth=1.5, label=f'Linear fit (r={np.corrcoef(cashless_pct, shadow_pct_c)[0,1]:.2f})')

ax.set_xlabel('Cashless Transaction Share (%)')
ax.set_ylabel('Shadow Economy (% of GDP)')
ax.set_title('Cross-Country: Digital Payment Penetration vs Shadow Economy Size (2025)')
ax.legend()
plt.savefig('charts/03-cross-country-scatter.png')
plt.close()

# Chart 4: Sector-level informality rates in Ukraine
sectors = ['Agriculture', 'Construction', 'Trade &\nFood Services', 'Transport', 'IT &\nCommunications', 'Finance &\nInsurance']
informality = [55, 48, 35, 28, 8, 5]
digital_penetration = [15, 22, 55, 60, 95, 98]

x = np.arange(len(sectors))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, informality, width, label='Informality Rate (%)', color='#555555')
bars2 = ax.bar(x + width/2, digital_penetration, width, label='Digital Payment Use (%)', color='#bbb')

ax.set_ylabel('Percentage (%)')
ax.set_title('Ukraine: Sector-Level Informality vs Digital Payment Penetration (2025)')
ax.set_xticks(x)
ax.set_xticklabels(sectors)
ax.legend()

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width()/2, height),
               xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width()/2, height),
               xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

plt.savefig('charts/04-sector-informality.png')
plt.close()

print("All 4 charts generated successfully.")
