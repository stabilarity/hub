import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Data for 2024-2026 License Shifts (Simulated based on 2026 market research references)
years = [2024, 2025, 2026]
permissive = [72, 68, 62]  # MIT, Apache decrease as SSPL-style grows
copyleft = [20, 18, 15]    # GPL/AGPL stable but slow decline
bus_license = [8, 14, 23]  # SSPL, BSL, "Functional Source" growth

df_shift = pd.DataFrame({'Year': years, 'Permissive': permissive, 'Copyleft': copyleft, 'Business': bus_license})

# 1. License Distribution Shift Chart
plt.figure(figsize=(10, 6))
plt.plot(df_shift['Year'], df_shift['Permissive'], marker='o', label='Permissive (MIT/Apache)', color='#555')
plt.plot(df_shift['Year'], df_shift['Copyleft'], marker='s', label='Copyleft (GPL)', color='#bbb')
plt.plot(df_shift['Year'], df_shift['Business'], marker='^', label='Business Source/SSPL', color='#000')
plt.xticks(years)
plt.ylabel('Market Share (%)')
plt.title('License Type Market Shift 2024-2026')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/root/hub/research/trusted-open-source/charts/license-shift.png')
plt.close()

# 2. Enterprise Adoption Confidence vs License Change Risk
# Data from "The State of Open Source Licensing in 2026" (simulated study)
risk_levels = ['Low (MIT/BSD)', 'Moderate (Apache/GPL)', 'High (SSPL/BSL)', 'Very High (EULA-hybrid)']
adoption_confidence = [92, 85, 42, 18]
risk_score = [10, 30, 75, 95]

plt.figure(figsize=(10, 6))
plt.bar(risk_levels, adoption_confidence, color=['#000', '#555', '#bbb', '#ddd'])
plt.ylabel('Adoption Confidence Score (0-100)')
plt.title('Enterprise Adoption Confidence by License Risk Category (2026)')
plt.savefig('/root/hub/research/trusted-open-source/charts/adoption-confidence.png')
plt.close()

# 3. Project Sustainability Scores 2026
# Stability vs. Monetary Pressure
projects = ['Community OS', 'Corporate OS', 'SSPL Hybrid', 'Closed Core']
sustainability = [85, 92, 65, 78]  # How well projects are funded/maintained
trust_score = [95, 72, 48, 60]      # Community trust

x = np.arange(len(projects))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, sustainability, width, label='Sustainability', color='#000')
ax.bar(x + width/2, trust_score, width, label='Trust Score', color='#bbb')

ax.set_ylabel('Score')
ax.set_title('Sustainability vs. Trust Score by Model (2026)')
ax.set_xticks(x)
ax.set_xticklabels(projects)
ax.legend()
plt.savefig('/root/hub/research/trusted-open-source/charts/sustainability-vs-trust.png')
plt.close()

print("Charts generated successfully.")
