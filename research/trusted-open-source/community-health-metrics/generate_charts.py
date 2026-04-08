#!/usr/bin/env python3
"""Generate community health metrics charts from research data"""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = "/root/hub/research/trusted-open-source/community-health-metrics/charts"
os.makedirs(OUT, exist_ok=True)

# ---- Chart 1: Contributor Diversity vs Age ----
div_data = [
    {"owner":"torvalds","repo":"linux","years_old":21,"stars":172000,"forks":28000,"diversity_score":82.0},
    {"owner":"golang","repo":"go","years_old":14,"stars":128000,"forks":18500,"diversity_score":91.4},
    {"owner":"microsoft","repo":"vscode","years_old":11,"stars":165000,"forks":31000,"diversity_score":100.0},
    {"owner":"facebook","repo":"react","years_old":13,"stars":230000,"forks":47000,"diversity_score":100.0},
    {"owner":"twbs","repo":"bootstrap","years_old":15,"stars":170000,"forks":79000,"diversity_score":75.6},
    {"owner":"tensorflow","repo":"tensorflow","years_old":11,"stars":185000,"forks":95000,"diversity_score":84.1},
    {"owner":"pytorch","repo":"pytorch","years_old":10,"stars":83000,"forks":22000,"diversity_score":83.0},
    {"owner":"kubernetes","repo":"kubernetes","years_old":12,"stars":112000,"forks":22000,"diversity_score":93.3},
    {"owner":"grafana","repo":"grafana","years_old":12,"stars":63000,"forks":15000,"diversity_score":52.5},
    {"owner":"tailscale","repo":"tailscale","years_old":8,"stars":32000,"forks":2900,"diversity_score":40.0},
    {"owner":"neovim","repo":"neovim","years_old":12,"stars":38000,"forks":3500,"diversity_score":31.7},
    {"owner":"ziglang","repo":"zig","years_old":10,"stars":36000,"forks":2300,"diversity_score":36.0},
    {"owner":"denoland","repo":"deno","years_old":8,"stars":96000,"forks":6400,"diversity_score":100.0},
    {"owner":"vitejs","repo":"vite","years_old":6,"stars":73000,"forks":9600,"diversity_score":100.0},
    {"owner":"tailwindlabs","repo":"tailwindcss","years_old":9,"stars":85000,"forks":9600,"diversity_score":94.4},
    {"owner":"vercel","repo":"next.js","years_old":7,"stars":130000,"forks":29000,"diversity_score":100.0},
    {"owner":"sveltejs","repo":"svelte","years_old":10,"stars":81000,"forks":11000,"diversity_score":81.0},
    {"owner":"shadcn-ui","repo":"ui","years_old":3,"stars":82000,"forks":8500,"diversity_score":100.0},
    {"owner":"huggingface","repo":"transformers","years_old":7,"stars":78000,"forks":20000,"diversity_score":100.0},
    {"owner":"langchain-ai","repo":"langchain","years_old":4,"stars":70000,"forks":13000,"diversity_score":100.0},
    {"owner":"numpy","repo":"numpy","years_old":20,"stars":27000,"forks":8700,"diversity_score":13.5},
    {"owner":"pandas-dev","repo":"pandas","years_old":17,"stars":42000,"forks":15000,"diversity_score":24.7},
    {"owner":"scikit-learn","repo":"scikit-learn","years_old":18,"stars":60000,"forks":27000,"diversity_score":33.3},
    {"owner":"matplotlib","repo":"matplotlib","years_old":23,"stars":20000,"forks":9000,"diversity_score":8.7},
    {"owner":"jupyter","repo":"jupyterlab","years_old":12,"stars":14000,"forks":3500,"diversity_score":11.7},
    {"owner":"diffusers","repo":"diffusers","years_old":4,"stars":38000,"forks":7000,"diversity_score":95.0},
    {"owner":"ollama","repo":"ollama","years_old":3,"stars":95000,"forks":7800,"diversity_score":100.0},
    {"owner":"apache","repo":"superset","years_old":9,"stars":62000,"forks":14000,"diversity_score":68.9},
    {"owner":"apache","repo":"airflow","years_old":11,"stars":40000,"forks":18000,"diversity_score":36.4},
    {"owner":"apache","repo":"kafka","years_old":15,"stars":29000,"forks":13000,"diversity_score":19.3},
]

years = [d["years_old"] for d in div_data]
stars = [d["stars"] for d in div_data]
diversity = [d["diversity_score"] for d in div_data]
labels = [f"{d['owner']}/{d['repo']}" for d in div_data]
sizes = [max(d["diversity_score"] * 3, 30) for d in div_data]

fig, ax = plt.subplots(figsize=(13, 8))
scatter = ax.scatter(years, stars, s=sizes, alpha=0.65, c=diversity, cmap='viridis', edgecolors='white', linewidth=0.5)
for i, label in enumerate(labels):
    if stars[i] > 80000 or years[i] > 17 or diversity[i] > 95:
        ax.annotate(label, (years[i], stars[i]), fontsize=6.5, alpha=0.85, xytext=(4, 4), textcoords='offset points')
ax.set_xlabel('Project Age (Years)', fontsize=12)
ax.set_ylabel('GitHub Stars (thousands)', fontsize=12)
ax.set_title('Contributor Diversity vs Project Age\n(Bubble Size = Community Engagement Score, Color = Diversity Index)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Diversity Score', fontsize=10)
ax.set_ylim(0, 250000)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "contributor_diversity_vs_age.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1 saved")

# ---- Chart 2: Bus Factor Distribution ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
cats = ["micro", "small", "medium", "large"]
titles = ["Micro Projects (<100 contribs)", "Small (100-500 contribs)", "Medium (500-5000 contribs)", "Large Projects (>5000 contribs)"]
colors = ['#c62828', '#f57f17', '#2e7d32', '#1565c0']
bus_data = {
    "micro": [("Express.js",3),("Request",2),("Moment.js",2),("Left-pad",1),("colors.js",1),("mkdirp",2),("rimraf",2),("nconf",1)],
    "small": [("Lodash",4),("Axios",3),("Webpack",5),("Babel",6),("Yarn",5),("Rollup",3),("Prettier",4)],
    "medium": [("React",12),("Vue.js",8),("Angular",15),("Django",18),("Flask",7),("Rails",14),("Spring",16)],
    "large": [("Linux",45),("Go",22),("VS Code",28),("Kubernetes",35),("TensorFlow",25),("CPython",30),("Firefox",40)],
}
for idx, (cat, title, color) in enumerate(zip(cats, titles, colors)):
    ax = axes[idx//2, idx%2]
    names_bf = bus_data[cat]
    names = [x[0] for x in names_bf]
    bfs = [x[1] for x in names_bf]
    bars = ax.barh(names, bfs, color=color, alpha=0.75, edgecolor='white')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Bus Factor (Core Contributors)', fontsize=10)
    for bar, bf in zip(bars, bfs):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, str(bf), va='center', fontsize=9)
fig.suptitle('Bus Factor Distribution by Project Size\n(Number of contributors whose departure would cripple the project)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, "bus_factor_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2 saved")

# ---- Chart 3: Sustainability Signals Heatmap ----
import numpy as np
sust_projects = [
    ("Kubernetes",12,95,4,98,97),
    ("Linux",52,99,2,100,99),
    ("VS Code",11,90,3,96,94),
    ("React",10,92,6,95,95),
    ("PostgreSQL",3,80,10,85,92),
    ("Vue.js",8,85,5,88,90),
    ("TensorFlow",6,88,8,90,88),
    ("Go",6,82,7,91,88),
    ("Django",4,78,12,82,85),
    ("Ansible",7,75,9,80,82),
    ("Flask",3,60,18,65,72),
    ("Express",4,55,15,60,68),
    ("Moment.js",1,30,30,40,45),
    ("Request",0,15,60,20,25),
    ("Left-pad",0,10,90,0,10),
]
names = [p[0] for p in sust_projects]
# release_freq, contrib_velocity, issue_response(inverted), ci_score, sustainability
matrix = np.array([[p[1], p[2], 100-p[3], p[4]] for p in sust_projects])
signal_names = ["Release\nFrequency", "Contributor\nVelocity", "Issue\nResponse\n(inverted)", "CI/CD\nAdoption"]

fig, ax = plt.subplots(figsize=(12, 9))
im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(4)); ax.set_xticklabels(signal_names, fontsize=10)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
for i in range(len(names)):
    for j in range(4):
        val = int(matrix[i,j])
        color = 'white' if val < 30 or val > 80 else 'black'
        ax.text(j, i, str(val), ha='center', va='center', color=color, fontsize=8, fontweight='bold')
ax.set_title('Sustainability Signal Heatmap\n(All signals normalized 0-100, higher = better for project longevity)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(im, ax=ax); cbar.set_label('Signal Score', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "sustainability_signals_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3 saved")

# ---- Chart 4: Diversity Analysis ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 4a: Stars vs Forks colored by diversity
ax1 = axes[0]
scatter1 = ax1.scatter(stars, [d["forks"] for d in div_data], c=diversity, cmap='plasma', s=80, alpha=0.7, edgecolors='white')
ax1.set_xlabel('GitHub Stars (thousands)', fontsize=11)
ax1.set_ylabel('GitHub Forks', fontsize=11)
ax1.set_title('Stars vs Forks by Diversity Score', fontsize=11, fontweight='bold')
plt.colorbar(scatter1, ax=ax1, label='Diversity Score')

# 4b: Mean diversity by age group
ax2 = axes[1]
age_bins = ['0-2 yrs', '3-5 yrs', '6-10 yrs', '10+ yrs']
age_groups_g = {'0-2 yrs':[],'3-5 yrs':[],'6-10 yrs':[],'10+ yrs':[]}
for d in div_data:
    if d["years_old"] <= 2: age_groups_g['0-2 yrs'].append(d["diversity_score"])
    elif d["years_old"] <= 5: age_groups_g['3-5 yrs'].append(d["diversity_score"])
    elif d["years_old"] <= 10: age_groups_g['6-10 yrs'].append(d["diversity_score"])
    else: age_groups_g['10+ yrs'].append(d["diversity_score"])
means = [np.mean(age_groups_g[b]) if age_groups_g[b] else 0 for b in age_bins]
counts = [len(age_groups_g[b]) for b in age_bins]
bar_colors = ['#ef9a9a','#fff59d','#a5d6a7','#81d4fa']
bars = ax2.bar(age_bins, means, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, count, mean in zip(bars, counts, means):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'{mean:.1f}\n(n={count})', ha='center', fontsize=9)
ax2.set_xlabel('Project Age Category', fontsize=11)
ax2.set_ylabel('Mean Diversity Score', fontsize=11)
ax2.set_title('Mean Diversity Score by Project Age', fontsize=11, fontweight='bold')
ax2.set_ylim(0, max(means)*1.4)
plt.suptitle('Community Health: Contributor Diversity Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "diversity_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Chart 4 saved")
print("All charts generated!")
