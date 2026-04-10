import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
labels = ['arXiv', 'Stabilarity Research Hub', 'arXiv preprint', 'Scientific Reports', 'Mathematics']
sizes = [177, 28, 19, 6, 5]
plt.figure(figsize=(8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Top Publication Venues for AI Papers (2025-2026)')
plt.tight_layout()
plt.savefig('charts/venue_pie.png', dpi=150)
print('Pie chart saved')
