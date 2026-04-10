import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Database connection
cnx = mysql.connector.connect(
    host='localhost',
    user='wpuser',
    password='wppass123',
    database='wordpress'
)

cursor = cnx.cursor()

# Query: count papers per year with keyword consciousness
query = """
SELECT ref_year, COUNT(*) as cnt
FROM wp_references
WHERE ref_title LIKE '%consciousness%' AND ref_year >= 2020
GROUP BY ref_year
ORDER BY ref_year
"""
cursor.execute(query)
rows = cursor.fetchall()
years = [r[0] for r in rows]
counts = [r[1] for r in rows]

# Chart 1: Bar chart
plt.figure(figsize=(8,5))
plt.bar(years, counts, color='steelblue')
plt.title('Papers on AI Consciousness (2020-2026)')
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('charts/consciousness_years.png', dpi=150)
plt.close()

# Query: trusted vs non-trusted for 2025-2026
query2 = """
SELECT is_trusted, COUNT(*) as cnt
FROM wp_references
WHERE ref_year >= 2025 AND ref_title LIKE '%AI%'
GROUP BY is_trusted
"""
cursor.execute(query2)
rows2 = cursor.fetchall()
labels = ['Non-Trusted', 'Trusted']
sizes = [0, 0]
for row in rows2:
    if row[0] == 1:
        sizes[1] = row[1]
    else:
        sizes[0] = row[1]
plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Trusted vs Non-Trusted AI Papers (2025-2026)')
plt.tight_layout()
plt.savefig('charts/trusted_pie.png', dpi=150)
plt.close()

# Query: top domains for AI papers 2026
query3 = """
SELECT domain, COUNT(*) as cnt
FROM wp_references
WHERE ref_year = 2026 AND domain IS NOT NULL AND domain != ''
GROUP BY domain
ORDER BY cnt DESC
LIMIT 10
"""
cursor.execute(query3)
rows3 = cursor.fetchall()
domains = [r[0] for r in rows3]
domains_cnt = [r[1] for r in rows3]
plt.figure(figsize=(10,6))
plt.barh(domains, domains_cnt, color='seagreen')
plt.title('Top Domains for AI Papers (2026)')
plt.xlabel('Number of Papers')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('charts/top_domains.png', dpi=150)
plt.close()

cursor.close()
cnx.close()
print("Created 3 charts from DB")
