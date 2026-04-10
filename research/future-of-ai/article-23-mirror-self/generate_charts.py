import arxiv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import time
import os

# Search for keywords
keywords = [
    "AI consciousness",
    "artificial consciousness",
    "machine consciousness",
    "AI agency",
    "AI personhood",
    "AI moral status",
    "AI rights",
    "AI self-awareness",
    "AI theory of mind",
    "AI mirror"
]

categories = ["cs.AI", "cs.CL", "cs.CY", "cs.RO", "cs.HC", "cs.LG", "cs.MA", "cs.SI", "cs.SD", "cs.IR"]

# Store results
results = []

for keyword in keywords:
    print(f"Searching for: {keyword}")
    query = f'({keyword}) AND ('
    query += ' OR '.join([f'cat:{cat}' for cat in categories])
    query += ')'
    search = arxiv.Search(
        query=query,
        max_results=200,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    try:
        for paper in search.results():
            year = paper.published.year
            if year >= 2020:
                results.append({
                    'title': paper.title,
                    'year': year,
                    'month': paper.published.month,
                    'category': paper.primary_category,
                    'keyword': keyword,
                    'doi': paper.doi if hasattr(paper, 'doi') else '',
                    'authors': ', '.join([a.name for a in paper.authors])
                })
    except Exception as e:
        print(f"Error for {keyword}: {e}")
    time.sleep(1)

# Convert to DataFrame
df = pd.DataFrame(results)
print(f"Total papers found: {len(df)}")
df.to_csv('arxiv_papers.csv', index=False)

# Chart 1: Papers per year (stacked by keyword)
plt.figure(figsize=(10,6))
pivot_year = df.pivot_table(index='year', columns='keyword', aggfunc='size', fill_value=0)
pivot_year.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('arXiv Papers on AI Consciousness/Agency (2020-2026)')
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.xticks(rotation=0)
plt.legend(title='Keyword', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('charts/arxiv_trends_stacked.png', dpi=300)
plt.close()

# Chart 2: Distribution across primary categories
plt.figure(figsize=(8,8))
category_counts = df['category'].value_counts().head(10)
category_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Primary Categories of AI Consciousness/Agency Papers')
plt.ylabel('')
plt.tight_layout()
plt.savefig('charts/category_distribution.png', dpi=300)
plt.close()

# Chart 3: Monthly trend for 2025-2026
df_recent = df[df['year'] >= 2025]
if len(df_recent) > 0:
    df_recent['month_year'] = df_recent['year'].astype(str) + '-' + df_recent['month'].astype(str).str.zfill(2)
    monthly = df_recent['month_year'].value_counts().sort_index()
    plt.figure(figsize=(12,6))
    monthly.plot(kind='line', marker='o')
    plt.title('Monthly Publication Trend (2025-2026)')
    plt.xlabel('Month')
    plt.ylabel('Papers')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('charts/monthly_trend.png', dpi=300)
    plt.close()

# Chart 4: Top authors by paper count
author_counts = df['authors'].str.split(', ').explode().value_counts().head(15)
plt.figure(figsize=(10,6))
author_counts.plot(kind='barh')
plt.title('Top Authors in AI Consciousness/Agency Research')
plt.xlabel('Number of Papers')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('charts/top_authors.png', dpi=300)
plt.close()

print("Charts generated.")
