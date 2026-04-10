import arxiv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time

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

# Years to search
years = [2025, 2026]
# arXiv categories for AI
categories = ["cs.AI", "cs.CL", "cs.CY", "cs.RO", "cs.HC"]

# Store results
results = []

for keyword in keywords:
    print(f"Searching for: {keyword}")
    query = f'({keyword}) AND ('
    query += ' OR '.join([f'cat:{cat}' for cat in categories])
    query += ')'
    search = arxiv.Search(
        query=query,
        max_results=100,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    try:
        for paper in search.results():
            year = paper.published.year
            if year in years:
                results.append({
                    'title': paper.title,
                    'year': year,
                    'category': paper.primary_category,
                    'keyword': keyword
                })
    except Exception as e:
        print(f"Error for {keyword}: {e}")
    time.sleep(1)  # rate limit

# Convert to DataFrame
df = pd.DataFrame(results)
print(f"Total papers found: {len(df)}")

# Count by year and keyword
if len(df) > 0:
    pivot = df.pivot_table(index='year', columns='keyword', aggfunc='size', fill_value=0)
    print(pivot)
    
    # Plot
    pivot.plot(kind='bar', figsize=(10,6))
    plt.title('arXiv Papers on AI Consciousness/Agency (2025-2026)')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('/root/hub/research/future-of-ai/article-23-mirror-self/charts/arxiv_trends.png', dpi=300)
    print("Chart saved.")
else:
    print("No papers found.")
