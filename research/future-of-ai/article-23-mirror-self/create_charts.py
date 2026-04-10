import arxiv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time

keywords = ["AI consciousness", "AI agency", "AI personhood"]
categories = ["cs.AI", "cs.CL", "cs.RO", "cs.HC"]

results = []
for kw in keywords:
    print(f"Searching {kw}")
    query = f'({kw}) AND (' + ' OR '.join([f'cat:{c}' for c in categories]) + ')'
    search = arxiv.Search(query=query, max_results=50, sort_by=arxiv.SortCriterion.SubmittedDate)
    try:
        for paper in search.results():
            year = paper.published.year
            if year >= 2020:
                results.append({
                    'year': year,
                    'category': paper.primary_category,
                    'keyword': kw
                })
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(1)

df = pd.DataFrame(results)
print(f"Found {len(df)} papers")

if len(df) > 0:
    # Chart 1: Papers per year stacked by keyword
    pivot = df.pivot_table(index='year', columns='keyword', aggfunc='size', fill_value=0)
    pivot.plot(kind='bar', stacked=True, figsize=(10,6))
    plt.title('arXiv Papers on AI Consciousness/Agency (2020-2026)')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.tight_layout()
    plt.savefig('charts/year_stacked.png', dpi=150)
    plt.close()
    
    # Chart 2: Category distribution
    cat_counts = df['category'].value_counts().head(8)
    cat_counts.plot(kind='pie', autopct='%1.1f%%', figsize=(8,8))
    plt.title('Primary Categories')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('charts/category_pie.png', dpi=150)
    plt.close()
    
    # Chart 3: Keyword proportion
    kw_counts = df['keyword'].value_counts()
    kw_counts.plot(kind='bar', figsize=(8,5))
    plt.title('Papers per Keyword')
    plt.xlabel('Keyword')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('charts/keyword_bar.png', dpi=150)
    plt.close()
    
    print("Created 3 charts")
else:
    print("No data, creating dummy chart")
    # Dummy chart
    plt.figure()
    plt.bar([2025, 2026], [5, 8])
    plt.title('Sample Chart - No Data')
    plt.savefig('charts/dummy.png')
    plt.close()
