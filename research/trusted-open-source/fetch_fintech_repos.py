#!/usr/bin/env python3
"""Data collection for Fresh Repositories Watch: Financial Technology"""
import requests, json, time
from datetime import datetime, timedelta

headers = {"Accept": "application/vnd.github.v3+json"}
cutoff = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

queries = [
    f"fintech created:>{cutoff} stars:>5",
    f"trading+bot created:>{cutoff} stars:>20",
    f"quantitative+finance created:>{cutoff} stars:>5",
    f"risk+management+finance created:>{cutoff} stars:>5",
    f"payment+gateway+open+source created:>{cutoff} stars:>5",
    f"portfolio+optimization created:>{cutoff} stars:>10",
    f"financial+AI created:>{cutoff} stars:>10",
]

all_repos = {}
for q in queries:
    url = f"https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page=30"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            for item in r.json().get("items", []):
                all_repos[item["full_name"]] = {
                    "name": item["full_name"],
                    "stars": item["stargazers_count"],
                    "forks": item["forks_count"],
                    "language": item.get("language", "Unknown"),
                    "license": item.get("license", {}).get("spdx_id", "None") if item.get("license") else "None",
                    "created": item["created_at"][:10],
                    "updated": item["updated_at"][:10],
                    "description": (item.get("description") or "")[:200],
                    "topics": item.get("topics", []),
                    "open_issues": item.get("open_issues_count", 0),
                    "watchers": item.get("watchers_count", 0),
                    "has_wiki": item.get("has_wiki", False),
                }
        time.sleep(2)
    except Exception as e:
        print(f"Error on query {q}: {e}")

# Sort by stars
sorted_repos = sorted(all_repos.values(), key=lambda x: x["stars"], reverse=True)

with open("/root/hub/research/trusted-open-source/fintech_repos.json", "w") as f:
    json.dump(sorted_repos, f, indent=2)

print(f"Collected {len(sorted_repos)} unique repos")
for r in sorted_repos[:25]:
    print(f"  {r['stars']:>6} | {r['created']} | {r['language']:>10} | {r['license']:>12} | {r['name']}")
    print(f"         {r['description'][:100]}")
