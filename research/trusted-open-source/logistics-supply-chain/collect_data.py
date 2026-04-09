#!/usr/bin/env python3
"""
Data collection for Logistics & Supply Chain OSS repositories.
"""
import json
import urllib.request
import urllib.error
import time
from datetime import datetime

REPOS = [
    {"owner": "meta", "repo": "osrm-backend", "category": "Routing Engine"},
    {"owner": "graphhopper", "repo": "graphhopper", "category": "Routing Engine"},
    {"owner": "Google", "repo": "or-tools", "category": "Optimization"},
    {"owner": "fleetbase", "repo": "fleetbase", "category": "Fleet Management"},
    {"owner": "openboxes", "repo": "openboxes", "category": "Warehouse"},
    {"owner": "openfoodnetwork", "repo": "openfoodnetwork", "category": "Supply Chain"},
    {"owner": "NetworkCube", "repo": "vesseljs", "category": "Tracking"},
    {"owner": "Calytica", "repo": "freight-shield", "category": "Freight"},
]

def fetch_repo(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    req = urllib.request.Request(url, headers={"User-Agent": "ResearchBot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            d = json.loads(resp.read().decode())
            return {
                "name": d["name"], "full_name": d["full_name"],
                "description": d.get("description", ""),
                "stars": d["stargazers_count"], "forks": d["forks_count"],
                "open_issues": d["open_issues_count"],
                "subscribers": d.get("subscribers_count", 0),
                "created_at": d["created_at"], "pushed_at": d["pushed_at"],
                "language": d.get("language", ""),
                "license": (d.get("license") or {}).get("name", "None"),
                "topics": d.get("topics", []),
                "category": next((r["category"] for r in REPOS if r["owner"]==owner and r["repo"]==repo), "Other")
            }
    except Exception as e:
        print(f"  Error {owner}/{repo}: {e}")
        return None

results = []
for r in REPOS:
    print(f"Fetching {r['owner']}/{r['repo']}...", end=" ", flush=True)
    d = fetch_repo(r["owner"], r["repo"])
    if d:
        results.append(d)
        print(f"OK - {d['stars']:,} stars")
    else:
        print("FAILED")
    time.sleep(0.6)

with open("/root/hub/research/trusted-open-source/logistics-supply-chain/repo_data.json", "w") as f:
    json.dump({"collection_date": datetime.now().isoformat(), "repos": results}, f, indent=2)

print(f"\nCollected {len(results)} repos")
cats = {}
for r in results:
    c = r["category"]
    if c not in cats: cats[c] = {"count":0,"stars":0}
    cats[c]["count"] += 1; cats[c]["stars"] += r["stars"]
for cat, s in sorted(cats.items(), key=lambda x: -x[1]["stars"]):
    print(f"  {cat}: {s['count']} repos, {s['stars']:,} stars")
