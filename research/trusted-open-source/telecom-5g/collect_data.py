#!/usr/bin/env python3
"""Collect data on open-source telecommunications and 5G tools from GitHub API."""
import requests
import json
import time

HEADERS = {"Accept": "application/vnd.github.v3+json"}

def get_repo(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code == 200:
        return resp.json()
    return None

def get_languages(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/languages"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code == 200:
        return resp.json()
    return {}

def get_contributors(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/contributors?per_page=5"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code == 200:
        return len(resp.json())
    return 0

MAJOR_REPOS = [
    ("open5gs", "open5gs"),
    ("srsRAN", "srsRAN_4G"),
    ("oai", "openairinterface5g"),
    ("free5gc", "free5gc"),
    ("srsRAN", "srsRAN"),
    ("open5gs", "open5gs-webui"),
    ("oai", "oai-gnb"),
    ("omecProject", "omec-project"),
    ("free5gc", "free5gc-compose"),
]

print("Collecting data for open-source telecom repositories...")
print("=" * 70)

results = []
for owner, repo in MAJOR_REPOS:
    print(f"Fetching {owner}/{repo}...")
    data = get_repo(owner, repo)
    if data:
        langs = get_languages(owner, repo)
        contributors = get_contributors(owner, repo)
        r = {
            "full_name": data["full_name"],
            "description": data.get("description", ""),
            "stars": data.get("stargazers_count", 0),
            "forks": data.get("forks_count", 0),
            "open_issues": data.get("open_issues_count", 0),
            "subscribers": data.get("subscribers_count", 0),
            "created": data.get("created_at", ""),
            "pushed": data.get("pushed_at", ""),
            "updated": data.get("updated_at", ""),
            "language": data.get("language", ""),
            "languages": langs,
            "contributors_count": contributors,
            "license": data.get("license", {}).get("name", "Unknown"),
            "topics": data.get("topics", []),
            "url": data["html_url"],
        }
        results.append(r)
        print(f"  Stars: {r['stars']:,} | Forks: {r['forks']:,} | Lang: {r['language']} | Created: {r['created'][:4]}")
    else:
        print(f"  NOT FOUND or rate limited")
    time.sleep(1.2)

output_path = "/root/hub/research/trusted-open-source/telecom-5g/repo_data.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} repos")
for r in sorted(results, key=lambda x: x["stars"], reverse=True):
    print(f"  {r['full_name']}: {r['stars']:,} stars, {r['forks']:,} forks")
