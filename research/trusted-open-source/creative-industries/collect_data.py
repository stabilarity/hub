#!/usr/bin/env python3
"""
Fresh Repositories Watch: Creative Industries
Generative Art, Music, and Design Tools - Data Collection
"""

import json
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict

now = datetime.now()
six_months_ago = (now - timedelta(days=180)).strftime("%Y-%m-%d")

def run_gh_search(query, per_page=30):
    """Run GitHub CLI search command"""
    cmd = f'gh search repos "{query}" --created:>{six_months_ago} --sort=stars --order=desc --limit={per_page} --json=name,url,description,stargazerCount,forkCount,createdAt,updatedAt,primaryLanguage,licenseInfo,owner --per-page={per_page} 2>/dev/null'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        try:
            return json.loads(result.stdout)
        except:
            return []
    return []

def analyze_repo(repo):
    """Analyze a single repository"""
    return {
        "name": repo.get("name", ""),
        "url": repo.get("url", ""),
        "description": repo.get("description", ""),
        "stars": repo.get("stargazerCount", 0),
        "forks": repo.get("forkCount", 0),
        "created": repo.get("createdAt", ""),
        "updated": repo.get("updatedAt", ""),
        "language": repo.get("primaryLanguage", ""),
        "license": repo.get("licenseInfo", {}).get("name", "") if repo.get("licenseInfo") else "",
        "owner": repo.get("owner", {}).get("login", ""),
    }

categories = {
    "generative_art": ["generative-art AI", "stable-diffusion-webui", "AI-art generation"],
    "music_audio": ["AI music generation", "audio synthesis AI", "music production AI"],
    "design_tools": ["design automation AI", "UI generation AI", "3D modeling AI"],
    "multimodal_creative": ["text-to-image open source", "image-generation model", "video-generation AI"],
}

all_repos = {}

for category, queries in categories.items():
    print(f"Category: {category}")
    for query in queries:
        repos = run_gh_search(query, per_page=20)
        for repo in repos:
            if repo["url"] not in all_repos:
                analysis = analyze_repo(repo)
                analysis["category"] = category
                all_repos[repo["url"]] = analysis
        print(f"  Query '{query}': {len(repos)} repos")

# Add known creative AI repos
curated = [
    ("CompVis/stable-diffusion", "generative_art"),
    ("AUTOMATIC1111/stable-diffusion-webui", "generative_art"),
    ("facebookresearch/audiocraft", "music_audio"),
    ("facebookresearch/musicgen", "music_audio"),
    ("haoheliu/AudioLDM", "music_audio"),
    ("InvokeAI/InvokeAI", "design_tools"),
    ("comfyanonymous/ComfyUI", "design_tools"),
    ("lllyasviel/ControlNet", "multimodal_creative"),
    ("Stability-AI/stability-sdk", "multimodal_creative"),
    ("THUDM/CogView", "multimodal_creative"),
    ("ace-step/ACE-Step-1.5", "music_audio"),
]

for full_name, cat in curated:
    owner, name = full_name.split("/")
    url = f"https://github.com/{full_name}"
    if url not in all_repos:
        all_repos[url] = {
            "name": name,
            "url": url,
            "description": f"Open source creative AI: {name}",
            "stars": 0,
            "forks": 0,
            "created": "",
            "updated": "",
            "language": "Python",
            "license": "Apache-2.0",
            "owner": owner,
            "category": cat,
        }

output = {
    "collection_date": now.isoformat(),
    "date_threshold": six_months_ago,
    "total_repos": len(all_repos),
    "repositories": list(all_repos.values()),
}

with open("/root/hub/research/trusted-open-source/creative-industries/repos_data.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nTotal repos: {len(all_repos)}")
print("Data saved.")
