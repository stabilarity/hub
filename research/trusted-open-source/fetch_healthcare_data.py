#!/usr/bin/env python3
"""Data collection for Healthcare AI Open Source Repositories Watch"""
import requests, json, time

# Key healthcare AI repos to analyze
repos = [
    "Project-MONAI/MONAI",
    "microsoft/FLAML", 
    "huggingface/transformers",
    "openai/whisper",
    "stanford-crfm/helm",
    "haotian-liu/LLaVA",
    "bowang-lab/MedSAM",
    "microsoft/lida",
    "rasbt/LLMs-from-scratch",
    "vllm-project/vllm"
]

healthcare_repos = [
    "Project-MONAI/MONAI",
    "bowang-lab/MedSAM", 
    "MIT-LCP/mimic-code",
    "synthetichealth/synthea",
    "microsoft/hi-ml",
    "google-health/genomics-research",
    "facebookresearch/fastMRI",
    "NVIDIA/NVFlare",
    "owkin/FLamby",
    "StanfordMIMI/skm-tea"
]

data = []
for repo in healthcare_repos:
    try:
        r = requests.get(f"https://api.github.com/repos/{repo}", 
                        headers={"Accept": "application/vnd.github.v3+json"},
                        timeout=10)
        if r.status_code == 200:
            d = r.json()
            data.append({
                "name": d["name"],
                "full_name": d["full_name"],
                "stars": d["stargazers_count"],
                "forks": d["forks_count"],
                "open_issues": d["open_issues_count"],
                "watchers": d["subscribers_count"],
                "created": d["created_at"][:10],
                "updated": d["pushed_at"][:10],
                "language": d["language"],
                "license": d.get("license", {}).get("spdx_id", "None") if d.get("license") else "None",
                "description": d.get("description", "")[:100]
            })
            time.sleep(0.5)
    except Exception as e:
        print(f"Error fetching {repo}: {e}")

with open("healthcare_repos_data.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Collected data for {len(data)} repos")
for d in data:
    print(f"  {d['full_name']}: {d['stars']} stars, {d['forks']} forks, license={d['license']}")
