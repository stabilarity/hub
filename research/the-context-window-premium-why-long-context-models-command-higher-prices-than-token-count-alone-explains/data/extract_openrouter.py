"""Flatten the OpenRouter /api/v1/models snapshot (2026-10-20) into openrouter_models.csv.

The raw snapshot was downloaded once from https://openrouter.ai/api/v1/models on 2026-10-20
and is committed as openrouter_models_2026-10-20.json.gz. This script only reads that file.
Prices in the API are USD per token; the CSV stores USD per 1M tokens.
"""
import csv
import gzip
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "openrouter_models_2026-10-20.json.gz")
OUT = os.path.join(HERE, "openrouter_models.csv")


def per_m(v):
    if v in (None, ""):
        return ""
    return round(float(v) * 1e6, 6)


def main():
    with gzip.open(SRC, "rt", encoding="utf-8") as f:
        data = json.load(f)["data"]
    rows = []
    for m in data:
        p = m.get("pricing", {})
        tier = next((o for o in p.get("overrides", []) or [] if "min_prompt_tokens" in o), None)
        aa = ((m.get("benchmarks") or {}).get("artificial_analysis") or {}).get("intelligence_index")
        rows.append({
            "id": m["id"],
            "name": m.get("name", ""),
            "context_tokens": m.get("context_length") or "",
            "output_modalities": "+".join(m.get("architecture", {}).get("output_modalities", [])),
            "input_usd_per_mtok": per_m(p.get("prompt")),
            "output_usd_per_mtok": per_m(p.get("completion")),
            "cached_input_usd_per_mtok": per_m(p.get("input_cache_read")),
            "tier_min_prompt_tokens": tier["min_prompt_tokens"] if tier else "",
            "tier_input_usd_per_mtok": per_m(tier.get("prompt")) if tier else "",
            "tier_output_usd_per_mtok": per_m(tier.get("completion")) if tier else "",
            "aa_intelligence_index": "" if aa is None else aa,
            "created_unix": m.get("created", ""),
        })
    rows.sort(key=lambda r: r["id"])
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
