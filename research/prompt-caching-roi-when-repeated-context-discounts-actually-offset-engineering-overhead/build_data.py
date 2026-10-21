"""Build the compact trace files in data/ from the public source traces.

Not needed to reproduce the article: analysis.py reads only the committed files in data/.
Run this only to rebuild them from the original downloads.

Sources
- TraceLab v0.0.1 (CC BY 4.0, SyFI Lab, University of Washington):
  https://github.com/uw-syfi/TraceLab/releases/download/v0.0.1/syfi_coding_trace.jsonl.gz
  sha256 9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b
  Changes made: kept one row per LLM round with timing and token fields only; session ids
  re-indexed to integers; user, project, tool and content fields dropped.
- Mooncake FAST'25 traces (Apache-2.0, Moonshot AI / Tsinghua University):
  https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces
  conversation_trace.jsonl and toolagent_trace.jsonl, copied unchanged (gzip-compressed).

Usage: python3 build_data.py <syfi_coding_trace.jsonl.gz> <mooncake_traces_dir>
"""
import csv
import datetime as dt
import gzip
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def build_tracelab(src):
    sess_ix = {}
    rows = []
    with gzip.open(src, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            te = d.get("timing_events") or []
            if not te:
                continue
            t0 = min(e["timestamp"] for e in te)
            ts = dt.datetime.fromisoformat(t0.replace("Z", "+00:00")).timestamp()
            key = (d["provider"], d["session_id"])
            sid = sess_ix.setdefault(key, len(sess_ix))
            rows.append([
                d["provider"], sid, d["round_index"], round(ts, 3),
                d.get("input_tokens_total") or 0, d.get("prefix_tokens") or 0,
                d.get("output_tokens") or 0,
                d.get("claude_cache_read_input_tokens") if d.get("claude_cache_read_input_tokens") is not None else -1,
                d.get("claude_cache_creation_input_tokens") if d.get("claude_cache_creation_input_tokens") is not None else -1,
                d.get("first_input_event_type") or "", d.get("model") or "",
            ])
    rows.sort(key=lambda r: (r[1], r[3], r[2]))
    t_min = min(r[3] for r in rows)
    for r in rows:
        r[3] = round(r[3] - t_min, 3)
    out = os.path.join(DATA, "tracelab_rounds.csv.gz")
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["provider", "session", "round_index", "t_s", "input_tokens", "observed_cached_tokens",
                    "output_tokens", "claude_cache_read", "claude_cache_write", "first_input_event", "model"])
        w.writerows(rows)
    print("tracelab rows", len(rows), "sessions", len(sess_ix))


def build_mooncake(src_dir):
    for name in ("conversation_trace", "toolagent_trace"):
        with open(os.path.join(src_dir, name + ".jsonl"), "rb") as fi, \
                gzip.open(os.path.join(DATA, "mooncake_" + name + ".jsonl.gz"), "wb") as fo:
            shutil.copyfileobj(fi, fo)
    print("mooncake copied")


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    build_tracelab(sys.argv[1])
    build_mooncake(sys.argv[2])
