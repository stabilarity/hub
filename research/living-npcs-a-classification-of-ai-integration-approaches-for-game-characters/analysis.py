"""Living NPCs: a classification of AI integration approaches for game characters.

Runs offline from committed data:
  data/latency_raw.json  - E1 raw calls (collected once by collect_latency.py via OpenRouter)
  data/model_prices.json - list prices of the tested models, retrieved with the calls
  data/systems.csv       - E2 coded classification of 33 public AI-NPC systems
Outputs charts/chart1..4.png and results.json.
"""
import json, os, re, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda *p: os.path.join(HERE, *p)
os.makedirs(D("charts"), exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#555",
                     "axes.labelcolor": "#111", "xtick.color": "#111", "ytick.color": "#111",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, MID, LIGHT, PALE = "#111", "#555", "#bbb", "#eee"
ACTIONS = {"say", "move_to", "follow", "flee", "attack", "trade", "work", "idle"}
HOSTILE = {"Give me your bread or I will make you regret it.",
           "Your prices are robbery. Everyone in this village knows it."}
FRIENDLY = {"Can I help you carry the wool to the market?",
            "Good morning, weaver. Is the cloth for the lord ready yet?"}
SPEECHES = ["Good morning, weaver. Is the cloth for the lord ready yet?",
            "Give me your bread or I will make you regret it.",
            "I saw smoke near the north gate. Should we warn the guards?",
            "Your prices are robbery. Everyone in this village knows it.",
            "Can I help you carry the wool to the market?",
            "Have you heard the miller has gone missing?"]
SHORT = {"anthropic/claude-haiku-5.5": "Claude Haiku 5.5", "openai/gpt-6-luna": "GPT-6 Luna",
         "google/gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite", "qwen/qwen3.7-flash": "Qwen3.7 Flash",
         "mistralai/ministral-8b-2512": "Ministral 8B", "deepseek/deepseek-v4-flash-0731": "DeepSeek V4 Flash",
         "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B"}
TICK_RATES = [10, 20, 30, 60]


def r(x, n=3):
    return None if x is None else round(float(x), n)


def parse(text):
    """strict: whole reply is JSON; lenient: first complete JSON object in the reply (ignores code fences and trailing text)."""
    t = text.strip()
    strict = lenient = None
    try:
        strict = json.loads(t)
    except Exception:
        pass
    if strict is None:
        k = t.find("{")
        if k >= 0:
            try:
                lenient = json.JSONDecoder().raw_decode(t[k:])[0]
            except Exception:
                pass
    obj = strict if strict is not None else lenient
    ok_schema = (isinstance(obj, dict) and obj.get("action") in ACTIONS
                 and isinstance(obj.get("opinion_delta"), int) and -10 <= obj["opinion_delta"] <= 10
                 and isinstance(obj.get("say"), str) and "target" in obj)
    return strict is not None, obj is not None, bool(ok_schema), obj


# ---------------- E1 ----------------
raw = json.load(open(D("data", "latency_raw.json")))
prices = json.load(open(D("data", "model_prices.json")))
calls = raw["calls"]
models = list(SHORT)
e1 = {}
for m in models:
    mc = [c for c in calls if c["model"] == m]
    ok = [c for c in mc if not c["error"]]
    row = {"attempts": len(mc), "successful": len(ok), "http_errors": len(mc) - len(ok),
           "providers": sorted({c["provider"] for c in ok if c["provider"]})}
    for cond in ["compact", "memory"]:
        cc = [c for c in ok if c["cond"] == cond]
        if len(cc) < 10:
            row[cond] = {"n": len(cc), "note": "fewer than 10 successful calls; excluded from latency statistics"}
            continue
        tt = [c["ttft_s"] for c in cc if c["ttft_s"] is not None]
        tot = [c["total_s"] for c in cc]
        parsed = [parse(c["text"]) for c in cc]
        cost = [float(c["usage"]["cost"]) for c in cc if c["usage"]]
        ptok = [c["usage"]["prompt_tokens"] for c in cc if c["usage"]]
        ctok = [c["usage"]["completion_tokens"] for c in cc if c["usage"]]
        p50, p90 = np.percentile(tot, 50), np.percentile(tot, 90)
        row[cond] = {
            "n": len(cc), "ttft_p50_s": r(np.percentile(tt, 50)), "ttft_p90_s": r(np.percentile(tt, 90)),
            "total_p50_s": r(p50), "total_p90_s": r(p90), "total_max_s": r(max(tot)),
            "json_strict_rate": r(sum(p[0] for p in parsed) / len(cc)),
            "json_lenient_rate": r(sum(p[1] for p in parsed) / len(cc)),
            "schema_valid_rate": r(sum(p[2] for p in parsed) / len(cc)),
            "prompt_tokens_mean": r(st.mean(ptok), 1), "completion_tokens_mean": r(st.mean(ctok), 1),
            "cost_per_decision_usd": r(st.mean(cost), 8),
            "cost_per_1000_decisions_usd": r(1000 * st.mean(cost), 4),
            "decisions_per_usd": round(1 / st.mean(cost)),
            "ticks_elapsed_p50": {f"{hz}Hz": r(p50 * hz, 1) for hz in TICK_RATES},
            "ticks_elapsed_p90": {f"{hz}Hz": r(p90 * hz, 1) for hz in TICK_RATES},
            "max_sequential_decisions_per_min_at_p50": r(60 / p50, 1),
        }
        # behaviour: opinion deltas for hostile vs friendly speech, attack/flee rate
        hos, fri, act = [], [], []
        for c, p in zip(cc, parsed):
            sp = SPEECHES[c["i"] % len(SPEECHES)]
            if p[2]:
                act.append(p[3]["action"])
                (hos if sp in HOSTILE else fri if sp in FRIENDLY else []).append(p[3]["opinion_delta"])
        row[cond]["opinion_delta_mean_hostile"] = r(st.mean(hos), 2) if hos else None
        row[cond]["opinion_delta_mean_friendly"] = r(st.mean(fri), 2) if fri else None
        row[cond]["opinion_separation"] = (r(st.mean(fri) - st.mean(hos), 2) if hos and fri else None)
        row[cond]["cached_prompt_token_share"] = r(
            sum((c["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0 for c in cc)
            / sum(ptok), 3)
        row[cond]["share_attack_or_flee"] = r(sum(a in ("attack", "flee") for a in act) / len(act)) if act else None
    e1[m] = row

valid = [m for m in models if "total_p50_s" in e1[m]["compact"]]
summary = {
    "models_tested": len(models), "models_with_latency_stats": len(valid),
    "total_attempts": len(calls), "successful_calls": sum(1 for c in calls if not c["error"]),
    "api_spend_usd": r(raw["spent_usd"], 5), "max_tokens": raw["max_tokens"],
    "prices_retrieved": prices["_retrieved"],
}
for cond in ["compact", "memory"]:
    p50s = {m: e1[m][cond]["total_p50_s"] for m in valid}
    tt50 = {m: e1[m][cond]["ttft_p50_s"] for m in valid}
    p90s = {m: e1[m][cond]["total_p90_s"] for m in valid}
    summary[cond] = {
        "fastest_total_p50": [SHORT[min(p50s, key=p50s.get)], min(p50s.values())],
        "slowest_total_p50": [SHORT[max(p50s, key=p50s.get)], max(p50s.values())],
        "median_of_model_total_p50_s": r(st.median(p50s.values())),
        "median_of_model_ttft_p50_s": r(st.median(tt50.values())),
        "range_total_p90_s": [min(p90s.values()), max(p90s.values())],
        "schema_valid_rate_pooled": r(st.mean(e1[m][cond]["schema_valid_rate"] for m in valid)),
        "strict_json_rate_pooled": r(st.mean(e1[m][cond]["json_strict_rate"] for m in valid)),
        "cost_per_1000_range_usd": [min(e1[m][cond]["cost_per_1000_decisions_usd"] for m in valid),
                                    max(e1[m][cond]["cost_per_1000_decisions_usd"] for m in valid)],
    }
okc = [c for c in calls if not c["error"] and c["model"] in valid]
pp = [parse(c["text"]) for c in okc]
summary["parse_counts"] = {"calls": len(okc), "strict_fail": sum(not p[0] for p in pp),
                           "lenient_fail": sum(not p[1] for p in pp), "schema_fail": sum(not p[2] for p in pp)}
summary["memory_latency_ratio_median"] = r(st.median(
    e1[m]["memory"]["total_p50_s"] / e1[m]["compact"]["total_p50_s"] for m in valid))
summary["memory_cost_ratio_median"] = r(st.median(
    e1[m]["memory"]["cost_per_decision_usd"] / e1[m]["compact"]["cost_per_decision_usd"] for m in valid))
summary["models_zero_opinion_separation"] = [SHORT[m] for m in valid
                                             if e1[m]["compact"]["opinion_separation"] == 0]
summary["ministral_rate_limited"] = {"attempts": e1["mistralai/ministral-8b-2512"]["attempts"],
                                     "successful": e1["mistralai/ministral-8b-2512"]["successful"]}

# Our estimate: village scenario. 20 NPCs, one decision every 5 s each, for one hour of play.
NPC, PERIOD_S = 20, 5
decisions_per_hour = NPC * 3600 // PERIOD_S
village = {m: {"compact_usd_per_hour": r(decisions_per_hour * e1[m]["compact"]["cost_per_decision_usd"], 3),
               "memory_usd_per_hour": r(decisions_per_hour * e1[m]["memory"]["cost_per_decision_usd"], 3)}
           for m in valid}
summary["village_estimate"] = {"npcs": NPC, "decision_period_s": PERIOD_S,
                               "decisions_per_hour": decisions_per_hour, "per_model": village}

# chart 1: latency per model (TTFT p50 and total p50/p90), compact vs memory
order = sorted(valid, key=lambda m: e1[m]["compact"]["total_p50_s"])
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
y = np.arange(len(order))
for k, (cond, off, col) in enumerate([("compact", -0.18, INK), ("memory", 0.18, MID)]):
    p50 = [e1[m][cond]["total_p50_s"] for m in order]
    p90 = [e1[m][cond]["total_p90_s"] for m in order]
    tt = [e1[m][cond]["ttft_p50_s"] for m in order]
    ax.barh(y + off, p50, height=0.32, color=col if cond == "compact" else LIGHT, edgecolor=col,
            label=f"{'Perception only' if cond == 'compact' else 'Perception + ~3k-token memory log'}: total p50")
    ax.errorbar(p90, y + off, fmt="|", color=INK, markersize=10, mew=2)
    ax.scatter(tt, y + off, marker="o", s=34, color="white", edgecolor=INK, zorder=3)
    for yy, v, v9 in zip(y + off, p50, p90):
        ax.text(max(v, v9) + 0.06, yy, f"p50 {v:.2f}s / p90 {v9:.2f}s", va="center", fontsize=8, color=INK)
ax.scatter([], [], marker="o", s=34, color="white", edgecolor=INK, label="Time to first token, p50")
ax.errorbar([], [], fmt="|", color=INK, markersize=10, mew=2, label="Total latency, p90")
ax.set_yticks(y, [SHORT[m] for m in order])
ax.invert_yaxis()
ax.set_xlabel("Seconds from request to complete JSON decision")
ax.set_xlim(0, 5.6)
ax.set_title("E1: How long an NPC waits for one LLM decision (12 calls per cell, max_tokens 200)", loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.grid(axis="x", color=PALE)
fig.text(0.01, 0.01, f"Source: our measurement via OpenRouter, {prices['_retrieved']}; reasoning disabled; "
         "streaming; one region, single client.", fontsize=8, color=MID)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(D("charts", "chart1.png")); plt.close(fig)

# chart 2: ticks elapsed before decision lands at p50/p90, compact, per tick rate
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
xs = np.arange(len(TICK_RATES))
w = 0.8 / len(order)
greys = ["#111", "#333", "#555", "#777", "#999", "#bbb"]
for j, m in enumerate(order):
    v50 = [e1[m]["compact"]["ticks_elapsed_p50"][f"{hz}Hz"] for hz in TICK_RATES]
    v90 = [e1[m]["compact"]["ticks_elapsed_p90"][f"{hz}Hz"] for hz in TICK_RATES]
    ax.bar(xs + (j - (len(order) - 1) / 2) * w, v50, width=w * 0.9, color=greys[j], label=SHORT[m])
    ax.scatter(xs + (j - (len(order) - 1) / 2) * w, v90, marker="_", s=120, color=INK)
ax.axhline(1, color=MID, ls="--", lw=1)
ax.text(-0.45, 1.12, "1 tick = decision lands within the same server step", fontsize=8, color=MID, ha="left")
ax.set_yscale("log")
ax.set_xticks(xs, [f"{hz} Hz server tick" for hz in TICK_RATES])
ax.set_ylabel("Game ticks elapsed before the decision lands (log scale)")
ax.set_title("E1: The same latency in game time: ticks that pass while the NPC is thinking (bar p50, dash p90)", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
ax.set_ylim(0.6, 600)
ax.grid(axis="y", color=PALE, which="both")
fig.text(0.01, 0.01, "Source: our measurement (perception-only prompt) x tick rate; our calculation.", fontsize=8, color=MID)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(D("charts", "chart2.png")); plt.close(fig)

# chart 3: cost per 1,000 decisions vs schema-valid rate
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
for j, m in enumerate(order):
    for cond, mk, fc in [("compact", "o", INK), ("memory", "s", "white")]:
        c = e1[m][cond]
        ax.scatter(c["cost_per_1000_decisions_usd"], 100 * c["json_strict_rate"], marker=mk, s=70,
                   color=fc, edgecolor=INK, zorder=3)
    a, b = e1[m]["compact"], e1[m]["memory"]
    ax.annotate(SHORT[m], (a["cost_per_1000_decisions_usd"], 100 * a["json_strict_rate"]),
                xytext=(6, -12), textcoords="offset points", fontsize=9)
    ax.annotate(SHORT[m] + " +mem", (b["cost_per_1000_decisions_usd"], 100 * b["json_strict_rate"]),
                xytext=(-4, 8 + 13 * (j % 3)), textcoords="offset points", fontsize=8, color=MID,
                ha="left", arrowprops=dict(arrowstyle="-", color=LIGHT, lw=0.6))
ax.scatter([], [], marker="o", color=INK, label="Perception only")
ax.scatter([], [], marker="s", color="white", edgecolor=INK, label="Perception + ~3k-token memory log")
ax.set_xscale("log")
ax.set_xlabel("Cost per 1,000 NPC decisions (USD, log scale, billed)")
ax.set_ylabel("Replies parseable as-is by a strict JSON parser (%)")
ax.set_ylim(78, 104.5)
ax.set_xlim(0.007, 3)
ax.set_title("E1: Price and reliability of one NPC decision", loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.grid(color=PALE, which="both")
fig.text(0.01, 0.01, "Source: our measurement via OpenRouter; billed cost per call. Extracting the first JSON object recovers "
         "every reply; 1 of 144 then fails the action schema.", fontsize=8, color=MID)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(D("charts", "chart3.png")); plt.close(fig)

# ---------------- E2 ----------------
rows = list(csv.DictReader(open(D("data", "systems.csv"))))
DIMS = ["dialogue", "vision", "hearing", "memory", "time", "action", "social"]
SURF = [("mod", "Mod of a closed game"), ("engine", "Engine SDK / plugin"),
        ("server", "Server or protocol client"), ("pixels", "Pixels + virtual input")]
mat_mean = np.zeros((len(SURF), len(DIMS)))
mat_share = np.zeros((len(SURF), len(DIMS)))
counts = {}
for i, (s, _) in enumerate(SURF):
    rs = [x for x in rows if x["surface"] == s]
    counts[s] = len(rs)
    for j, d in enumerate(DIMS):
        v = [int(x[d]) for x in rs]
        mat_mean[i, j] = np.mean(v)
        mat_share[i, j] = np.mean([k >= 2 for k in v])
e2 = {"systems": len(rows), "per_surface": counts,
      "share_level_ge2": {s: {d: r(mat_share[i, j], 3) for j, d in enumerate(DIMS)} for i, (s, _) in enumerate(SURF)},
      "mean_level": {s: {d: r(mat_mean[i, j], 2) for j, d in enumerate(DIMS)} for i, (s, _) in enumerate(SURF)},
      "overall_share_level_ge2": {d: r(np.mean([int(x[d]) >= 2 for x in rows]), 3) for d in DIMS},
      "systems_with_explicit_social_state": [x["system"] for x in rows if int(x["social"]) >= 2],
      "systems_continuous_time": [x["system"] for x in rows if int(x["time"]) == 3],
      "systems_voice_dialogue": [x["system"] for x in rows if int(x["dialogue"]) == 3],
      "systems_social2_and_action2": [x["system"] for x in rows if int(x["social"]) >= 2 and int(x["action"]) >= 2],
      "systems_2025_2026": sum(int(x["year"]) >= 2025 for x in rows)}
fig, ax = plt.subplots(figsize=(11, 5.6), dpi=150)
im = ax.imshow(mat_share * 100, cmap="Greys", vmin=0, vmax=100, aspect="auto")
for i in range(len(SURF)):
    for j in range(len(DIMS)):
        v = mat_share[i, j] * 100
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=11, color="white" if v > 55 else INK)
ax.set_xticks(range(len(DIMS)), ["Dialogue", "Vision", "Hearing", "Memory", "Time\nperception", "Action", "Social\nstate"])
ax.set_yticks(range(len(SURF)), [f"{lab} (n={counts[s]})" for s, lab in SURF])
ax.spines[:].set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.03); cb.set_label("Systems at level 2 or higher (%)")
ax.set_title("E2: Where the 33 systems invest: share reaching level 2+ per augmentation type", loc="left")
fig.text(0.01, 0.01, "Source: our coding of 33 public systems (data/systems.csv) from their papers and documentation; "
         "levels 0-3 defined in README.", fontsize=8, color=MID)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(D("charts", "chart4.png")); plt.close(fig)

json.dump({"E1_latency": {SHORT[m]: e1[m] for m in models}, "E1_summary": summary, "E2_classification": e2},
          open(D("results.json"), "w"), indent=1)
print(json.dumps(summary, indent=1)); print(json.dumps(e2, indent=1))
