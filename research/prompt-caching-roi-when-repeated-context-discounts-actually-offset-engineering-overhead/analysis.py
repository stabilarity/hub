"""Prompt Caching ROI: when repeated-context discounts actually offset engineering overhead.

Runs offline and deterministically from the files in data/:
  python3 analysis.py
Writes charts/chart1.png ... chart4.png and results.json.

RQ1  break-even reuse per provider (reads within TTL, Poisson arrival-rate threshold, storage-billed caches)
RQ2  how much of the list-price discount survives real arrival timing (Poisson model, TraceLab coding-agent
     sessions, Mooncake conversation and tool-agent traces), and which TTL is cheapest
RQ3  engineering-overhead payback: the monthly input-token volume at which caching work pays back in 3 months
"""
import csv
import gzip
import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CHARTS = os.path.join(HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)
ACCESSED = "2026-10-21"

# site palette (greyscale) with line styles and markers for identity
INK, INK2, MID, LIGHT, GRID = "#000000", "#111111", "#555555", "#bbbbbb", "#eeeeee"
plt.rcParams.update({
    "figure.figsize": (10.67, 6.0), "figure.dpi": 150, "savefig.dpi": 150,
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MID, "axes.labelcolor": INK2,
    "xtick.color": MID, "ytick.color": MID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff", "legend.frameon": False,
})


def r4(x):
    return float(round(float(x), 4))


def source_line(fig, text):
    fig.text(0.01, 0.01, text, fontsize=8, color=MID, ha="left", va="bottom")


# ---------------------------------------------------------------- prices
prices = []
with open(os.path.join(DATA, "prices.csv")) as fh:
    for row in csv.DictReader(fh):
        for k in ("input_usd_mtok", "write_usd_mtok", "read_usd_mtok", "output_usd_mtok", "ttl_min", "min_tokens"):
            row[k] = float(row[k])
        row["storage_usd_mtok_hour"] = float(row["storage_usd_mtok_hour"]) if row["storage_usd_mtok_hour"] else None
        row["w"] = row["write_usd_mtok"] / row["input_usd_mtok"]
        row["r"] = row["read_usd_mtok"] / row["input_usd_mtok"]
        prices.append(row)


def policy(model, mode):
    for p in prices:
        if p["model"] == model and p["mode"] == mode:
            return p
    raise KeyError((model, mode))


# ---------------------------------------------------------------- RQ1: break-even
def breakeven_reads(w, r):
    """Smallest integer number of cache reads k with w + k*r <= 1 + k (one write, k reads vs k+1 uncached)."""
    if w <= 1.0:
        return 0  # no premium: the first read is pure gain (and a miss costs nothing extra)
    return int(math.ceil((w - 1.0) / (1.0 - r) - 1e-12))


def poisson_hit(lam_per_min, ttl):
    return 1.0 - math.exp(-lam_per_min * ttl)


def poisson_cost(lam_per_min, w, r, ttl):
    """Expected billed multiple of the base input price per request for the cached prefix, with
    Poisson arrivals at rate lam and a TTL refreshed by every request (write on a miss, read on a hit)."""
    h = poisson_hit(lam_per_min, ttl)
    return h * r + (1.0 - h) * w


def poisson_breakeven_interarrival(w, r, ttl):
    """Mean inter-arrival time (minutes) below which caching beats no caching: h > (w-1)/(w-r)."""
    if w <= 1.0:
        return float("inf")
    h_star = (w - 1.0) / (w - r)
    lam = -math.log(1.0 - h_star) / ttl
    return 1.0 / lam


rq1_rows = []
for p in prices:
    k = breakeven_reads(p["w"], p["r"])
    ia = poisson_breakeven_interarrival(p["w"], p["r"], p["ttl_min"])
    row = {
        "provider": p["provider"], "model": p["model"], "mode": p["mode"],
        "write_multiplier": r4(p["w"]), "read_multiplier": r4(p["r"]), "ttl_min": p["ttl_min"],
        "min_tokens": int(p["min_tokens"]), "breakeven_reads": k,
        "max_prefix_saving_pct": r4(100 * (1 - p["r"])),
        "saving_at_10_reuses_pct": r4(100 * (1 - (p["w"] + 10 * p["r"]) / 11.0)),
        # -1 encodes "no threshold: never worse than uncached" (no write premium)
        "poisson_breakeven_mean_interarrival_min": r4(ia) if math.isfinite(ia) else -1.0,
    }
    if p["storage_usd_mtok_hour"]:
        # explicit, storage-billed cache: hits per hour needed so that read savings cover storage
        row["storage_breakeven_hits_per_hour"] = r4(p["storage_usd_mtok_hour"] /
                                                   (p["input_usd_mtok"] - p["read_usd_mtok"]))
    rq1_rows.append(row)

# Poisson curves for chart 1
chart1_policies = [
    ("Claude Sonnet 5.5", "explicit-5m", "Claude Sonnet 5.5, 5-min write 1.25x", "-", "o", INK),
    ("Claude Sonnet 5.5", "explicit-1h", "Claude Sonnet 5.5, 1-hour write 2x", "--", "s", INK),
    ("gpt-6.1-sol", "explicit-30m", "gpt-6.1-sol, 30-min write 1.25x", "-", "^", MID),
    ("gpt-6.1-sol", "explicit-12h", "gpt-6.1-sol, 12-hour write 2.5x", "--", "v", MID),
    ("gpt-5.5", "implicit", "gpt-5.5 implicit, no write premium (5 min)", ":", "D", INK2),
]
ia_grid = np.logspace(math.log10(0.1), math.log10(24 * 60), 400)  # minutes between requests per prefix
poisson_curves = {}
for model, mode, label, *_ in chart1_policies:
    p = policy(model, mode)
    poisson_curves[label] = [100 * (1 - poisson_cost(1 / ia, p["w"], p["r"], p["ttl_min"])) for ia in ia_grid]

# best explicit Anthropic TTL by inter-arrival time
s5, s60 = policy("Claude Sonnet 5.5", "explicit-5m"), policy("Claude Sonnet 5.5", "explicit-1h")
o30, o720 = policy("gpt-6.1-sol", "explicit-30m"), policy("gpt-6.1-sol", "explicit-12h")


def crossover(pa, pb):
    """Mean inter-arrival (min) above which the shorter-TTL policy pa becomes cheaper than the longer-TTL pb.
    With a refreshed TTL the longer TTL is cheaper (or equal) for frequent reuse; it loses only once reuse is
    so rare that most requests miss and pay its larger write premium."""
    f = lambda ia: poisson_cost(1 / ia, pb["w"], pb["r"], pb["ttl_min"]) - poisson_cost(1 / ia, pa["w"], pa["r"], pa["ttl_min"])
    lo, hi = 1.0, 1e6  # f(lo) < 0 (long TTL cheaper), f(hi) > 0
    assert f(lo) < 0 < f(hi)
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


poisson_summary = {
    "sonnet55_5m_vs_1h_crossover_interarrival_min": r4(crossover(s5, s60)),
    "gpt61sol_30m_vs_12h_crossover_interarrival_min": r4(crossover(o30, o720)),
    "sonnet55_5m_saving_at_1min_pct": r4(100 * (1 - poisson_cost(1.0, s5["w"], s5["r"], 5))),
    "sonnet55_5m_saving_at_5min_pct": r4(100 * (1 - poisson_cost(1 / 5, s5["w"], s5["r"], 5))),
    "sonnet55_5m_saving_at_15min_pct": r4(100 * (1 - poisson_cost(1 / 15, s5["w"], s5["r"], 5))),
    "sonnet55_1h_saving_at_15min_pct": r4(100 * (1 - poisson_cost(1 / 15, s60["w"], s60["r"], 60))),
    "sonnet55_1h_saving_at_60min_pct": r4(100 * (1 - poisson_cost(1 / 60, s60["w"], s60["r"], 60))),
    "sonnet55_5m_saving_at_60min_pct": r4(100 * (1 - poisson_cost(1 / 60, s5["w"], s5["r"], 5))),
    "sonnet55_5m_worst_case_loss_pct": r4(100 * (s5["w"] - 1)),
    "gpt61sol_12h_worst_case_loss_pct": r4(100 * (o720["w"] - 1)),
}

# ---------------------------------------------------------------- RQ2: trace replay
REPLAY = [
    # key, model, mode, label
    ("A5", "Claude Sonnet 5.5", "explicit-5m", "Sonnet 5.5 5-min"),
    ("A60", "Claude Sonnet 5.5", "explicit-1h", "Sonnet 5.5 1-hour"),
    ("A5std", "Claude Sonnet 4.6", "explicit-5m", "Sonnet 4.6 5-min (0.1x read)"),
    ("O30", "gpt-6.1-sol", "explicit-30m", "gpt-6.1-sol 30-min"),
    ("O720", "gpt-6.1-sol", "explicit-12h", "gpt-6.1-sol 12-hour"),
    ("I5", "gpt-5.5", "implicit", "gpt-5.5 implicit"),
    ("I4o", "gpt-4o", "implicit", "gpt-4o implicit (0.5x read)"),
    ("G", "Gemini 3.1 Pro Preview", "implicit", "Gemini 3.1 Pro implicit"),
    ("D", "deepseek-flash", "implicit-disk", "DeepSeek flash disk cache"),
]

# TraceLab
sessions = defaultdict(list)
with gzip.open(os.path.join(DATA, "tracelab_rounds.csv.gz"), "rt") as fh:
    for row in csv.DictReader(fh):
        sessions[(row["provider"], int(row["session"]))].append((
            float(row["t_s"]), int(row["round_index"]), int(row["input_tokens"]), int(row["observed_cached_tokens"]),
            int(row["output_tokens"]), row["first_input_event"]))
for k in sessions:
    sessions[k].sort()


def tracelab_arrays(provider):
    I, R, G, O, OBS, FIRST = [], [], [], [], [], []
    for (prov, _), rs in sessions.items():
        if prov != provider:
            continue
        prev = None
        for t, _ri, inp, obs, out, fie in rs:
            I.append(inp); O.append(out); OBS.append(obs); FIRST.append(fie)
            if prev is None:
                R.append(0); G.append(float("inf"))
            else:
                # append-only context: the previous request's prompt is the reusable prefix;
                # a shorter prompt (compaction or a new context) is treated as no reuse
                R.append(prev[2] if inp >= prev[2] else 0)
                G.append((t - prev[0]) / 60.0)
            prev = (t, _ri, inp)
    return (np.array(I, float), np.array(R, float), np.array(G, float), np.array(O, float),
            np.array(OBS, float), np.array(FIRST))


def replay_cost(I, R, G, w, r, ttl, min_tokens):
    hit = (G <= ttl) & (R > 0) & (I >= min_tokens)
    cacheable = I >= min_tokens
    cost = np.where(hit, R * r + (I - R) * w, np.where(cacheable, I * w, I))
    return cost, hit


def oracle_two_ttl(I, R, G, short, long_, min_tokens):
    """Per-request TTL choice with hindsight (dynamic programme over each session's sequence):
    a lower bound on what any TTL-selection heuristic between the two TTLs can achieve."""
    total = 0.0
    n = len(I)
    i = 0
    starts = np.where(~np.isfinite(G))[0].tolist() + [n]
    for a, b in zip(starts[:-1], starts[1:]):
        best = {0: 0.0, 1: 0.0}  # state = TTL chosen for previous write (0 short, 1 long)
        first = True
        for j in range(a, b):
            new = {}
            for choice, pol in ((0, short), (1, long_)):
                cands = []
                for prev_state in ((0, 1) if not first else (0,)):
                    ttl_prev = (short if prev_state == 0 else long_)["ttl_min"]
                    hit = (not first) and G[j] <= ttl_prev and R[j] > 0 and I[j] >= min_tokens
                    if hit:
                        c = R[j] * pol["r"] + (I[j] - R[j]) * pol["w"]
                    elif I[j] >= min_tokens:
                        c = I[j] * pol["w"]
                    else:
                        c = I[j]
                    cands.append(best[prev_state] + c)
                new[choice] = min(cands)
            best = new
            first = False
        total += min(best.values())
        i = b
    return total


trace_results = {}
gap_stats = {}
for provider in ("claude", "codex"):
    I, R, G, O, OBS, FIRST = tracelab_arrays(provider)
    base = I.sum()
    fin = np.isfinite(G)
    g = G[fin]
    tok_w = I[fin]
    gs = {
        "rounds": int(len(I)), "sessions": int(sum(1 for k in sessions if k[0] == provider)),
        "mean_input_tokens": r4(I.mean()), "median_input_tokens": r4(np.median(I)),
        "gap_p50_min": r4(np.percentile(g, 50)), "gap_p90_min": r4(np.percentile(g, 90)),
        "gap_p95_min": r4(np.percentile(g, 95)), "gap_p99_min": r4(np.percentile(g, 99)),
        "share_gaps_over_5min_pct": r4(100 * (g > 5).mean()),
        "share_gaps_over_30min_pct": r4(100 * (g > 30).mean()),
        "share_gaps_over_60min_pct": r4(100 * (g > 60).mean()),
        "token_weighted_share_over_5min_pct": r4(100 * tok_w[g > 5].sum() / tok_w.sum()),
        "token_weighted_share_over_60min_pct": r4(100 * tok_w[g > 60].sum() / tok_w.sum()),
        "share_gaps_over_5min_after_user_message_pct": r4(100 * ((g > 5) & (FIRST[fin] == "user_message")).sum() / max(1, (g > 5).sum())),
        "share_rounds_below_1024_tokens_pct": r4(100 * (I < 1024).mean()),
        "share_rounds_below_4096_tokens_pct": r4(100 * (I < 4096).mean()),
        "reusable_prefix_share_of_input_pct": r4(100 * R.sum() / base),
        "observed_provider_cached_share_pct": r4(100 * OBS.sum() / base),
    }
    gap_stats[provider] = gs
    res = {}
    for key, model, mode, label in REPLAY:
        p = policy(model, mode)
        cost, hit = replay_cost(I, R, G, p["w"], p["r"], p["ttl_min"], p["min_tokens"])
        out_mult = p["output_usd_mtok"] / p["input_usd_mtok"]
        bill_nocache = base + out_mult * O.sum()
        bill_cache = cost.sum() + out_mult * O.sum()
        res[key] = {
            "label": label, "input_saving_pct": r4(100 * (1 - cost.sum() / base)),
            "hit_token_share_pct": r4(100 * (R * hit).sum() / base),
            "total_bill_saving_pct": r4(100 * (1 - bill_cache / bill_nocache)),
            "list_price_ceiling_pct": r4(100 * (1 - p["r"])),
        }
    # hindsight TTL choice between 5 min and 1 h on Sonnet 5.5
    orc = oracle_two_ttl(I, R, G, s5, s60, s5["min_tokens"])
    res["A_oracle"] = {"label": "Sonnet 5.5 hindsight 5-min/1-hour", "input_saving_pct": r4(100 * (1 - orc / base))}
    # replay validation for Claude Code sessions: replayed hit share vs provider-reported cache reads
    trace_results["tracelab_" + provider] = res

# Mooncake: service-wide block cache (512-token blocks), TTL refreshed on every touch
BLOCK = 512


def mooncake_replay(name):
    reqs = [json.loads(l) for l in gzip.open(os.path.join(DATA, "mooncake_" + name + ".jsonl.gz"), "rt")]
    reqs.sort(key=lambda d: d["timestamp"])
    ttls = sorted({p_["ttl_min"] for p_ in prices} | {1, 10})
    hit_tokens = {}
    for ttl in ttls:
        last = {}
        hits = np.zeros(len(reqs))
        lim = ttl * 60000.0
        for i, d in enumerate(reqs):
            t = d["timestamp"]
            k = 0
            for h in d["hash_ids"]:
                lt = last.get(h)
                if lt is not None and t - lt <= lim:
                    k += 1
                else:
                    break
            hits[i] = min(k * BLOCK, d["input_length"])
            for h in d["hash_ids"]:
                last[h] = t
        hit_tokens[ttl] = hits
    I = np.array([d["input_length"] for d in reqs], float)
    O = np.array([d["output_length"] for d in reqs], float)
    ts = np.array([d["timestamp"] for d in reqs], float)
    return I, O, ts, hit_tokens


mooncake = {}
for name in ("conversation_trace", "toolagent_trace"):
    I, O, ts, hit_tokens = mooncake_replay(name)
    base = I.sum()
    res = {"requests": int(len(I)), "mean_input_tokens": r4(I.mean()),
           "hit_share_by_ttl_pct": {str(int(t)): r4(100 * h.sum() / base) for t, h in sorted(hit_tokens.items())}}
    for key, model, mode, label in REPLAY:
        p = policy(model, mode)
        h = hit_tokens[p["ttl_min"]]
        cacheable = I >= p["min_tokens"]
        cost = np.where(cacheable, h * p["r"] + (I - h) * p["w"], I)
        res[key] = {"label": label, "input_saving_pct": r4(100 * (1 - cost.sum() / base)),
                    "list_price_ceiling_pct": r4(100 * (1 - p["r"]))}
    mooncake[name] = res

# ---------------------------------------------------------------- RQ3: engineering payback (our assumptions)
BLS_MEDIAN_WAGE = 135980.0  # US software developers, May 2025 (BLS OOH)
LOADED_MULT = 1.4           # our assumption: benefits and overhead
WORKDAYS = 230              # our assumption
DAY_RATE = BLS_MEDIAN_WAGE * LOADED_MULT / WORKDAYS
HORIZON = 3.0               # months to payback
TIERS = {
    # name: (one-off engineer-days, recurring engineer-days per month) -- our assumptions
    "switch-on": (0.5, 0.0),
    "prefix-stable": (5.0, 1.0),
    "full-programme": (20.0, 2.0),
}


def required_monthly_saving(tier, day_rate=DAY_RATE, horizon=HORIZON):
    d0, dm = TIERS[tier]
    return d0 * day_rate / horizon + dm * day_rate


def required_input_tokens_bn(tier, price_in, saving_frac, **kw):
    """Monthly input tokens (billions) at which the saving covers the tier's cost within the horizon."""
    return required_monthly_saving(tier, **kw) / (price_in * saving_frac) / 1000.0


# realised input saving per model, from the Claude Code TraceLab replay (the agent workload);
# models without their own replay row use the policy with the same multipliers
s_claude = trace_results["tracelab_claude"]
payback_models = [
    ("Claude Opus 5.5", "explicit-5m", s_claude["A5"]["input_saving_pct"]),
    ("Claude Sonnet 5.5", "explicit-5m", s_claude["A5"]["input_saving_pct"]),
    ("Claude Haiku 5.5", "explicit-5m", s_claude["A5std"]["input_saving_pct"]),
    ("gpt-6-astra", "explicit-30m", s_claude["A5std"]["input_saving_pct"]),
    ("gpt-6.1-sol", "explicit-30m", s_claude["O30"]["input_saving_pct"]),
    ("gpt-5.6-luna", "explicit-30m", s_claude["A5std"]["input_saving_pct"]),
    ("Gemini 3.1 Pro Preview", "implicit", s_claude["G"]["input_saving_pct"]),
    ("deepseek-flash", "implicit-disk", s_claude["D"]["input_saving_pct"]),
]
# gpt-6-astra and gpt-5.6-luna use 1.25x write / 0.1x read like Sonnet 4.6 but with a 30-min TTL;
# recompute their replay saving directly for accuracy
Ic, Rc, Gc, Oc, _, _ = tracelab_arrays("claude")
for idx, (m, mode, _s) in enumerate(payback_models):
    p = policy(m, mode)
    cost, _h = replay_cost(Ic, Rc, Gc, p["w"], p["r"], p["ttl_min"], p["min_tokens"])
    payback_models[idx] = (m, mode, 100 * (1 - cost.sum() / Ic.sum()))

payback = {"day_rate_usd": r4(DAY_RATE), "horizon_months": HORIZON,
           "tiers": {k: {"one_off_days": v[0], "monthly_days": v[1],
                         "required_monthly_saving_usd": r4(required_monthly_saving(k))} for k, v in TIERS.items()},
           "models": []}
for m, mode, s in payback_models:
    p = policy(m, mode)
    entry = {"model": m, "mode": mode, "input_usd_mtok": p["input_usd_mtok"], "replay_input_saving_pct": r4(s)}
    for tier in TIERS:
        entry["required_input_btok_month_" + tier] = r4(required_input_tokens_bn(tier, p["input_usd_mtok"], s / 100))
    # requests per day needed for 'prefix-stable' payback at three prefix lengths (prefix-dominated prompts)
    for P in (2000, 10000, 50000):
        need_tok = required_input_tokens_bn("prefix-stable", p["input_usd_mtok"], s / 100) * 1e9
        entry["requests_per_day_prefix_%d" % P] = r4(need_tok / P / 30.0)
    entry["requests_per_month_at_tracelab_claude_mean_prompt"] = r4(
        required_input_tokens_bn("prefix-stable", p["input_usd_mtok"], s / 100) * 1e9 / gap_stats["claude"]["mean_input_tokens"])
    payback["models"].append(entry)

# sensitivity of the Sonnet 5.5 'prefix-stable' threshold
sens = {}
s_son = payback["models"][1]["replay_input_saving_pct"] / 100
for name, kw in {"base": {}, "day_rate_x0.5": {"day_rate": DAY_RATE * 0.5}, "day_rate_x1.5": {"day_rate": DAY_RATE * 1.5},
                 "horizon_1m": {"horizon": 1.0}, "horizon_12m": {"horizon": 12.0}}.items():
    sens[name] = r4(required_input_tokens_bn("prefix-stable", 2.0, s_son, **kw))
sens["one_off_20d_monthly_2d"] = r4(required_input_tokens_bn("full-programme", 2.0, s_son))
sens["hit_rate_halved_saving_pct"] = r4(100 * s_son / 2)
sens["hit_rate_halved_btok"] = r4(required_input_tokens_bn("prefix-stable", 2.0, s_son / 2))
payback["sonnet55_prefix_stable_sensitivity_btok_month"] = sens

# ---------------------------------------------------------------- charts
# Chart 1: Poisson model
fig, ax = plt.subplots()
for model, mode, label, ls, mk, col in chart1_policies:
    y = poisson_curves[label]
    ax.plot(ia_grid, y, ls=ls, color=col, lw=2, label=label, marker=mk, markevery=40, ms=6)
ax.axhline(0, color=MID, lw=1)
ax.set_xscale("log")
ax.set_xlabel("Mean time between requests that share the prefix (minutes, log scale)")
ax.set_ylabel("Saving on the cached prefix vs no caching (%)")
ax.set_title("Chart 1. Poisson arrivals: net saving on the prefix by TTL and write premium")
ax.set_ylim(-160, 100)
for x in (5, 30, 60, 720):
    ax.axvline(x, color=LIGHT, lw=1, ls=":")
    ax.text(x * 1.04, -152, {5: "5 min", 30: "30 min", 60: "1 h", 720: "12 h"}[x], fontsize=8, color=MID, ha="left")
ax.legend(loc="lower left", fontsize=9, bbox_to_anchor=(0.0, 0.06))
source_line(fig, "Source: list prices accessed %s (platform.claude.com, developers.openai.com); TTL refreshed by every request; our model." % ACCESSED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(CHARTS, "chart1.png"))
plt.close(fig)

# Chart 2: TraceLab inter-request gap CDF
fig, ax = plt.subplots()
for provider, ls, col, lab in (("claude", "-", INK, "Claude Code sessions"), ("codex", "--", MID, "Codex sessions")):
    _, _, G, _, _, _ = tracelab_arrays(provider)
    g = np.sort(G[np.isfinite(G)])
    g = np.clip(g, 1e-3, None)
    y = np.arange(1, len(g) + 1) / len(g) * 100
    ax.plot(g, y, ls=ls, color=col, lw=2, label=lab)
for x, t in ((5, "5 min"), (30, "30 min"), (60, "1 h")):
    ax.axvline(x, color=LIGHT, lw=1, ls=":")
    ax.text(x * 1.05, 40, t, fontsize=9, color=MID)
ax.set_xscale("log")
ax.set_xlim(0.005, 24 * 60)
ax.set_xlabel("Time since the previous request in the same session (minutes, log scale)")
ax.set_ylabel("Share of requests (%)")
ax.set_title("Chart 2. Coding-agent requests arrive in bursts: gap distribution vs cache TTLs")
ax.legend(loc="lower right")
source_line(fig, "Source: TraceLab v0.0.1 (CC BY 4.0), %d Claude Code and %d Codex requests; our calculation."
            % (gap_stats["claude"]["rounds"], gap_stats["codex"]["rounds"]))
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(CHARTS, "chart2.png"))
plt.close(fig)

# Chart 3: realised input saving vs list-price ceiling across workloads
keys = ["A5", "A60", "O30", "I5", "G", "D", "I4o"]
workloads = [("Claude Code (TraceLab)", trace_results["tracelab_claude"]),
             ("Codex (TraceLab)", trace_results["tracelab_codex"]),
             ("Kimi conversation (Mooncake)", mooncake["conversation_trace"]),
             ("Kimi tool-agent (Mooncake)", mooncake["toolagent_trace"])]
fig, ax = plt.subplots(figsize=(10.67, 6.0))
x = np.arange(len(keys))
width = 0.2
shades = [INK, MID, LIGHT, "#dddddd"]
hatches = ["", "", "//", ".."]
for i, (wl, res) in enumerate(workloads):
    vals = [res[k]["input_saving_pct"] for k in keys]
    ax.bar(x + (i - 1.5) * width, vals, width - 0.02, color=shades[i], edgecolor=MID, hatch=hatches[i], label=wl, linewidth=0.6)
ceil_vals = [trace_results["tracelab_claude"][k]["list_price_ceiling_pct"] for k in keys]
ax.scatter(x, ceil_vals, marker="_", s=900, color=INK, linewidths=2, label="List-price ceiling (1 - read multiplier)", zorder=5)
ax.set_xticks(x)
short = {"A5": "Sonnet 5.5\n5-min", "A60": "Sonnet 5.5\n1-hour", "O30": "gpt-6.1-sol\n30-min", "I5": "gpt-5.5\nimplicit",
         "G": "Gemini 3.1 Pro\nimplicit", "D": "DeepSeek flash\ndisk cache", "I4o": "gpt-4o implicit\n(0.5x read)"}
ax.set_xticklabels([short[k] for k in keys], fontsize=9)
ax.axhline(0, color=MID, lw=1)
for i, (wl, res) in enumerate(workloads):
    for j, k in enumerate(keys):
        v = res[k]["input_saving_pct"]
        if v < 0:
            ax.text(x[j] + (i - 1.5) * width, v - 2, "%.1f" % v, ha="center", va="top", fontsize=7.5, color=INK2)
ax.set_ylabel("Input-cost saving vs no caching (%)")
ax.set_ylim(-45, 105)
ax.set_title("Chart 3. Realised input saving on four real traces vs the list-price ceiling", pad=34)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5, ncol=5, handlelength=2.2, columnspacing=1.2)
source_line(fig, "Sources: TraceLab v0.0.1; Mooncake FAST'25 traces; prices accessed %s; replay assumptions in README; our calculation." % ACCESSED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(CHARTS, "chart3.png"))
plt.close(fig)

# Chart 4: payback threshold vs input price
fig, ax = plt.subplots()
pgrid = np.logspace(math.log10(0.05), math.log10(20), 200)
s_ref = s_son
for tier, ls in (("switch-on", ":"), ("prefix-stable", "-"), ("full-programme", "--")):
    d0, dm = TIERS[tier]
    ax.plot(pgrid, [required_input_tokens_bn(tier, pp, s_ref) for pp in pgrid], ls=ls, color=INK if tier == "prefix-stable" else MID,
            lw=2, label="%s: %.1f days once + %.2g days/month" % (tier, d0, dm))
at2 = [m["model"] for m in payback["models"] if m["input_usd_mtok"] == 2.0]
for m in payback["models"]:
    yv = m["required_input_btok_month_prefix-stable"]
    ax.plot(m["input_usd_mtok"], yv, "o", color=INK, ms=7, markeredgecolor="#ffffff")
    if m["input_usd_mtok"] == 2.0:
        if m["model"] == at2[0]:
            ax.annotate(", ".join(at2), (2.0, yv), textcoords="offset points", xytext=(-215, 12), fontsize=8, color=INK2)
        continue
    ax.annotate(m["model"], (m["input_usd_mtok"], yv), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Base input price (USD per million tokens, log scale)")
ax.set_ylabel("Monthly input tokens for 3-month payback (billions, log)")
ax.set_title("Chart 4. Engineering payback: volume needed before caching work pays for itself")
ax.legend(loc="upper right", fontsize=9)
source_line(fig, "Lines: saving %.1f%% (Sonnet 5.5 5-min replay); points: each model's own replay saving. Day rate $%.0f from BLS May 2025 wage x1.4 / 230 days (our assumption)."
            % (100 * s_ref, DAY_RATE))
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(CHARTS, "chart4.png"))
plt.close(fig)

# ---------------------------------------------------------------- results.json
results = {
    "meta": {"prices_accessed": ACCESSED, "currency": "USD",
             "notes": "Multipliers are relative to each model's base input price. TTL refreshed by every request. "
                      "-1 in poisson_breakeven_mean_interarrival_min means no threshold (no write premium)."},
    "rq1_breakeven": rq1_rows,
    "rq2_poisson": poisson_summary,
    "rq2_tracelab_gaps": gap_stats,
    "rq2_tracelab_replay": trace_results,
    "rq2_mooncake_replay": mooncake,
    "rq3_payback": payback,
}


def check(o, path="root"):
    if o is None:
        raise ValueError("null at " + path)
    if isinstance(o, float) and not math.isfinite(o):
        raise ValueError("non-finite at " + path)
    if isinstance(o, dict):
        for k, v in o.items():
            check(v, path + "." + k)
    if isinstance(o, list):
        for i, v in enumerate(o):
            check(v, "%s[%d]" % (path, i))


check(results)
with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(results, fh, indent=1)
print(json.dumps(results, indent=1))
