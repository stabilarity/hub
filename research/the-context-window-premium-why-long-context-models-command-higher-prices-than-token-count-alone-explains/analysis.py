"""The Context-Window Premium: is long context priced as a distinct feature?

Deterministic, offline analysis. Reads data/ and writes charts/chart1.png ... chart4.png and
results.json next to this file. No network access, no random numbers.

Data
  data/pricing.csv            provider-direct prices, read from public pricing pages (accessed 2026-10-20)
  data/openrouter_models.csv  market cross-section from the OpenRouter models API snapshot (2026-10-20),
                              including the Artificial Analysis Intelligence Index carried by that API
  data/effective_context.csv  effective context lengths reported by NoLiMa and RULER
"""
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CHARTS = os.path.join(HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)

# Site palette: black and greys on white.
INK, INK2, MUTED, LINE, FAINT = "#000000", "#111111", "#555555", "#bbbbbb", "#dddddd"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
})
FIGSIZE, DPI = (10.67, 6.0), 150  # 1600 x 900 px
OUTPUT_TOKENS = 2000               # output length assumed for a single long request
POINTS = [32000, 128000, 200000, 500000, 990000]  # 990K = largest 1M-class prompt that leaves room for output
CACHE_SHARE = 0.9                  # share of the prompt served from cache in the cached scenario


def fnum(x):
    return None if x in ("", None) else float(x)


def load_csv(name):
    with open(os.path.join(DATA, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ols(y, X, names):
    """OLS with HC1 robust standard errors."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    meat = X.T @ (X * (resid ** 2)[:, None])
    cov = XtX_inv @ meat @ XtX_inv * n / (n - k)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (resid @ resid) / (((y - y.mean()) ** 2).sum())
    out = {"n": int(n), "r2": round(float(r2), 3)}
    for nm, b, s in zip(["const"] + names, beta, se):
        out[nm] = {"coef": round(float(b), 3), "se": round(float(s), 3),
                   "ci95": [round(float(b - 1.96 * s), 3), round(float(b + 1.96 * s), 3)]}
    return out


# --------------------------------------------------------------------------------------------
# Provider-direct dataset
# --------------------------------------------------------------------------------------------
rows = load_csv("pricing.csv")
for r in rows:
    for k in ("context_tokens", "input_usd_per_mtok", "output_usd_per_mtok", "cached_input_usd_per_mtok",
              "tier2_above_tokens", "tier2_input", "tier2_output", "tier2_cached",
              "tier3_above_tokens", "tier3_input", "tier3_output", "tier3_cached", "batch_discount"):
        r[k] = fnum(r[k])
    r["threshold_inclusive"] = r["threshold_inclusive"] == "1"
    r["tiered"] = r["tier2_above_tokens"] is not None

priced = [r for r in rows if r["input_usd_per_mtok"] is not None and r["context_tokens"]]


def rates(r, prompt):
    """Input, output and cached rates (USD per 1M tokens) for a request with `prompt` input tokens."""
    rin, rout, rc = r["input_usd_per_mtok"], r["output_usd_per_mtok"], r["cached_input_usd_per_mtok"]
    for t in ("tier2", "tier3"):
        thr = r[f"{t}_above_tokens"]
        if thr is None:
            continue
        crossed = prompt >= thr if r["threshold_inclusive"] else prompt > thr
        if crossed:
            rin, rout = r[f"{t}_input"], r[f"{t}_output"]
            rc = r[f"{t}_cached"] if r[f"{t}_cached"] is not None else rc
    return rin, rout, rc


def flat_rates(r):
    return r["input_usd_per_mtok"], r["output_usd_per_mtok"], r["cached_input_usd_per_mtok"]


def request_cost(r, prompt, out=OUTPUT_TOKENS, cache_share=0.0, batch=False, flat=False):
    """USD cost of one request. Returns None when the request does not fit the window,
    or when the requested discount is not published for the model."""
    if prompt + out > r["context_tokens"]:
        return None
    rin, rout, rc = flat_rates(r) if flat else rates(r, prompt)
    if cache_share > 0 and rc is None:
        return None
    cost_in = prompt * ((1 - cache_share) * rin + cache_share * (rc if cache_share > 0 else 0)) / 1e6
    cost = cost_in + out * rout / 1e6
    if batch:
        if not r["batch_discount"]:
            return None
        if "no long-context batch price" in r["notes"] and not flat and rates(r, prompt) != flat_rates(r):
            return None
        cost *= (1 - r["batch_discount"])
    return cost


results = {"meta": {
    "accessed": "2026-10-20", "output_tokens_per_request": OUTPUT_TOKENS, "prompt_points": POINTS,
    "cache_share": CACHE_SHARE,
    "n_rows_pricing_csv": len(rows), "n_rows_priced": len(priced),
    "n_providers_pricing_csv": len({r["provider"].split(" (via")[0] for r in rows}),
    "n_rows_missing_price": len([r for r in rows if r["input_usd_per_mtok"] is None]),
    "n_rows_missing_cached_price": len([r for r in priced if r["cached_input_usd_per_mtok"] is None]),
    "n_rows_missing_batch_discount": len([r for r in priced if r["batch_discount"] is None]),
}}

# --- RQ1a: tier prevalence and multipliers in the provider-direct data
tiered = [r for r in priced if r["tiered"]]
mult_in = [r["tier2_input"] / r["input_usd_per_mtok"] for r in tiered]
mult_out = [r["tier2_output"] / r["output_usd_per_mtok"] for r in tiered]
ge1m = [r for r in priced if r["context_tokens"] >= 1000000]
results["rq1_direct"] = {
    "n_priced": len(priced),
    "n_tiered": len(tiered),
    "share_tiered_pct": round(100 * len(tiered) / len(priced), 1),
    "n_ge_1m": len(ge1m),
    "n_tiered_ge_1m": len([r for r in ge1m if r["tiered"]]),
    "share_tiered_ge_1m_pct": round(100 * len([r for r in ge1m if r["tiered"]]) / len(ge1m), 1),
    "tier_input_multiplier_median": round(float(np.median(mult_in)), 2),
    "tier_input_multiplier_min": round(min(mult_in), 2),
    "tier_input_multiplier_max": round(max(mult_in), 2),
    "tier_output_multiplier_median": round(float(np.median(mult_out)), 2),
    "thresholds": sorted({int(r["tier2_above_tokens"]) for r in tiered}),
    "tiered_by_provider": {},
}
for r in tiered:
    results["rq1_direct"]["tiered_by_provider"].setdefault(r["provider"], []).append(r["model"])

# Provider-direct regression: log10 input price on log10 context, without and with provider fixed effects
direct_reg = [r for r in priced if r["channel"] == "first-party"]
y = [math.log10(r["input_usd_per_mtok"]) for r in direct_reg]
x = [math.log10(r["context_tokens"]) for r in direct_reg]
results["rq1_direct"]["ols_log_price_on_log_context"] = ols(y, [x], ["log10_context"])
provs = sorted({r["provider"] for r in direct_reg})
fe = [[1.0 if r["provider"] == p else 0.0 for r in direct_reg] for p in provs[1:]]
reg_fe = ols(y, [x] + fe, ["log10_context"] + [f"fe_{p}" for p in provs[1:]])
results["rq1_direct"]["ols_with_provider_fe"] = {"n": reg_fe["n"], "r2": reg_fe["r2"],
                                                 "log10_context": reg_fe["log10_context"]}

# --- RQ1b: market cross-section (OpenRouter snapshot) with a capability control
orr = []
for m in load_csv("openrouter_models.csv"):
    if ":" in m["id"] or "text" not in m["output_modalities"].split("+"):
        continue
    pin, ctx = fnum(m["input_usd_per_mtok"]), fnum(m["context_tokens"])
    if not pin or pin <= 0 or not ctx:
        continue
    m["pin"], m["ctx"], m["pout"] = pin, ctx, fnum(m["output_usd_per_mtok"])
    m["aa"] = fnum(m["aa_intelligence_index"])
    m["tier_thr"] = fnum(m["tier_min_prompt_tokens"])
    m["tier_in"] = fnum(m["tier_input_usd_per_mtok"])
    orr.append(m)
with_aa = [m for m in orr if m["aa"] is not None]
reg_b0 = ols([math.log10(m["pin"]) for m in orr], [[math.log10(m["ctx"]) for m in orr]], ["log10_context"])
reg_b1 = ols([math.log10(m["pin"]) for m in with_aa], [[math.log10(m["ctx"]) for m in with_aa]], ["log10_context"])
reg_b2 = ols([math.log10(m["pin"]) for m in with_aa],
             [[math.log10(m["ctx"]) for m in with_aa], [m["aa"] for m in with_aa]],
             ["log10_context", "aa_index"])
reg_b3 = ols([math.log10(m["pout"]) for m in with_aa if m["pout"] and m["pout"] > 0],
             [[math.log10(m["ctx"]) for m in with_aa if m["pout"] and m["pout"] > 0],
              [m["aa"] for m in with_aa if m["pout"] and m["pout"] > 0]],
             ["log10_context", "aa_index"])
or_tiered = [m for m in orr if m["tier_thr"]]


# Join the capability index onto provider-direct first-party rows by normalised model slug
def norm(s):
    s = s.lower().replace("+", "plus").split(" (")[0]
    return "".join(ch for ch in s if ch.isalnum())


aa_by_slug = {norm(m["id"].split("/", 1)[1]): m["aa"] for m in orr if m["aa"] is not None}
joined = [r for r in direct_reg if norm(r["model"]) in aa_by_slug]
for r in joined:
    r["aa"] = aa_by_slug[norm(r["model"])]
jy = [math.log10(r["input_usd_per_mtok"]) for r in joined]
jx = [math.log10(r["context_tokens"]) for r in joined]
jprov = sorted({r["provider"] for r in joined})
jfe = [[1.0 if r["provider"] == p else 0.0 for r in joined] for p in jprov[1:]]
reg_j = ols(jy, [jx, [r["aa"] for r in joined]] + jfe,
            ["log10_context", "aa_index"] + [f"fe_{p}" for p in jprov[1:]])
results["rq1_direct"]["ols_with_provider_fe_and_capability"] = {
    "n": reg_j["n"], "r2": reg_j["r2"], "log10_context": reg_j["log10_context"], "aa_index": reg_j["aa_index"],
    "providers": jprov, "models": [r["model"] for r in joined]}
results["rq1_direct"]["price_multiple_per_10x_context_fe_capability"] = round(10 ** reg_j["log10_context"]["coef"], 2)
big = [m for m in orr if m["ctx"] >= 1000000]
results["rq1_market"] = {
    "n_models": len(orr), "n_with_aa_index": len(with_aa),
    "ols_all_no_control": reg_b0, "ols_aa_subset_no_control": reg_b1,
    "ols_aa_subset_with_capability": reg_b2, "ols_output_price_with_capability": reg_b3,
    "price_multiple_per_10x_context_no_control": round(10 ** reg_b1["log10_context"]["coef"], 2),
    "price_multiple_per_10x_context_with_capability": round(10 ** reg_b2["log10_context"]["coef"], 2),
    "price_multiple_per_aa_point": round(10 ** reg_b2["aa_index"]["coef"], 3),
    "n_tiered": len(or_tiered), "share_tiered_pct": round(100 * len(or_tiered) / len(orr), 1),
    "n_ge_1m": len(big), "share_tiered_ge_1m_pct": round(100 * len([m for m in big if m["tier_thr"]]) / len(big), 1),
    "tier_input_multiplier_median": round(float(np.median([m["tier_in"] / m["pin"] for m in or_tiered])), 2),
}

# --- RQ2: the cost of one long request, with step functions
cost_table = {}
for r in priced:
    cost_table[r["model"] + " | " + r["provider"]] = {
        str(p): (None if request_cost(r, p) is None else round(request_cost(r, p), 4)) for p in POINTS}
results["rq2_request_cost_usd"] = cost_table

cliffs = []
for r in tiered:
    thr = int(r["tier2_above_tokens"])
    below = thr - 1 if r["threshold_inclusive"] else thr
    above = thr if r["threshold_inclusive"] else thr + 1
    c0, c1 = request_cost(r, below), request_cost(r, above)
    if c0 is None or c1 is None:
        continue
    # smallest prompt length at which the one-token-longer request is still not cheaper than the short tier
    rin = r["input_usd_per_mtok"]
    equiv_tokens = (c1 - OUTPUT_TOKENS * r["output_usd_per_mtok"] / 1e6) * 1e6 / rin
    two_t = 2 * thr
    one = request_cost(r, two_t)
    split = 2 * request_cost(r, thr - 1 if r["threshold_inclusive"] else thr)
    cliffs.append({
        "provider": r["provider"], "model": r["model"], "threshold_tokens": thr,
        "cost_just_below_usd": round(c0, 4), "cost_just_above_usd": round(c1, 4),
        "jump_pct": round(100 * (c1 / c0 - 1), 1),
        "flat_rate_equivalent_prompt_tokens": int(round(equiv_tokens)),
        "one_request_at_2x_threshold_usd": None if one is None else round(one, 4),
        "two_requests_below_threshold_usd": round(split, 4),
        "split_saving_pct": None if one is None else round(100 * (1 - split / one), 1),
    })
results["rq2_cliffs"] = cliffs
results["rq2_summary"] = {
    "median_jump_pct": round(float(np.median([c["jump_pct"] for c in cliffs])), 1),
    "max_jump_pct": max(c["jump_pct"] for c in cliffs),
    "min_jump_pct": min(c["jump_pct"] for c in cliffs),
    "median_split_saving_pct": round(float(np.median([c["split_saving_pct"] for c in cliffs
                                                      if c["split_saving_pct"] is not None])), 1),
}

# Premium at 990K versus the same tokens at short-tier rates
prem = []
for r in tiered:
    p = 990000
    tiered_cost, flat_cost = request_cost(r, p), request_cost(r, p, flat=True)
    if tiered_cost is None:
        continue
    prem.append({"provider": r["provider"], "model": r["model"], "tiered_usd": round(tiered_cost, 4),
                 "short_rate_usd": round(flat_cost, 4),
                 "premium_pct": round(100 * (tiered_cost / flat_cost - 1), 1)})
results["rq2_premium_990k"] = prem
results["rq2_summary"]["median_premium_990k_pct"] = round(float(np.median([x["premium_pct"] for x in prem])), 1)

# Within-family comparisons: identical price, different window
results["rq2_same_price_different_window"] = [
    {"pair": "Claude Sonnet 4.5 (200K) vs Claude Sonnet 4.6 (1M)", "input_usd": [3, 3], "output_usd": [15, 15]},
    {"pair": "Qwen3.7 Plus first-party tiered vs Together flat",
     "first_party_990k_usd": round(request_cost(next(r for r in priced if r["model"] == "qwen3.7-plus"), 990000), 4),
     "together_990k_usd": round(request_cost(next(r for r in priced if r["model"] == "Qwen3.7 Plus"), 990000), 4)},
]

# --- RQ3: how much of the premium survives caching and batching
STACK_OK = {"Anthropic", "OpenAI"}  # both pages state that cache and batch discounts stack
rq3 = []
for r in tiered:
    p = 990000
    row = {"provider": r["provider"], "model": r["model"]}
    for scen, kw in {"standard": {}, "cached": {"cache_share": CACHE_SHARE}, "batch": {"batch": True},
                     "cached_batch": {"cache_share": CACHE_SHARE, "batch": True}}.items():
        if scen == "cached_batch" and r["provider"] not in STACK_OK:
            row[scen] = None
            continue
        t, f = request_cost(r, p, **kw), request_cost(r, p, flat=True, **kw)
        row[scen] = None if t is None else {"tiered_usd": round(t, 4), "short_rate_usd": round(f, 4),
                                            "premium_usd": round(t - f, 4),
                                            "premium_pct": round(100 * (t / f - 1), 1)}
    if row["standard"] is not None:
        rq3.append(row)
results["rq3_cache_batch_990k"] = rq3


def med(key, field):
    vals = [x[key][field] for x in rq3 if x.get(key)]
    return round(float(np.median(vals)), 1) if vals else None


results["rq3_summary"] = {
    "median_premium_pct_standard": med("standard", "premium_pct"),
    "median_premium_pct_cached": med("cached", "premium_pct"),
    "median_premium_pct_batch": med("batch", "premium_pct"),
    "median_premium_usd_standard": round(float(np.median([x["standard"]["premium_usd"] for x in rq3])), 3),
    "median_premium_usd_cached": round(float(np.median([x["cached"]["premium_usd"] for x in rq3 if x["cached"]])), 3),
    "n_models_cached": len([x for x in rq3 if x["cached"]]),
    "n_models_batch": len([x for x in rq3 if x["batch"]]),
}
std_c = [x for x in rq3 if x["cached"]]
results["rq3_summary"]["median_premium_usd_reduction_by_cache_pct"] = round(float(np.median(
    [100 * (1 - x["cached"]["premium_usd"] / x["standard"]["premium_usd"]) for x in std_c])), 1)

# Cost per usable token: claimed window versus effective window
eff = load_csv("effective_context.csv")
by_model = {r["model"]: r for r in priced}
usable = []
for e in eff:
    r = by_model.get(e["priced_as"])
    if r is None:
        continue
    claimed, effective = float(e["claimed_tokens"]), float(e["effective_tokens"])
    usable.append({
        "benchmark": e["benchmark"], "model": e["model"], "priced_as": r["model"], "provider": r["provider"],
        "claimed_tokens": int(claimed), "effective_tokens": int(effective),
        "effective_upper_bound": e["effective_is_upper_bound"] == "1",
        "claimed_to_effective_ratio": round(claimed / effective, 1),
        "list_input_usd_per_mtok": r["input_usd_per_mtok"],
        "usd_per_mtok_reliably_used_at_full_window": round(r["input_usd_per_mtok"] * claimed / effective, 2),
    })
usable.sort(key=lambda u: -u["claimed_to_effective_ratio"])
results["rq3_usable_tokens"] = usable
nolima = [u for u in usable if u["benchmark"] == "NoLiMa"]
results["rq3_usable_summary"] = {
    "n_models": len(nolima),
    "median_claimed_to_effective_ratio": round(float(np.median([u["claimed_to_effective_ratio"] for u in nolima])), 1),
    "min_claimed_to_effective_ratio": min(u["claimed_to_effective_ratio"] for u in nolima),
    "max_claimed_to_effective_ratio": max(u["claimed_to_effective_ratio"] for u in nolima),
}

# --------------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------------
SRC = "Source: provider pricing pages and OpenRouter models API, accessed 2026-10-20; analysis: stabilarity/hub"


def finish(fig, ax, title, xlabel, ylabel, source, name):
    ax.set_title(title, loc="left", fontsize=14, color=INK, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=FAINT, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.01, source, fontsize=8.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(CHARTS, name), dpi=DPI)
    plt.close(fig)


# Chart 1: price vs context, market cross-section and provider-direct points
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.scatter([m["ctx"] for m in orr], [m["pin"] for m in orr], s=14, color=LINE, label=f"OpenRouter listings (n={len(orr)})",
           edgecolors="none")
fr = [r for r in direct_reg]
ax.scatter([r["context_tokens"] for r in fr], [r["input_usd_per_mtok"] for r in fr], s=34, color=INK,
           marker="D", label=f"Provider-direct, first-party (n={len(fr)})", edgecolors="#ffffff", linewidths=0.8)
xs = np.logspace(math.log10(3e4), math.log10(1.1e7), 50)
b = reg_b0
ax.plot(xs, 10 ** (b["const"]["coef"] + b["log10_context"]["coef"] * np.log10(xs)), color=MUTED, lw=2,
        ls="--", label=f"OLS fit, all listings: elasticity {b['log10_context']['coef']}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(frameon=False, loc="lower right", fontsize=10)
finish(fig, ax, "Chart 1. Input price vs maximum context window",
       "Maximum context window (tokens, log scale)", "Input price (USD per 1M tokens, log scale)", SRC, "chart1.png")

# Chart 2: effective input rate index vs prompt length (step functions)
sel = ["gpt-5.4", "gemini-3.1-pro-preview", "grok-4.7", "Claude Haiku 5.5", "qwen3.7-plus", "qwen3.7-flash",
       "Claude Sonnet 5.5"]
styles = [("-", INK, None), ("--", INK, None), (":", INK, None), ("-", MUTED, None), ("--", MUTED, None),
          ("-.", INK2, None), ("-", LINE, None)]
fig, ax = plt.subplots(figsize=FIGSIZE)
grid = np.unique(np.concatenate([np.linspace(8000, 1_000_000, 900).astype(int),
                                 np.array([32000, 32001, 100000, 100001, 199999, 200000, 200001,
                                           256000, 256001, 272000, 272001])]))
for (name, (ls, col, _)) in zip(sel, styles):
    r = next(x for x in priced if x["model"] == name)
    base = request_cost(r, 32000) / 32000
    xs_, ys_ = [], []
    for p in grid:
        c = request_cost(r, int(p))
        if c is None:
            continue
        xs_.append(p / 1000)
        ys_.append((c / p) / base)
    ax.plot(xs_, ys_, ls=ls, color=col, lw=2, label=f"{name} ({r['provider']})", drawstyle="steps-post")
ax.set_xlim(0, 1000)
ax.set_ylim(0.5, 6.2)
ax.legend(frameon=False, fontsize=9.5, loc="upper center", ncol=3)
finish(fig, ax, "Chart 2. Cost per prompt token rises in steps at tier thresholds",
       "Prompt length (thousand tokens), 2,000 output tokens per request",
       "Cost per prompt token, index (32K request = 1.0)", SRC, "chart2.png")

# Chart 3: long-context premium in USD for a 990K request: standard, cached, batch
lab = [f"{x['model']}" for x in rq3]
std = [x["standard"]["premium_usd"] for x in rq3]
cac = [x["cached"]["premium_usd"] if x["cached"] else 0 for x in rq3]
bat = [x["batch"]["premium_usd"] if x["batch"] else 0 for x in rq3]
order = np.argsort(std)[::-1]
fig, ax = plt.subplots(figsize=FIGSIZE)
ypos = np.arange(len(rq3))
h = 0.26
ax.barh(ypos - h, [std[i] for i in order], height=h - 0.02, color=INK, label="Standard")
ax.barh(ypos, [bat[i] for i in order], height=h - 0.02, color=MUTED, label="Batch (where offered)")
ax.barh(ypos + h, [cac[i] for i in order], height=h - 0.02, color=LINE,
        label=f"{int(CACHE_SHARE * 100)}% of prompt cached (where priced)")
ax.set_yticks(ypos)
ax.set_yticklabels([lab[i] for i in order], fontsize=9.5)
ax.invert_yaxis()
ax.set_xscale("log")
ax.legend(frameon=False, loc="lower right", fontsize=10)
finish(fig, ax, "Chart 3. Long-context surcharge on one 990K-token request",
       "Surcharge vs short-tier rates (USD per request, log scale)", "", SRC, "chart3.png")

# Chart 4: claimed vs effective context, USD per 1M reliably used tokens at full window
nl = sorted(nolima, key=lambda u: u["usd_per_mtok_reliably_used_at_full_window"])
fig, ax = plt.subplots(figsize=FIGSIZE)
yp = np.arange(len(nl))
ax.barh(yp, [u["list_input_usd_per_mtok"] for u in nl], height=0.36, color=LINE, label="List input price")
ax.barh(yp + 0.38, [u["usd_per_mtok_reliably_used_at_full_window"] for u in nl], height=0.36, color=INK,
        label="Price per reliably used token, full window filled")
ax.set_yticks(yp + 0.19)
ax.set_yticklabels([f"{u['model'].replace(' (w/o thinking)', '')}: {u['claimed_tokens'] // 1000:,}K / "
                    f"{'<' if u['effective_upper_bound'] else ''}{u['effective_tokens'] // 1000}K" for u in nl], fontsize=9.5)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(0.05, 3e4)
ax.legend(frameon=False, loc="upper right", fontsize=10)
finish(fig, ax, "Chart 4. Price per reliably used token at a full window",
       "USD per 1M input tokens (log scale); labels: claimed / effective context (NoLiMa)", "",
       "Source: NoLiMa effective lengths (Modarressi et al., ICML 2025); current list prices, accessed 2026-10-20",
       "chart4.png")

with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1)
print(json.dumps({k: results[k] for k in ("meta", "rq1_direct", "rq2_summary", "rq3_summary", "rq3_usable_summary")},
                 indent=1))
print(json.dumps({k: results["rq1_market"][k] for k in results["rq1_market"] if not k.startswith("ols")}, indent=1))
