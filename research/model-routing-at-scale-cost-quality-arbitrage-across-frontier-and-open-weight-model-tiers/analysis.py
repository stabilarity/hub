"""Model routing at scale: cost-quality arbitrage across frontier and open-weight model tiers.

Deterministic, offline analysis. Reads data/ and writes charts/chart1.png ... chart4.png and results.json.

Data
  data/sprout_scores.csv.gz       per-prompt judged score (0-1) and input/output tokens of 13 models, SPROUT dataset
                                  (CARROT-LLM-Routing/SPROUT, Somerstep et al., arXiv:2502.03261), built by build_data.py
  data/router_predictions.csv.gz  predicted score per model for validation and test prompts from a TF-IDF + ridge
                                  router trained on the SPROUT train split ('in_dist') or with the prompt's task
                                  family held out ('lofo'); built by build_data.py
  data/prices.csv                 list prices (USD per 1M tokens) read on 2026-10-21 from official pricing pages
  data/tier_pairs.csv             current frontier / cheap-tier price pairs, same pages, same date

Method (summary; README has details)
  Pool: GPT-4o (frontier), GPT-4o-mini, Llama-3.3-70B, Llama-3.1-8B. Cost per query = the model's own measured
  input and output tokens x its list price. Quality = mean judged score on the 6,637 test prompts.
  Pool router: pick argmax_m (predicted score_m - lambda x expected cost_m), expected cost = mean train cost.
  Two-tier router: escalate to GPT-4o when predicted(GPT-4o) - predicted(cheap) > t.
  lambda and t are chosen on the validation split for each quality target, then applied unchanged to test.
  Oracle: the same rules with true test scores and true costs (an upper bound, not a router).
  Random mix: the best random split of traffic between two pool models at matched cost (lower convex hull).
"""
import csv
import gzip
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CHARTS = os.path.join(HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)

INK, INK2, MUTED, LINE, FAINT = "#000000", "#111111", "#555555", "#bbbbbb", "#dddddd"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
})
FIGSIZE, DPI = (10.67, 6.0), 150
SOURCE = "Source: SPROUT (arXiv:2502.03261) per-query scores and tokens; list prices read 2026-10-21; our analysis."

FRONTIER = "openai-gpt-4o"
POOL = ["openai-gpt-4o", "openai-gpt-4o-mini", "wxai-llama-3-3-70b-instruct", "wxai-llama-3-1-8b-instruct"]
CHEAP = ["wxai-llama-3-3-70b-instruct", "wxai-llama-3-1-8b-instruct", "openai-gpt-4o-mini"]
TARGETS = [0.95, 0.98, 1.00]
LAMBDAS = np.concatenate([[0.0], np.geomspace(1e-3, 1e3, 400)])  # score points per USD per query
NOISE_SIGMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
SEED = 20261021


def r4(x):
    return round(float(x), 4)


# ---------------------------------------------------------------- load
prices = {p["model"]: p for p in csv.DictReader(open(os.path.join(DATA, "prices.csv"), encoding="utf-8"))}
LABEL = {m: prices[m]["label"] for m in POOL}
rows = list(csv.DictReader(gzip.open(os.path.join(DATA, "sprout_scores.csv.gz"), "rt", encoding="utf-8")))
preds = {r["key"]: r for r in csv.DictReader(gzip.open(os.path.join(DATA, "router_predictions.csv.gz"), "rt",
                                                       encoding="utf-8"))}


def cost_of(r, m):
    p = prices[m]
    return (int(r[m + "|in"]) * float(p["usd_per_1m_input"]) + int(r[m + "|out"]) * float(p["usd_per_1m_output"])) / 1e6


def table(split):
    rs = [r for r in rows if r["split"] == split]
    S = np.array([[float(r[m + "|score"]) for m in POOL] for r in rs])
    C = np.array([[cost_of(r, m) for m in POOL] for r in rs])
    fam = np.array([r["family"] for r in rs])
    out = {"S": S, "C": C, "fam": fam, "n": len(rs), "tok_in": np.array([[int(r[m + "|in"]) for m in POOL] for r in rs]),
           "tok_out": np.array([[int(r[m + "|out"]) for m in POOL] for r in rs])}
    if split != "train":
        for v in ("in_dist", "lofo"):
            out["P_" + v] = np.array([[float(preds[r["key"]][f"{v}|{m}"]) for m in POOL] for r in rs])
    return out


TR, VA, TE = table("train"), table("validation"), table("test")
EXP_COST = TR["C"].mean(axis=0)  # router's expected cost per model (mean train cost)
iF = POOL.index(FRONTIER)


def evaluate(choice, T):
    idx = np.arange(T["n"])
    return T["S"][idx, choice].mean(), T["C"][idx, choice].mean()


# ---------------------------------------------------------------- RQ1: single models, random mix, oracle headroom
single = {}
for j, m in enumerate(POOL):
    single[m] = {"label": LABEL[m], "mean_score": r4(TE["S"][:, j].mean()),
                 "usd_per_1k_queries": r4(TE["C"][:, j].mean() * 1000),
                 "mean_input_tokens": r4(TE["tok_in"][:, j].mean()), "mean_output_tokens": r4(TE["tok_out"][:, j].mean())}
qF, cF = TE["S"][:, iF].mean(), TE["C"][:, iF].mean()
qF_val = VA["S"][:, iF].mean()


def random_mix_cost(target_q):
    """Cheapest expected cost of a random split between any two pool models reaching quality target_q."""
    best = None
    pts = [(TE["C"][:, j].mean(), TE["S"][:, j].mean()) for j in range(len(POOL))]
    for a in range(len(pts)):
        for b in range(len(pts)):
            (ca, qa), (cb, qb) = pts[a], pts[b]
            if qa >= target_q:
                c = ca
            elif qb > qa and qb >= target_q:
                x = (target_q - qa) / (qb - qa)
                c = (1 - x) * ca + x * cb
            else:
                continue
            best = c if best is None else min(best, c)
    return best


def pool_choice(Sx, cost_vec_or_mat, lam):
    util = Sx - lam * cost_vec_or_mat
    return np.argmax(util, axis=1)


def oracle_cost(target_q, models=None):
    """Oracle with true test scores and costs over a model subset: cheapest lambda meeting the target."""
    cols = list(range(len(POOL))) if models is None else models
    S, C = TE["S"][:, cols], TE["C"][:, cols]
    best = None
    for lam in LAMBDAS:
        ch = np.argmax(S - lam * C, axis=1)
        idx = np.arange(TE["n"])
        q, c = S[idx, ch].mean(), C[idx, ch].mean()
        if q >= target_q - 1e-12:
            best = (c, q) if best is None or c < best[0] else best
    return best


# oracle headroom: share of test prompts on which each cheap model scores at least as well as GPT-4o
headroom = {}
for m in CHEAP:
    j = POOL.index(m)
    headroom[m] = {"label": LABEL[m], "share_cheap_at_least_frontier": r4((TE["S"][:, j] >= TE["S"][:, iF]).mean()),
                   "share_cheap_strictly_better": r4((TE["S"][:, j] > TE["S"][:, iF]).mean()),
                   "share_frontier_strictly_better": r4((TE["S"][:, j] < TE["S"][:, iF]).mean())}
oracle_best = oracle_cost(-1)  # lambda sweep min-cost point
oracle_max_q = max(TE["S"].max(axis=1).mean(), 0)

# ---------------------------------------------------------------- RQ2: trained routers, thresholds set on validation
def pool_router(variant, target_ratio):
    """Pick the largest lambda whose validation quality >= target_ratio x GPT-4o validation score."""
    Pv, Pt = VA["P_" + variant], TE["P_" + variant]
    chosen = 0.0
    for lam in LAMBDAS:
        q, _ = evaluate(pool_choice(Pv, EXP_COST, lam), VA)
        if q >= target_ratio * qF_val:
            chosen = lam
    ch = pool_choice(Pt, EXP_COST, chosen)
    q, c = evaluate(ch, TE)
    share = {LABEL[m]: r4((ch == j).mean()) for j, m in enumerate(POOL)}
    # matched-quality view: the cheapest lambda whose TEST quality meets the target (ideal calibration)
    best = None
    max_ret = 0.0
    for lam in LAMBDAS:
        qq, cc = evaluate(pool_choice(Pt, EXP_COST, lam), TE)
        max_ret = max(max_ret, qq / qF)
        if qq >= target_ratio * qF - 1e-12 and (best is None or cc < best):
            best = cc
    matched = r4(1 - best / cF) if best is not None else -1.0
    return {"lambda": r4(chosen), "test_score": r4(q), "test_retention": r4(q / qF), "usd_per_1k": r4(c * 1000),
            "saving_vs_frontier": r4(1 - c / cF), "traffic_share": share,
            "matched_quality_saving": matched, "target_reachable_on_test": best is not None,
            "max_test_retention": r4(max_ret)}


def two_tier_router(variant, cheap, target_ratio, T_cal=None, T_eval=None):
    jc = POOL.index(cheap)
    T_cal = VA if T_cal is None else T_cal
    T_eval = TE if T_eval is None else T_eval

    def run(T, t, gap):
        esc = gap > t
        ch = np.where(esc, iF, jc)
        q, c = evaluate(ch, T)
        return q, c, esc.mean()

    gap_v = T_cal["P_" + variant][:, iF] - T_cal["P_" + variant][:, jc]
    gap_t = T_eval["P_" + variant][:, iF] - T_eval["P_" + variant][:, jc]
    cand = np.unique(np.concatenate([[-np.inf, np.inf], gap_v]))
    chosen = -np.inf
    for t in cand[::-1]:  # from escalate-nothing toward escalate-everything; keep first that meets target
        q, _, _ = run(T_cal, t, gap_v)
        if q >= target_ratio * T_cal["S"][:, iF].mean():
            chosen = t
            break
    q, c, e = run(T_eval, chosen, gap_t)
    return {"test_score": r4(q), "test_retention": r4(q / T_eval["S"][:, iF].mean()), "usd_per_1k": r4(c * 1000),
            "saving_vs_frontier": r4(1 - c / T_eval["C"][:, iF].mean()), "escalation_rate": r4(e)}


def two_tier_oracle(cheap, target_ratio):
    jc = POOL.index(cheap)
    gap = TE["S"][:, iF] - TE["S"][:, jc]
    order = np.argsort(-gap, kind="stable")
    base = TE["S"][:, jc].copy()
    total_q = base.sum()
    need = target_ratio * TE["S"][:, iF].sum()
    k = 0
    while total_q < need - 1e-9 and k < TE["n"]:
        total_q += gap[order[k]]
        k += 1
    esc = np.zeros(TE["n"], bool)
    esc[order[:k]] = True
    ch = np.where(esc, iF, jc)
    q, c = evaluate(ch, TE)
    return {"test_retention": r4(q / qF), "usd_per_1k": r4(c * 1000), "saving_vs_frontier": r4(1 - c / cF),
            "escalation_rate": r4(esc.mean())}


rq2_pool = {}
for tr in TARGETS:
    key = f"{int(round(tr * 100))}pct"
    orc = oracle_cost(tr * qF)
    rm = random_mix_cost(tr * qF)
    rq2_pool[key] = {
        "oracle_usd_per_1k": r4(orc[0] * 1000), "oracle_saving": r4(1 - orc[0] / cF),
        "random_mix_usd_per_1k": r4(rm * 1000), "random_mix_saving": r4(1 - rm / cF),
        "router_in_dist": pool_router("in_dist", tr), "router_lofo": pool_router("lofo", tr),
    }
    for v in ("in_dist", "lofo"):
        s = rq2_pool[key]["router_" + v]["saving_vs_frontier"]
        rq2_pool[key]["router_" + v]["share_of_oracle_saving"] = r4(s / rq2_pool[key]["oracle_saving"]) \
            if rq2_pool[key]["oracle_saving"] > 0 else 0.0

rq2_two_tier = {}
for m in CHEAP:
    rq2_two_tier[LABEL[m]] = {}
    for tr in TARGETS:
        key = f"{int(round(tr * 100))}pct"
        rq2_two_tier[LABEL[m]][key] = {"oracle": two_tier_oracle(m, tr),
                                       "router_in_dist": two_tier_router("in_dist", m, tr),
                                       "router_lofo": two_tier_router("lofo", m, tr)}

# per-family realised result of the in-distribution pool router at the 98% target (lambda from all validation)
lam98 = rq2_pool["98pct"]["router_in_dist"]["lambda"]
lam98_lofo = rq2_pool["98pct"]["router_lofo"]["lambda"]
fam_res = {}
for fam in sorted(set(TE["fam"])):
    sel = TE["fam"] == fam
    sub = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in TE.items()}
    sub["n"] = int(sel.sum())
    out = {"n": sub["n"], "gpt4o_score": r4(sub["S"][:, iF].mean())}
    for v, lam in (("in_dist", lam98), ("lofo", lam98_lofo)):
        ch = pool_choice(sub["P_" + v], EXP_COST, lam)
        q, c = evaluate(ch, sub)
        out[v] = {"retention": r4(q / sub["S"][:, iF].mean()), "saving": r4(1 - c / sub["C"][:, iF].mean())}
    best_single = max(range(len(POOL)), key=lambda j: sub["S"][:, j].mean())
    out["best_single_model"] = LABEL[POOL[best_single]]
    out["best_single_score"] = r4(sub["S"][:, best_single].mean())
    fam_res[fam] = out

# ---------------------------------------------------------------- RQ3a: router quality sensitivity (two-tier, Llama-3.3-70B)
def auc(score, label):
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    s_sorted = score[order]
    i = 0
    r = 1
    while i < len(score):  # average ranks for ties
        j = i
        while j + 1 < len(score) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (r + r + (j - i)) / 2
        r += j - i + 1
        i = j + 1
    pos = label.sum()
    neg = len(label) - pos
    return (ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg)


def min_cost_at_target(esc_score, cheap_j, target_ratio):
    """Escalate in descending order of esc_score until the target is met (threshold set on test)."""
    order = np.argsort(-esc_score, kind="stable")
    gap = TE["S"][:, iF] - TE["S"][:, cheap_j]
    dc = TE["C"][:, iF] - TE["C"][:, cheap_j]
    q = TE["S"][:, cheap_j].sum() + np.concatenate([[0], np.cumsum(gap[order])])
    c = TE["C"][:, cheap_j].sum() + np.concatenate([[0], np.cumsum(dc[order])])
    ok = np.where(q >= target_ratio * TE["S"][:, iF].sum() - 1e-9)[0]
    k = ok[0] if len(ok) else TE["n"]
    return 1 - c[k] / TE["C"][:, iF].sum(), k / TE["n"]


jL = POOL.index("wxai-llama-3-3-70b-instruct")
true_gap = TE["S"][:, iF] - TE["S"][:, jL]
label = true_gap > 0
rng = np.random.default_rng(SEED)
z = rng.standard_normal(TE["n"])
noise_curve = []
for sg in NOISE_SIGMAS:
    sc = true_gap + sg * z
    s, e = min_cost_at_target(sc, jL, 0.98)
    noise_curve.append({"sigma": sg, "auc": r4(auc(sc, label)), "saving_98": r4(s), "escalation_98": r4(e)})
rand_s, rand_e = min_cost_at_target(z, jL, 0.98)
for m in CHEAP:
    jc = POOL.index(m)
    for tr in TARGETS:
        key = f"{int(round(tr * 100))}pct"
        for v in ("in_dist", "lofo"):
            sc = TE["P_" + v][:, iF] - TE["P_" + v][:, jc]
            s_m, e_m = min_cost_at_target(sc, jc, tr)
            rq2_two_tier[LABEL[m]][key]["router_" + v]["matched_quality_saving"] = r4(s_m)
            rq2_two_tier[LABEL[m]][key]["router_" + v]["matched_quality_escalation"] = r4(e_m)
router_pts = {}
for v in ("in_dist", "lofo"):
    sc = TE["P_" + v][:, iF] - TE["P_" + v][:, jL]
    s, e = min_cost_at_target(sc, jL, 0.98)
    router_pts[v] = {"auc": r4(auc(sc, label)), "saving_98": r4(s), "escalation_98": r4(e)}
router_pts["random"] = {"auc": r4(auc(z, label)), "saving_98": r4(rand_s), "escalation_98": r4(rand_e)}

# ---------------------------------------------------------------- RQ3b: tier price-gap sensitivity
pairs = list(csv.DictReader(open(os.path.join(DATA, "tier_pairs.csv"), encoding="utf-8")))
mix_in = float(TE["tok_in"].mean())
mix_out = float(TE["tok_out"].mean())
pair_res = []
for p in pairs:
    f = (mix_in * float(p["frontier_in"]) + mix_out * float(p["frontier_out"]))
    c = (mix_in * float(p["cheap_in"]) + mix_out * float(p["cheap_out"]))
    pair_res.append({"pair": p["pair"], "blended_price_ratio": r4(f / c)})


def saving_at_ratio(e, ratio):
    return 1 - ((1 - e) / ratio + e)


esc_levels = {
    "oracle": rq2_two_tier["Llama-3.3-70B"]["98pct"]["oracle"]["escalation_rate"],
    "router_in_dist": rq2_two_tier["Llama-3.3-70B"]["98pct"]["router_in_dist"]["matched_quality_escalation"],
    "router_lofo": rq2_two_tier["Llama-3.3-70B"]["98pct"]["router_lofo"]["matched_quality_escalation"],
}
for pr in pair_res:
    pr["saving_at_escalation"] = {k: r4(saving_at_ratio(e, pr["blended_price_ratio"])) for k, e in esc_levels.items()}
ceiling = {k: r4(1 - e) for k, e in esc_levels.items()}

# ---------------------------------------------------------------- charts
def source_line(fig, extra=""):
    fig.text(0.01, 0.01, SOURCE + extra, fontsize=8, color=MUTED, ha="left")


# Chart 1: cost-quality plane
fig, ax = plt.subplots(figsize=FIGSIZE)
pts = [(TE["C"][:, j].mean() * 1000, TE["S"][:, j].mean(), LABEL[m]) for j, m in enumerate(POOL)]
curves = {}
for name, P, C, T in (("Oracle (true scores, upper bound)", TE["S"], TE["C"], None),
                      ("Trained router, in-distribution", TE["P_in_dist"], EXP_COST, "in_dist"),
                      ("Trained router, unseen task family", TE["P_lofo"], EXP_COST, "lofo")):
    xs, ys = [], []
    for lam in LAMBDAS:
        ch = pool_choice(P, C, lam)
        q, c = evaluate(ch, TE)
        xs.append(c * 1000)
        ys.append(q)
    curves[name] = (xs, ys)
styles = {"Oracle (true scores, upper bound)": dict(color=LINE, lw=2, ls="-"),
          "Trained router, in-distribution": dict(color=INK, lw=2, ls="-"),
          "Trained router, unseen task family": dict(color=MUTED, lw=2, ls="--")}
for name, (xs, ys) in curves.items():
    ax.plot(xs, ys, label=name, **styles[name])
# random-mix hull
hx = np.geomspace(min(p[0] for p in pts), max(p[0] for p in pts), 200)
hy = []
for x in hx:
    best = None
    for a in pts:
        for b in pts:
            if a[0] <= x <= b[0] and b[0] > a[0]:
                w = (x - a[0]) / (b[0] - a[0])
                y = (1 - w) * a[1] + w * b[1]
                best = y if best is None else max(best, y)
            elif abs(a[0] - x) < 1e-12:
                best = a[1] if best is None else max(best, a[1])
    hy.append(best)
ax.plot(hx, hy, color=INK2, lw=1.2, ls=":", label="Random split between two models")
for x, y, lab in pts:
    ax.scatter([x], [y], s=60, color=INK, zorder=5, edgecolor="#ffffff", linewidth=1.5)
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -14), fontsize=10, color=INK2)
ax.axhline(0.98 * qF, color=FAINT, lw=1)
ax.text(hx[0], 0.98 * qF + 0.002, "98% of GPT-4o score", fontsize=9, color=MUTED)
ax.set_xscale("log")
ax.set_xticks([0.2, 0.5, 1, 2, 5])
ax.set_xticklabels(["0.2", "0.5", "1", "2", "5"])
ax.set_xlabel("Cost, USD per 1,000 queries (log scale)")
ax.set_ylabel("Mean judged score on 6,637 test prompts (0-1)")
ax.set_title("Cost-quality plane: single models, random splits, trained routers and the oracle bound")
ax.legend(frameon=False, loc="lower right", fontsize=9)
source_line(fig)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(CHARTS, "chart1.png"), dpi=DPI)
plt.close(fig)

# Chart 2: savings at matched test quality
fig, ax = plt.subplots(figsize=FIGSIZE)
groups = ["95pct", "98pct", "100pct"]
series = [("Oracle bound", lambda g: rq2_pool[g]["oracle_saving"], LINE, None),
          ("Router, in-distribution", lambda g: rq2_pool[g]["router_in_dist"]["matched_quality_saving"], INK, None),
          ("Router, unseen task family", lambda g: rq2_pool[g]["router_lofo"]["matched_quality_saving"], MUTED, "//"),
          ("Best random two-model split", lambda g: rq2_pool[g]["random_mix_saving"], "#ffffff", "..")]
w = 0.2
xs = np.arange(len(groups))
for i, (name, fn, col, hatch) in enumerate(series):
    vals = [fn(g) * 100 for g in groups]
    shown = [max(v, 0) for v in vals]
    bars = ax.bar(xs + (i - 1.5) * w, shown, w * 0.92, color=col, edgecolor=INK2, hatch=hatch, label=name, lw=0.8)
    for b, v in zip(bars, vals):
        txt = "not\nreached" if v < 0 else f"{v:.1f}"
        ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 1, txt, ha="center", fontsize=8.5, color=INK2)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_ylim(0, 122)
ax.set_xticks(xs)
ax.set_xticklabels(["95% of GPT-4o score", "98% of GPT-4o score", "100% of GPT-4o score"])
ax.set_ylabel("Cost saving vs sending all traffic to GPT-4o (%)")
ax.set_title("Saving at matched test quality: four-model pool, routers vs random split vs oracle")
ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
source_line(fig, " 'Not reached': no setting met the target.")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(CHARTS, "chart2.png"), dpi=DPI)
plt.close(fig)

# Chart 3: router quality sensitivity
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot([p["auc"] for p in noise_curve], [p["saving_98"] * 100 for p in noise_curve], color=INK, lw=2,
        marker="o", ms=5, label="Oracle gap plus Gaussian noise (sigma 0 to 10)")
for v, lab, mk in (("in_dist", "Trained router, in-distribution", "s"), ("lofo", "Trained router, unseen family", "D"),
                   ("random", "Random escalation order", "^")):
    p = router_pts[v]
    ax.scatter([p["auc"]], [p["saving_98"] * 100], marker=mk, s=90, color=MUTED if v != "in_dist" else INK2,
               edgecolor="#ffffff", linewidth=1.5, zorder=5,
               label=f"{lab}: AUC {p['auc']:.3f}, saving {p['saving_98'] * 100:.1f}%")
ax.set_xlabel("Router AUC for 'GPT-4o scores higher than Llama-3.3-70B' (0.5 = random)")
ax.set_ylabel("Saving at 98% of GPT-4o score (%)")
ax.set_title("Two-tier GPT-4o / Llama-3.3-70B at 98% of GPT-4o score: saving vs router discrimination")
ax.legend(frameon=False, fontsize=9, loc="lower right")
source_line(fig, " Thresholds set on test prompts for all points (sensitivity view).")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(CHARTS, "chart3.png"), dpi=DPI)
plt.close(fig)

# Chart 4: price-gap sensitivity
fig, ax = plt.subplots(figsize=FIGSIZE)
ratios = np.geomspace(1, 300, 300)
for k, lab, col, ls in (("oracle", "Oracle escalation rate", LINE, "-"),
                        ("router_in_dist", "In-distribution router escalation rate", INK, "-"),
                        ("router_lofo", "Unseen-family router escalation rate", MUTED, "--")):
    e = esc_levels[k]
    ax.plot(ratios, [saving_at_ratio(e, r) * 100 for r in ratios], color=col, ls=ls, lw=2,
            label=f"{lab} ({e * 100:.1f}% to frontier)")
ymin = min(saving_at_ratio(esc_levels["router_lofo"], 1) * 100, -5)
for i, pr in enumerate(pair_res):
    ax.axvline(pr["blended_price_ratio"], color=FAINT, lw=1)
    ax.text(pr["blended_price_ratio"] * 1.03, 5 + (i % 3) * 9, f"{pr['pair']}\n{pr['blended_price_ratio']:.1f}x",
            fontsize=7.5, color=MUTED, rotation=0)
ax.set_xscale("log")
ax.set_ylim(min(ymin, -5), 100)
ax.set_xlabel("Blended frontier-to-cheap price ratio at the SPROUT token mix (log scale)")
ax.set_ylabel("Saving vs all-frontier traffic (%)")
ax.set_title("Beyond a 20x price gap, the escalation rate, not the price gap, sets the saving (98% target)")
ax.legend(frameon=False, fontsize=9, loc="upper left")
source_line(fig, " Assumes equal tokens per query across tiers.")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(CHARTS, "chart4.png"), dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------- results.json
results = {
    "dataset": {"name": "SPROUT", "source": "https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT",
                "paper": "arXiv:2502.03261", "n_train": TR["n"], "n_validation": VA["n"], "n_test": TE["n"],
                "test_family_counts": {f: int((TE["fam"] == f).sum()) for f in sorted(set(TE["fam"]))}},
    "prices_accessed": "2026-10-21",
    "pool": [LABEL[m] for m in POOL],
    "router_expected_cost_usd_per_1k": {LABEL[m]: r4(EXP_COST[j] * 1000) for j, m in enumerate(POOL)},
    "rq1": {"single_models": single, "frontier_usd_per_1k": r4(cF * 1000), "frontier_score": r4(qF),
            "headroom_vs_gpt4o": headroom, "oracle_best_per_query_score": r4(oracle_max_q),
            "oracle_best_retention": r4(oracle_max_q / qF),
            "oracle_cost_at_100pct_usd_per_1k": rq2_pool["100pct"]["oracle_usd_per_1k"],
            "price_ratio_gpt4o_to_llama70b_blended": r4(cF / TE["C"][:, jL].mean()),
            "price_ratio_gpt4o_to_llama8b_blended": r4(cF / TE["C"][:, POOL.index("wxai-llama-3-1-8b-instruct")].mean())},
    "rq2_pool_router": rq2_pool,
    "rq2_two_tier": rq2_two_tier,
    "rq2_family_at_98pct": fam_res,
    "rq3_router_quality": {"setting": "two-tier GPT-4o / Llama-3.3-70B, 98% target, thresholds on test",
                           "noise_curve": noise_curve, "points": router_pts},
    "rq3_price_gap": {"token_mix_input": r4(mix_in), "token_mix_output": r4(mix_out),
                      "escalation_rates_98pct": esc_levels, "saving_ceiling": ceiling, "pairs": pair_res},
}


def no_nulls(o, path="results"):
    if o is None or (isinstance(o, float) and (np.isnan(o) or np.isinf(o))):
        raise ValueError("null at " + path)
    if isinstance(o, dict):
        for k, v in o.items():
            no_nulls(v, path + "." + str(k))
    if isinstance(o, list):
        for i, v in enumerate(o):
            no_nulls(v, f"{path}[{i}]")


no_nulls(results)
with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=1)[:6000])
