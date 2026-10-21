#!/usr/bin/env python3
"""Explainability-as-a-Service: managed XAI platforms vs in-house observability stacks.

Three analyses, all offline and deterministic:

1. Latency benchmark (RQ1): single-row explanation latency of the estimators that managed services expose
   (Kernel SHAP, as in SageMaker Clarify; sampled Shapley with a path count, as in Vertex Explainable AI) against
   what an in-house stack can run (exact TreeSHAP, gradient attributions), on two public tabular datasets.
2. Fidelity (RQ2): error of the sampling estimators against exact Shapley values for the same model, baseline and
   value function, as a function of their sample budget.
3. TCO model (RQ3): monthly cost of span-priced managed LLM-observability plans (list prices read 2026-10-21)
   against a self-hosted open-source stack (infrastructure + engineering time), across traffic volumes;
   plus documented retention against the six-month log-retention floor of the EU AI Act (Art. 19 and 26(6)).

Run:  python3 analysis.py      (writes charts/chart1-4.png and results.json)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import gzip
import itertools
import json
import math
import platform
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import sklearn
import torch
import xgboost as xgb
from captum.attr import GradientShap, IntegratedGradients, Saliency
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 20261021
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CHARTS = HERE / "charts"
CHARTS.mkdir(exist_ok=True)
torch.set_num_threads(1)
torch.manual_seed(SEED)
RNG = np.random.default_rng(SEED)

N_EXPLAIN = 40          # test rows explained per method
REPEATS = 3             # timing repeats per row; the median is kept
KERNEL_BUDGETS = [64, 256, 1024, 4096]
PATH_COUNTS = [1, 5, 10, 25, 50]   # Vertex sampled Shapley accepts path count 1..50, suggests 25

# Site palette (greys) for charts
INK, MID, LIGHT, GRID = "#111111", "#555555", "#bbbbbb", "#eeeeee"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MID, "axes.labelcolor": INK,
    "xtick.color": MID, "ytick.color": MID, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
})


# ----------------------------------------------------------------------------------------------- data
def load_adult():
    cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
            "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country",
            "income"]
    rows = []
    with gzip.open(DATA / "adult.data.gz", "rt") as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) == len(cols):
                rows.append(parts)
    cat = {1, 3, 5, 6, 7, 8, 9, 13}
    X = np.zeros((len(rows), 14), dtype=np.float64)
    for j in range(14):
        col = [r[j] for r in rows]
        if j in cat:
            levels = {v: k for k, v in enumerate(sorted(set(col)))}   # ordinal codes, sorted -> deterministic
            X[:, j] = [levels[v] for v in col]
        else:
            X[:, j] = [float(v) for v in col]
    y = np.array([1 if r[14] == ">50K" else 0 for r in rows])
    return X, y, cols[:14]


def load_bc():
    d = load_breast_cancer()
    return d.data.astype(np.float64), d.target, list(d.feature_names)


# ------------------------------------------------------------------------------------------ helpers
def timed(fn, x):
    """Median wall time in ms of fn(x) over REPEATS runs, plus the last result."""
    ts, out = [], None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = fn(x)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts)), out


def exact_baseline_shapley(f, x, b):
    """Exact Shapley values of f at x relative to a single baseline b (all 2^M coalitions; M <= 16)."""
    M = len(x)
    masks = np.array(list(itertools.product([0, 1], repeat=M)), dtype=bool)
    Z = np.where(masks, x[None, :], b[None, :])
    v = f(Z)
    idx = masks.dot(1 << np.arange(M - 1, -1, -1))
    val = np.empty(1 << M)
    val[idx] = v
    sizes = masks.sum(1)
    w = np.array([math.factorial(s) * math.factorial(M - s - 1) / math.factorial(M) if s < M else 0.0
                  for s in range(M + 1)])
    phi = np.zeros(M)
    for i in range(M):
        bit = 1 << (M - 1 - i)
        without = idx[~masks[:, i]]
        phi[i] = np.sum(w[sizes[~masks[:, i]]] * (val[without | bit] - val[without]))
    return phi


def sampled_shapley(f, x, b, paths, rng):
    """Permutation-sampling Shapley estimator with a single baseline (the method behind 'path count')."""
    M = len(x)
    Z = np.empty((paths * (M + 1), M))
    perms = [rng.permutation(M) for _ in range(paths)]
    r = 0
    for p in perms:
        z = b.copy()
        Z[r] = z; r += 1
        for j in p:
            z[j] = x[j]
            Z[r] = z; r += 1
    v = f(Z).reshape(paths, M + 1)
    phi = np.zeros(M)
    for k, p in enumerate(perms):
        phi[p] += np.diff(v[k])
    return phi / paths


def fidelity(est, ref):
    """Normalised MAE, Spearman rank correlation and top-3 overlap of |phi| against the exact reference."""
    nmae, rho, top3 = [], [], []
    for e, r in zip(est, ref):
        denom = np.mean(np.abs(r))
        nmae.append(np.mean(np.abs(e - r)) / denom if denom > 0 else 0.0)
        rr = spearmanr(np.abs(e), np.abs(r)).correlation
        rho.append(0.0 if np.isnan(rr) else rr)
        top3.append(len(set(np.argsort(-np.abs(e))[:3]) & set(np.argsort(-np.abs(r))[:3])) / 3.0)
    return float(np.mean(nmae)), float(np.mean(rho)), float(np.mean(top3))


# ------------------------------------------------------------------------------------- benchmark
def bench_tree(name, X, y, feat):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9,
                              colsample_bytree=0.9, random_state=SEED, n_jobs=1, tree_method="hist")
    model.fit(Xtr, ytr)
    booster = model.get_booster()
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    baseline = np.median(Xtr, axis=0)

    def f(Z):  # log-odds output, the scale TreeSHAP explains
        return booster.inplace_predict(np.asarray(Z, dtype=np.float64), predict_type="margin")

    rows = Xte[RNG.choice(len(Xte), N_EXPLAIN, replace=False)]
    M = X.shape[1]
    res = {"dataset": name, "n_rows": int(len(X)), "n_features": int(M), "test_auc": round(float(auc), 4),
           "methods": {}}

    lat = {}
    lat["predict"] = [timed(lambda r: f(r[None, :]), r)[0] for r in rows]

    dm = lambda r: xgb.DMatrix(r[None, :])
    lat["TreeSHAP path-dependent (xgboost native)"] = [
        timed(lambda r: booster.predict(dm(r), pred_contribs=True), r)[0] for r in rows]

    tex = shap.TreeExplainer(booster, data=baseline[None, :], feature_perturbation="interventional",
                             model_output="raw")
    ref, t = [], []
    for r in rows:
        ms, out = timed(lambda q: tex.shap_values(q[None, :], check_additivity=False), r)
        t.append(ms); ref.append(np.asarray(out).reshape(-1))
    lat["TreeSHAP exact, 1 baseline (reference)"] = t
    ref = np.array(ref)

    # sanity check: exact enumeration agrees with interventional TreeSHAP when feasible
    if M <= 16:
        chk = exact_baseline_shapley(f, rows[0], baseline)
        res["reference_check_max_abs_diff"] = float(np.max(np.abs(chk - ref[0])))

    kex = shap.KernelExplainer(f, baseline[None, :])
    for ns in KERNEL_BUDGETS:
        est, t = [], []
        for r in rows:
            np.random.seed(SEED)
            ms, out = timed(lambda q: kex.shap_values(q[None, :], nsamples=ns, l1_reg=False, silent=True), r)
            t.append(ms); est.append(np.asarray(out).reshape(-1))
        key = f"Kernel SHAP, {ns} samples"
        lat[key] = t
        nmae, rho, top3 = fidelity(np.array(est), ref)
        res["methods"][key] = {"family": "kernel", "budget": ns, "nmae": round(nmae, 4), "spearman": round(rho, 4),
                               "top3_overlap": round(top3, 4)}

    # the library's default l1_reg="auto" adds AIC feature selection when samples < 20% of all coalitions
    est, t = [], []
    for r in rows:
        np.random.seed(SEED)
        ms, out = timed(lambda q: kex.shap_values(q[None, :], nsamples=1024, silent=True), r)
        t.append(ms); est.append(np.asarray(out).reshape(-1))
    key = "Kernel SHAP, 1024 samples, default l1_reg"
    lat[key] = t
    nmae, rho, top3 = fidelity(np.array(est), ref)
    res["methods"][key] = {"family": "kernel_default", "budget": 1024, "nmae": round(nmae, 4),
                           "spearman": round(rho, 4), "top3_overlap": round(top3, 4)}

    for pc in PATH_COUNTS:
        est, t = [], []
        rng = np.random.default_rng(SEED + pc)
        for r in rows:
            ms, out = timed(lambda q: sampled_shapley(f, q, baseline, pc, rng), r)
            t.append(ms); est.append(out)
        key = f"Sampled Shapley, path count {pc}"
        lat[key] = t
        nmae, rho, top3 = fidelity(np.array(est), ref)
        res["methods"][key] = {"family": "sampled", "budget": pc, "nmae": round(nmae, 4), "spearman": round(rho, 4),
                               "top3_overlap": round(top3, 4)}

    pred_ms = float(np.median(lat["predict"]))
    for k, v in lat.items():
        d = res["methods"].setdefault(k, {"family": "other"})
        d["median_ms"] = round(float(np.median(v)), 4)
        d["p95_ms"] = round(float(np.percentile(v, 95)), 4)
        d["x_predict"] = round(float(np.median(v)) / pred_ms, 1)
    return res


class MLP(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(m, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                                       torch.nn.ReLU(), torch.nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def bench_mlp(name, X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
    net = MLP(X.shape[1]).double()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt, yt = torch.tensor(Xtr_s), torch.tensor(ytr, dtype=torch.float64)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(15):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b]), yt[b])
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        auc = roc_auc_score(yte, net(torch.tensor(Xte_s)).numpy())
    baseline = np.median(Xtr_s, axis=0)

    def f(Z):
        with torch.no_grad():
            return net(torch.as_tensor(np.asarray(Z, dtype=np.float64))).numpy()

    rows = Xte_s[np.random.default_rng(SEED).choice(len(Xte_s), N_EXPLAIN, replace=False)]
    bt = torch.tensor(baseline[None, :])
    ref = np.array([exact_baseline_shapley(f, r, baseline) for r in rows])
    res = {"dataset": name, "n_features": int(X.shape[1]), "test_auc": round(float(auc), 4), "methods": {}}

    ig, gs, sal = IntegratedGradients(net), GradientShap(net), Saliency(net)
    torch.manual_seed(SEED)
    fns = {
        "predict": lambda r: f(r[None, :]),
        "Saliency (gradient)": lambda r: sal.attribute(torch.tensor(r[None, :]), abs=False).numpy()[0],
        "Input x Gradient (vs baseline)": lambda r: ((r - baseline) * sal.attribute(torch.tensor(r[None, :]),
                                                                                   abs=False).numpy()[0]),
        "Integrated Gradients, 32 steps": lambda r: ig.attribute(torch.tensor(r[None, :]), baselines=bt,
                                                                   n_steps=32).numpy()[0],
        "GradientSHAP, 16 samples": lambda r: gs.attribute(torch.tensor(r[None, :]), baselines=bt, n_samples=16,
                                                           stdevs=0.0).numpy()[0],
        "Exact Shapley, 2^14 coalitions": lambda r: exact_baseline_shapley(f, r, baseline),
    }
    kex = shap.KernelExplainer(f, baseline[None, :])
    fns["Kernel SHAP, 1024 samples"] = lambda r: np.asarray(kex.shap_values(r[None, :], nsamples=1024,
                                                                             l1_reg=False, silent=True)).reshape(-1)
    rng = np.random.default_rng(SEED)
    fns["Sampled Shapley, path count 25"] = lambda r: sampled_shapley(f, r, baseline, 25, rng)

    pred_ms = None
    for k, fn in fns.items():
        t, est = [], []
        for r in rows:
            np.random.seed(SEED)
            ms, out = timed(fn, r)
            t.append(ms); est.append(np.asarray(out).reshape(-1))
        d = {"median_ms": round(float(np.median(t)), 4), "p95_ms": round(float(np.percentile(t, 95)), 4)}
        if k == "predict":
            pred_ms = float(np.median(t))
        else:
            nmae, rho, top3 = fidelity(np.array(est), ref)
            d.update({"nmae": round(nmae, 4), "spearman": round(rho, 4), "top3_overlap": round(top3, 4)})
        d["x_predict"] = round(float(np.median(t)) / pred_ms, 1)
        res["methods"][k] = d
    return res


# ------------------------------------------------------------------------------------------- TCO
def read_csv(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


BLS_MEDIAN_WAGE = 135980.0   # US software developers, May 2025 (BLS OOH)
HOURS_PER_MONTH = 730.0


def tco(results):
    pricing = read_csv(DATA / "managed_pricing.csv")
    infra = {r["item"]: float(r["usd_per_hour"]) for r in read_csv(DATA / "infra_prices.csv")}
    node_small = infra["m7i.xlarge on-demand Linux us-east-1"]
    node_big = infra["m7i.2xlarge on-demand Linux us-east-1"]

    A = {  # labelled assumptions (our estimates), varied in the sensitivity analysis
        "llm_spans_per_request": 4, "other_spans_per_request": 6, "scores_per_request": 1,
        "base_nodes_m7i_xlarge": 3,                 # web+worker, ClickHouse, Postgres/Redis
        "extra_m7i_2xlarge_per_100M_units": 1,      # ClickHouse/worker scale-out
        "storage_and_backup_uplift": 0.25,          # EBS/S3/backup as a share of compute
        "fte_share": 0.25, "fte_low": 0.10, "fte_high": 0.50, "loading_factor": 1.3,
    }

    def units(R, a):
        return R * (1 + a["llm_spans_per_request"] + a["other_spans_per_request"] + a["scores_per_request"])

    def managed_cost(plan, R, a):
        if plan["billing_unit"] == "LLM span":
            u = R * a["llm_spans_per_request"]
        else:
            u = units(R, a)
        inc, per = float(plan["included_units"]), float(plan["overage_per_units"])
        if plan.get("graduated_tiers"):  # Langfuse calculator: graduated $/100k above the included units
            cost, lo = 0.0, inc
            for tier in plan["graduated_tiers"].split(";"):
                cap, rate = tier.split(":")
                hi = float(cap)
                if u > lo:
                    cost += (min(u, hi) - lo) / per * float(rate)
                lo = hi
            return float(plan["base_usd_month"]) + cost
        over = max(0.0, u - inc)
        return float(plan["base_usd_month"]) + math.ceil(over / per) * float(plan["overage_usd"])

    def self_cost(R, a, fte):
        u = units(R, a)
        compute = (a["base_nodes_m7i_xlarge"] * node_small + math.ceil(u / 1e8) * a[
            "extra_m7i_2xlarge_per_100M_units"] * node_big) * HOURS_PER_MONTH
        infra_c = compute * (1 + a["storage_and_backup_uplift"])
        eng = fte * BLS_MEDIAN_WAGE * a["loading_factor"] / 12.0
        return infra_c, eng

    grid = np.unique(np.round(np.logspace(4, 8, 161)).astype(int))
    plans = [p for p in pricing if p["plan"] in ("Pro annual", "Pro on-demand", "Core")]
    curves = {f'{p["platform"]} {p["plan"]}': [managed_cost(p, R, A) for R in grid] for p in plans}
    sh = {k: [sum(self_cost(R, A, A[k2])) for R in grid] for k, k2 in
          (("self_base", "fte_share"), ("self_low", "fte_low"), ("self_high", "fte_high"))}

    def crossover(plan, a, fte):
        lo, hi = 1e3, 1e10
        if managed_cost(plan, hi, a) <= sum(self_cost(hi, a, fte)):
            return -1.0
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if managed_cost(plan, mid, a) > sum(self_cost(mid, a, fte)):
                hi = mid
            else:
                lo = mid
        return float(hi)

    cross = {}
    for p in plans:
        key = f'{p["platform"]} {p["plan"]}'
        cross[key] = {lab: round(crossover(p, A, A[k]) / 1e6, 3) for lab, k in
                      (("fte_0.25", "fte_share"), ("fte_0.10", "fte_low"), ("fte_0.50", "fte_high"))}
        for s in (2, 8):
            a2 = dict(A, llm_spans_per_request=s)
            cross[key][f"llm_spans_{s}_fte_0.25"] = round(crossover(p, a2, A["fte_share"]) / 1e6, 3)

    infra_c, eng = self_cost(1e6, A, A["fte_share"])
    at = {}
    for R in (1e5, 1e6, 1e7):
        at[f"{int(R)}"] = {k: round(managed_cost(p, R, A), 2) for p in plans
                           for k in [f'{p["platform"]} {p["plan"]}']}
        at[f"{int(R)}"]["self_hosted_fte_0.25"] = round(sum(self_cost(R, A, A["fte_share"])), 2)
    results["tco"] = {
        "assumptions": A, "bls_median_wage_usd": BLS_MEDIAN_WAGE, "node_price_usd_h": {"m7i.xlarge": node_small,
                                                                                      "m7i.2xlarge": node_big},
        "self_hosted_at_1M_requests": {"infra_usd_month": round(infra_c, 2), "engineering_usd_month": round(eng, 2)},
        "monthly_cost_usd": at, "crossover_million_requests_per_month": cross,
    }

    # retention vs the six-month floor
    ret = read_csv(DATA / "retention_coverage.csv")
    results["retention"] = [{"platform": r["platform"], "plan": r["plan"],
                             "default_days": int(r["default_retention_days"]),
                             "longest_listed_days": int(r["longest_listed_retention_days"]),
                             "meets_183_days_listed": int(r["longest_listed_retention_days"]) >= 183}
                            for r in ret]
    return grid, curves, sh, ret


# ----------------------------------------------------------------------------------------- charts
SRC = "Source: Stabilarity Research Hub analysis; code github.com/stabilarity/hub/tree/master/research/" \
      "explainability-as-a-service-comparing-managed-xai-platforms-against-in-house-observability-stacks"


def chart1(results):
    a, b = results["bench"]["adult_xgb"]["methods"], results["bench"]["bc_xgb"]["methods"]
    keys = ["predict", "TreeSHAP path-dependent (xgboost native)", "TreeSHAP exact, 1 baseline (reference)",
            "Sampled Shapley, path count 1", "Sampled Shapley, path count 25", "Sampled Shapley, path count 50",
            "Kernel SHAP, 256 samples", "Kernel SHAP, 1024 samples", "Kernel SHAP, 4096 samples"]
    fig, ax = plt.subplots(figsize=(10.7, 6), dpi=150)
    yy = np.arange(len(keys))
    ax.barh(yy + 0.2, [a[k]["median_ms"] for k in keys], height=0.38, color=INK, label="Adult (14 features)")
    ax.barh(yy - 0.2, [b[k]["median_ms"] for k in keys], height=0.38, color=LIGHT, edgecolor=MID, linewidth=0.6,
            label="Breast cancer (30 features)")
    for i, k in enumerate(keys):
        ax.text(a[k]["median_ms"] * 1.15, i + 0.2, f'{a[k]["median_ms"]:.3g} ms', va="center", fontsize=8.5, color=INK)
        ax.text(b[k]["median_ms"] * 1.15, i - 0.2, f'{b[k]["median_ms"]:.3g} ms', va="center", fontsize=8.5, color=MID)
    ax.set_yticks(yy, keys)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.01, 3e4)
    ax.set_xlabel("Median latency per single-row explanation, ms (log scale, one CPU thread)")
    ax.set_title("Explanation latency by estimator: XGBoost, 200 trees", loc="left", fontsize=13, color=INK)
    ax.legend(loc="upper right", frameon=False)
    fig.text(0.01, 0.01, SRC, fontsize=7, color=MID)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(CHARTS / "chart1.png"); plt.close(fig)


def chart2(results):
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 5.2), dpi=150, sharey=True)
    for ax, key, title in ((axes[0], "adult_xgb", "Adult, 14 features"), (axes[1], "bc_xgb", "Breast cancer, 30 features")):
        m = results["bench"][key]["methods"]
        for fam, mk, ls, lab in (("kernel", "o", "-", "Kernel SHAP (sample budget)"),
                                  ("sampled", "s", "--", "Sampled Shapley (path count)")):
            pts = sorted([(v["median_ms"], max(v["nmae"], 1e-4), v["budget"]) for v in m.values()
                          if v.get("family") == fam], key=lambda p: p[2])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ls=ls, lw=2, ms=8,
                    color=INK if fam == "kernel" else MID, label=lab)
            for p in pts:
                ax.annotate(str(p[2]), (p[0], p[1]), textcoords="offset points", xytext=(5, 5), fontsize=8, color=MID)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Median latency per explanation, ms (log)")
    axes[0].set_ylabel("Normalised MAE vs exact Shapley (log; floor 1e-4)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Cost-fidelity frontier of sampling Shapley estimators", x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.01, SRC, fontsize=7, color=MID)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(CHARTS / "chart2.png"); plt.close(fig)


def chart3(grid, curves, sh):
    fig, ax = plt.subplots(figsize=(10.7, 6), dpi=150)
    styles = {"Datadog Agent Observability Pro annual": ("-", INK),
              "Datadog Agent Observability Pro on-demand": (":", INK),
              "Langfuse Cloud Core": ("--", MID)}
    for k, v in curves.items():
        ls, c = styles[k]
        ax.plot(grid, v, ls=ls, color=c, lw=2, label=k + (" (graduated list price)" if "Langfuse" in k else " (list price)"))
    ax.fill_between(grid, sh["self_low"], sh["self_high"], color=GRID, label="Self-hosted, 0.10-0.50 FTE")
    ax.plot(grid, sh["self_base"], color=LIGHT, lw=2.5, label="Self-hosted, 0.25 FTE (base case)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Traced requests per month (log); 4 LLM spans + 6 other spans + 1 score per request")
    ax.set_ylabel("Monthly cost, USD (log)")
    ax.set_title("Managed LLM observability vs self-hosted stack: monthly cost", loc="left", fontsize=13, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.01, 0.01, "Prices: datadoghq.com/pricing, langfuse.com/pricing (2026-10-21); AWS EC2 on-demand "
             "m7i (2026-10-15); BLS median wage May 2025. Self-hosted figures are our estimates.", fontsize=7, color=MID)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(CHARTS / "chart3.png"); plt.close(fig)


def chart4(ret):
    labels = [f'{r["platform"]}\n{r["plan"]}' for r in ret]
    d = [int(r["default_retention_days"]) for r in ret]
    mx = [int(r["longest_listed_retention_days"]) for r in ret]
    fig, ax = plt.subplots(figsize=(10.7, 5.6), dpi=150)
    x = np.arange(len(ret))
    ax.bar(x - 0.2, d, width=0.38, color=LIGHT, edgecolor=MID, linewidth=0.6, label="Default retention")
    ax.bar(x + 0.2, mx, width=0.38, color=INK, label="Longest retention listed on pricing page")
    for i in range(len(ret)):
        ax.text(x[i] - 0.2, d[i] * 1.08, str(d[i]), ha="center", fontsize=8.5, color=MID)
        ax.text(x[i] + 0.2, mx[i] * 1.08, str(mx[i]), ha="center", fontsize=8.5, color=INK)
    ax.axhline(183, color="#c62828", lw=1.5, ls="--", label="Six months (about 183 days): AI Act Art. 19 / 26(6)")
    ax.set_yscale("log")
    ax.set_ylim(5, 3000)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("Trace retention, days (log)")
    ax.set_title("Documented trace retention vs the six-month log-retention floor", loc="left", fontsize=13, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.01, 0.01, "Sources: vendor pricing pages read 2026-10-21; Regulation (EU) 2024/1689. Langfuse Pro "
             "figure is its listed data access period.", fontsize=7, color=MID)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(CHARTS / "chart4.png"); plt.close(fig)


# ------------------------------------------------------------------------------------------- main
def main():
    results = {
        "environment": {
            "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")),
                        platform.processor() or "unknown"),
            "threads_used": 1, "python": platform.python_version(), "numpy": np.__version__,
            "shap": shap.__version__, "xgboost": xgb.__version__, "torch": torch.__version__,
            "scikit_learn": sklearn.__version__, "explained_rows_per_method": N_EXPLAIN, "timing_repeats": REPEATS,
            "seed": SEED,
        },
        "bench": {},
    }
    Xa, ya, _ = load_adult()
    Xb, yb, _ = load_bc()
    results["bench"]["adult_xgb"] = bench_tree("UCI Adult", Xa, ya, None)
    results["bench"]["bc_xgb"] = bench_tree("Wisconsin breast cancer", Xb, yb, None)
    results["bench"]["adult_mlp"] = bench_mlp("UCI Adult", Xa, ya)

    # cost per million explanations at the per-vCPU price of an m7i.xlarge (our estimate)
    vcpu_h = 0.2016 / 4
    for key in ("adult_xgb", "bc_xgb", "adult_mlp"):
        for k, v in results["bench"][key]["methods"].items():
            v["usd_per_million_on_one_vcpu"] = round(v["median_ms"] / 1000.0 * 1e6 / 3600.0 * vcpu_h, 4)

    grid, curves, sh, ret = tco(results)
    chart1(results); chart2(results); chart3(grid, curves, sh); chart4(ret)
    with open(HERE / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=1)[:6000])


if __name__ == "__main__":
    main()
