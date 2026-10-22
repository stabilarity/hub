"""
Step 2 of 2: can a third party verify "was X in the training data?" from weights alone?

Reads data/scores.csv (written by score.py) and the committed text samples,
then computes:
  RQ1  document-level membership AUC on matched (IID) Pile train-vs-test sets
  RQ2  how a date-shifted non-member set inflates AUC, against model-free
       ("blind") baselines that never look at the model
  RQ3  set-level dataset inference: power and false-positive rate vs. set size,
       with a matched and with a date-shifted validation set
Writes charts/chart1..4.png and results.json. Offline, deterministic.

    python3 analysis.py
"""
import json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from sets import load_sets

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(HERE, "charts")
os.makedirs(CH, exist_ok=True)
SEED = 20261021
N_BOOT = 1000
N_TRIALS = 500
SET_SIZES = [25, 50, 100, 150, 200]
ALPHA = 0.05

INK, MID, LIGHT, RULE = "#111111", "#555555", "#bbbbbb", "#dddddd"
RED, GREEN = "#c62828", "#2e7d32"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MID,
                     "axes.labelcolor": INK, "xtick.color": MID, "ytick.color": MID,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
SRC = "Source: Stabilarity analysis of Pythia checkpoints on Pile train/test samples, fresh 2026 Wikipedia and WikiMIA (data/scores.csv)."

# membership score orientation: higher = "more likely a member"
ATTACKS = {"loss": ("Loss", -1), "zlib_ratio": ("Zlib ratio", -1),
           "mink": ("Min-K% (k=20%)", 1), "minkpp": ("Min-K%++ (k=20%)", 1)}
MODELS = ["pythia-70m", "pythia-160m", "pythia-410m"]
TARGET = "pythia-410m"

df = pd.read_csv(os.path.join(HERE, "data", "scores.csv"))
texts = [t for _, _, t in load_sets()]
env = json.load(open(os.path.join(HERE, "data", "scoring_env.json")))
rng = np.random.default_rng(SEED)


def auc_ci(y, s):
    y, s = np.asarray(y), np.asarray(s)
    a = roc_auc_score(y, s)
    r = np.random.default_rng(SEED)
    bs = []
    for _ in range(N_BOOT):
        i = r.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        bs.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(a), float(lo), float(hi)


def tpr_at(y, s, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_target].max())


def comparison(model, pos_set, neg_set):
    d = df[(df.model == model) & (df.set.isin([pos_set, neg_set]))]
    y = (d.set == pos_set).astype(int).values
    return d, y


# ---------------------------------------------------------------- comparisons
COMPS = {
    "matched_wikipedia": ("pile_train_wiki", "pile_test_wiki", "Wikipedia: Pile train vs Pile test"),
    "matched_arxiv": ("pile_train_arxiv", "pile_test_arxiv", "ArXiv: Pile train vs Pile test"),
    "matched_pilecc": ("pile_train_cc", "pile_test_cc", "Pile-CC: Pile train vs Pile test"),
    "shift_member_vs_fresh2026": ("pile_train_wiki", "fresh_wiki_2026", "Pile train Wikipedia vs Wikipedia created Oct 2026"),
    "shift_nonmember_vs_fresh2026": ("pile_test_wiki", "fresh_wiki_2026", "Pile test Wikipedia (non-members) vs Wikipedia created Oct 2026"),
    "wikimia_64": ("wikimia_64", "wikimia_64", "WikiMIA, 64 words"),
    "wikimia_128": ("wikimia_128", "wikimia_128", "WikiMIA, 128 words"),
}


def comp_frame(model, key):
    p, n, _ = COMPS[key]
    if key.startswith("wikimia"):
        d = df[(df.model == model) & (df.set == p)]
        return d, d.member.values
    return comparison(model, p, n)


YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")


def latest_year(t):
    ys = [int(y) for y in YEAR.findall(t) if int(y) <= 2026]
    return max(ys) if ys else 0


def blind_scores(idx, y):
    tx = [texts[i] for i in idx]
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(C=1.0, max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    bow = cross_val_predict(pipe, tx, y, cv=cv, method="predict_proba")[:, 1]
    yr = np.array([latest_year(t) for t in tx], dtype=float)
    return bow, yr


results = {"environment": env, "settings": {"seed": SEED, "bootstrap": N_BOOT, "dataset_inference_trials": N_TRIALS,
           "set_sizes": SET_SIZES, "alpha": ALPHA, "max_tokens_scored": 256, "k_fraction": 0.2,
           "target_model": TARGET}, "counts": {}, "doc_level": {}, "blind": {}}
for s, g in df[df.model == TARGET].groupby("set"):
    results["counts"][s] = {"n": int(len(g)), "members": int(g.member.sum()),
                            "median_tokens": float(g.n_tokens.median())}

for key in COMPS:
    results["doc_level"][key] = {"label": COMPS[key][2]}
    for m in MODELS:
        d, y = comp_frame(m, key)
        row = {}
        for a, (_, sign) in ATTACKS.items():
            auc, lo, hi = auc_ci(y, sign * d[a].values)
            row[a] = {"auc": round(auc, 4), "ci95": [round(lo, 4), round(hi, 4)],
                      "tpr_at_1pct_fpr": round(tpr_at(y, sign * d[a].values), 4)}
        best = max(row, key=lambda k: row[k]["auc"])
        row["best_attack"] = best
        results["doc_level"][key][m] = row
    d, y = comp_frame(TARGET, key)
    bow, yr = blind_scores(d.doc.values, y)
    zl = -d.zlib_bytes_per_char.values  # model-free compressibility
    results["blind"][key] = {}
    for name, s in [("bag_of_words_cv", bow), ("latest_year_mentioned", yr), ("compressibility", zl)]:
        auc, lo, hi = auc_ci(y, s)
        # orientation-free for the single-feature baselines: an auditor would pick the better sign
        if name != "bag_of_words_cv" and auc < 0.5:
            auc, lo, hi = 1 - auc, 1 - hi, 1 - lo
        results["blind"][key][name] = {"auc": round(auc, 4), "ci95": [round(lo, 4), round(hi, 4)]}

# ---------------------------------------------------------------- dataset inference (RQ3)
FEATS = list(ATTACKS)


def feat(set_name):
    d = df[(df.model == TARGET) & (df.set == set_name)]
    X = d[FEATS].values * np.array([ATTACKS[a][1] for a in FEATS])
    return X


def di_test(S, V, r):
    """Maini et al.-style dataset inference: learn a weighting of MIA features on
    one half, t-test suspect > validation on the other half."""
    iS, iV = r.permutation(len(S)), r.permutation(len(V))
    hS, hV = len(S) // 2, len(V) // 2
    A = np.vstack([S[iS[:hS]], V[iV[:hV]]])
    ya = np.r_[np.ones(hS), np.zeros(hV)]
    sc = StandardScaler().fit(A)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(A), ya)
    sS = clf.decision_function(sc.transform(S[iS[hS:]]))
    sV = clf.decision_function(sc.transform(V[iV[hV:]]))
    t, p = stats.ttest_ind(sS, sV, equal_var=False, alternative="greater")
    return p


def run_di(member_set, nonmember_set, n, r, h1=True, shifted_val=None):
    M, N = feat(member_set), feat(nonmember_set)
    rej, ps = 0, []
    for _ in range(N_TRIALS):
        if shifted_val is not None:
            Vp = feat(shifted_val)
            V = Vp[r.choice(len(Vp), n, replace=False)]
            src = M if h1 else N
            S = src[r.choice(len(src), n, replace=False)]
        else:
            perm = r.permutation(len(N))
            V = N[perm[:n]]
            S = M[r.choice(len(M), n, replace=False)] if h1 else N[perm[n:2 * n]]
        p = di_test(S, V, r)
        ps.append(p)
        rej += p < ALPHA
    return round(rej / N_TRIALS, 4), round(float(np.median(ps)), 4)


results["dataset_inference"] = {}
r = np.random.default_rng(SEED)
for dom, key in [("wikipedia", "wiki"), ("arxiv", "arxiv"), ("pilecc", "cc")]:
    out = {"power": {}, "false_positive_rate": {}, "median_p_h1": {}, "median_p_h0": {}}
    for n in SET_SIZES:
        pw, mp1 = run_di(f"pile_train_{key}", f"pile_test_{key}", n, r, h1=True)
        fp, mp0 = run_di(f"pile_train_{key}", f"pile_test_{key}", n, r, h1=False)
        out["power"][str(n)], out["median_p_h1"][str(n)] = pw, mp1
        out["false_positive_rate"][str(n)], out["median_p_h0"][str(n)] = fp, mp0
    results["dataset_inference"][f"matched_{dom}"] = out
out = {"power": {}, "false_positive_rate": {}}
for n in SET_SIZES:
    out["power"][str(n)] = run_di("pile_train_wiki", "pile_test_wiki", n, r, h1=True, shifted_val="fresh_wiki_2026")[0]
    out["false_positive_rate"][str(n)] = run_di("pile_train_wiki", "pile_test_wiki", n, r, h1=False, shifted_val="fresh_wiki_2026")[0]
results["dataset_inference"]["shifted_validation_wikipedia"] = out

# ---------------------------------------------------------------- headline summaries
dl = results["doc_level"]
matched = ["matched_wikipedia", "matched_arxiv", "matched_pilecc"]
summ = {}
summ["matched_best_auc_410m"] = {k: dl[k][TARGET][dl[k][TARGET]["best_attack"]]["auc"] for k in matched}
summ["matched_mean_auc_410m_by_attack"] = {a: round(float(np.mean([dl[k][TARGET][a]["auc"] for k in matched])), 4) for a in ATTACKS}
summ["matched_max_auc_any_model_attack"] = round(float(max(dl[k][m][a]["auc"] for k in matched for m in MODELS for a in ATTACKS)), 4)
summ["matched_mean_best_auc_by_model"] = {m: round(float(np.mean([max(dl[k][m][a]["auc"] for a in ATTACKS) for k in matched])), 4) for m in MODELS}
summ["shift_best_auc_by_model"] = {k: {m: dl[k][m][dl[k][m]["best_attack"]]["auc"] for m in MODELS}
                                   for k in ["shift_member_vs_fresh2026", "shift_nonmember_vs_fresh2026", "wikimia_64", "wikimia_128"]}
summ["inflation_points_410m"] = round(dl["shift_member_vs_fresh2026"][TARGET][dl["shift_member_vs_fresh2026"][TARGET]["best_attack"]]["auc"]
                                      - summ["matched_best_auc_410m"]["matched_wikipedia"], 4)
results["summary"] = summ

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(results, f, indent=2)


def has_null(o):
    if o is None or (isinstance(o, float) and np.isnan(o)):
        return True
    if isinstance(o, dict):
        return any(has_null(v) for v in o.values())
    if isinstance(o, list):
        return any(has_null(v) for v in o)
    return False


assert not has_null(results), "results.json contains nulls"

# ---------------------------------------------------------------- charts
def finish(fig, ax, title, fname):
    ax.set_title(title, loc="left", fontsize=14, color=INK, pad=14)
    fig.text(0.01, 0.01, SRC, fontsize=8, color=MID)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(CH, fname), dpi=160)
    plt.close(fig)


# chart1: matched AUC per attack and domain (410m) with CI
fig, ax = plt.subplots(figsize=(10, 5.6))
shades = [INK, MID, LIGHT, "#eeeeee"]
hatches = ["", "", "", "//"]
w = 0.2
x = np.arange(len(matched))
for j, (a, (lab, _)) in enumerate(ATTACKS.items()):
    v = [dl[k][TARGET][a]["auc"] for k in matched]
    lo = [v[i] - dl[k][TARGET][a]["ci95"][0] for i, k in enumerate(matched)]
    hi = [dl[k][TARGET][a]["ci95"][1] - v[i] for i, k in enumerate(matched)]
    ax.bar(x + (j - 1.5) * w, v, w * 0.9, color=shades[j], edgecolor=MID, hatch=hatches[j], label=lab,
           yerr=[lo, hi], error_kw={"ecolor": MID, "capsize": 3, "lw": 1})
ax.axhline(0.5, color=RED, lw=1.5, ls="--")
ax.text(-0.45, 0.403, "dashed line: random guess = 0.5", color=RED, ha="left", va="bottom", fontsize=9)
ax.set_xticks(x, ["Wikipedia", "ArXiv", "Pile-CC"])
ax.set_ylim(0.4, 0.7)
ax.set_ylabel("ROC AUC (95% bootstrap CI)")
ax.yaxis.grid(True, color=RULE, lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=9)
finish(fig, ax, "Matched members vs non-members: Pythia-410M, 400 vs 400 documents per domain", "chart1.png")

# chart2: shift inflation, best MIA vs blind baselines
keys2 = ["matched_wikipedia", "shift_member_vs_fresh2026", "shift_nonmember_vs_fresh2026", "wikimia_64", "wikimia_128"]
labs2 = ["Matched\nPile train vs test", "Pile train vs\nfresh 2026", "Pile TEST vs\nfresh 2026\n(no members)", "WikiMIA\n64 words", "WikiMIA\n128 words"]
fig, ax = plt.subplots(figsize=(10, 5.6))
x = np.arange(len(keys2))
mia = [dl[k][TARGET][dl[k][TARGET]["best_attack"]]["auc"] for k in keys2]
bow = [results["blind"][k]["bag_of_words_cv"]["auc"] for k in keys2]
yr = [results["blind"][k]["latest_year_mentioned"]["auc"] for k in keys2]
ax.bar(x - 0.27, mia, 0.25, color=INK, label="Best MIA on Pythia-410M")
ax.bar(x, bow, 0.25, color=LIGHT, edgecolor=MID, label="Blind: bag-of-words classifier (no model)")
ax.bar(x + 0.27, yr, 0.25, color="#eeeeee", edgecolor=MID, hatch="//", label="Blind: latest year in text (no model)")
for i in range(len(keys2)):
    ax.text(x[i] - 0.27, mia[i] + 0.01, f"{mia[i]:.2f}", ha="center", fontsize=9, color=INK)
    ax.text(x[i], bow[i] + 0.01, f"{bow[i]:.2f}", ha="center", fontsize=9, color=INK)
ax.axhline(0.5, color=RED, lw=1.5, ls="--")
ax.set_xticks(x, labs2, fontsize=9)
ax.set_ylim(0.3, 1.22)
ax.set_yticks(np.arange(0.3, 1.01, 0.1))
ax.set_ylabel("ROC AUC")
ax.yaxis.grid(True, color=RULE, lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=3)
finish(fig, ax, "Date-shifted non-member sets: the model is not needed to score high", "chart2.png")

# chart3: AUC vs model size, matched vs shifted
params = {m: env["models"]["EleutherAI/" + m]["params"] / 1e6 for m in MODELS}
fig, ax = plt.subplots(figsize=(10, 5.6))
series = [("Matched, mean of 3 domains (best attack)", [summ["matched_mean_best_auc_by_model"][m] for m in MODELS], INK, "o", "-"),
          ("Pile train vs fresh 2026 Wikipedia", [summ["shift_best_auc_by_model"]["shift_member_vs_fresh2026"][m] for m in MODELS], MID, "s", "-"),
          ("Pile test vs fresh 2026 (no members)", [summ["shift_best_auc_by_model"]["shift_nonmember_vs_fresh2026"][m] for m in MODELS], MID, "^", "--"),
          ("WikiMIA, 64 words", [summ["shift_best_auc_by_model"]["wikimia_64"][m] for m in MODELS], LIGHT, "D", "-")]
xs = [params[m] for m in MODELS]
for lab, v, c, mk, ls in series:
    ax.plot(xs, v, color=c, marker=mk, ms=8, lw=2, ls=ls, label=lab)
    dy = {"s": 0.012, "^": -0.012}.get(mk, 0)
    ax.text(xs[-1] * 1.06, v[-1] + dy, f"{v[-1]:.2f}", va="center", fontsize=9, color=INK)
ax.axhline(0.5, color=RED, lw=1.5, ls="--")
ax.set_xscale("log")
ax.set_xticks(xs, [f"Pythia-{m.split('-')[1].upper()}\n({p:.1f}M parameters)" for m, p in zip(MODELS, xs)])
ax.minorticks_off()
ax.set_xlim(xs[0] * 0.8, xs[-1] * 1.4)
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("ROC AUC, best of 4 attacks")
ax.yaxis.grid(True, color=RULE, lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", fontsize=9)
finish(fig, ax, "Model size moves matched AUC little; the shift gap stays", "chart3.png")

# chart4: dataset inference power / FPR vs n
fig, ax = plt.subplots(figsize=(10, 5.6))
di = results["dataset_inference"]
for dom, lab, c, mk in [("matched_wikipedia", "Wikipedia", INK, "o"), ("matched_arxiv", "ArXiv", MID, "s"), ("matched_pilecc", "Pile-CC", LIGHT, "D")]:
    ax.plot(SET_SIZES, [di[dom]["power"][str(n)] for n in SET_SIZES], color=c, marker=mk, ms=8, lw=2, label=f"Power, matched validation: {lab}")
    ax.plot(SET_SIZES, [di[dom]["false_positive_rate"][str(n)] for n in SET_SIZES], color=c, marker=mk, ms=6, lw=1.2, ls=":", label=f"False positives, matched: {lab}")
ax.plot(SET_SIZES, [di["shifted_validation_wikipedia"]["power"][str(n)] for n in SET_SIZES], color=RED, marker="o", ms=6, lw=1.2, ls="--",
        label="Power, 2026 validation set (Wikipedia)")
ax.plot(SET_SIZES, [di["shifted_validation_wikipedia"]["false_positive_rate"][str(n)] for n in SET_SIZES], color=RED, marker="x", ms=9, lw=2,
        label="False positives, 2026 validation set (Wikipedia)")
ax.axhline(ALPHA, color=GREEN, lw=1.2, ls="--")
ax.text(SET_SIZES[-1], ALPHA + 0.015, "nominal alpha = 0.05", color=GREEN, ha="right", fontsize=9)
ax.set_xlabel("Documents in suspect set and in validation set (each)")
ax.set_ylabel("Share of 500 trials with p < 0.05")
ax.set_ylim(-0.02, 1.05)
ax.yaxis.grid(True, color=RULE, lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="center right")
finish(fig, ax, "Set-level dataset inference on Pythia-410M: power, and false accusations", "chart4.png")

print(json.dumps(results["summary"], indent=2))
print(json.dumps({k: v for k, v in results["blind"].items()}, indent=1))
print(json.dumps(results["dataset_inference"], indent=1))
