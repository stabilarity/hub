"""AI Data Readiness Score: controlled defect injection on 10 public tabular datasets.

Question: which of six training-data defects actually break tabular classifiers, and does an
empirically weighted readiness score predict the performance loss of unseen datasets with
random defect combinations better than an equal-weight checklist?

Run:  python3 analysis.py            (full run, about 3 min on 6 CPU cores)
      python3 analysis.py --tiny     (smoke test)
Reads data/*.csv.gz (built once by fetch_data.py), writes results.json and charts/chart1..6.png.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CHARTS = os.path.join(HERE, "charts")
TINY = "--tiny" in sys.argv

DATASETS = ["breast-cancer", "california-housing", "adult", "credit-g", "bank-marketing",
            "spambase", "diabetes", "phoneme", "magic-telescope", "blood-transfusion"]
DEFECTS = ["missing", "label_noise", "duplicates", "imbalance", "leakage", "sample_size"]
DEFECT_LABEL = {
    "missing": "Missing values (MCAR)",
    "label_noise": "Label noise",
    "duplicates": "Duplicates / contamination",
    "imbalance": "Class imbalance",
    "leakage": "Target leakage",
    "sample_size": "Reduced sample size",
}
DEFECT_UNIT = {
    "missing": "share of feature cells set missing",
    "label_noise": "share of training labels flipped",
    "duplicates": "duplicate rows added, share of rows",
    "imbalance": "share of minority-class rows removed",
    "leakage": "share of rows where a leaked proxy equals the label",
    "sample_size": "share of training rows removed",
}
GRID = {
    "missing": [0.05, 0.1, 0.2, 0.3, 0.4],
    "label_noise": [0.05, 0.1, 0.2, 0.3, 0.4],
    "duplicates": [0.05, 0.1, 0.2, 0.3, 0.4],
    "imbalance": [0.2, 0.4, 0.6, 0.8, 0.9],
    "leakage": [0.2, 0.4, 0.6, 0.8, 0.95],
    "sample_size": [0.2, 0.5, 0.75, 0.9, 0.98],
}
MODELS = ["logreg", "random_forest", "hist_gbm"]
MODEL_LABEL = {"logreg": "Logistic regression", "random_forest": "Random forest",
               "hist_gbm": "HistGradientBoosting"}
SEEDS = [0, 1, 2]
N_COMBOS = 30
TEST_SHARE = 0.3
DEV_VAL_SHARE = 0.2
REF_LEVEL = 0.2
N_JOBS = 6
if TINY:
    DATASETS = DATASETS[:2]
    GRID = {k: [0.2, v[-1]] for k, v in GRID.items()}
    SEEDS = [0]
    N_COMBOS = 3


# ---------------------------------------------------------------- data
def load(name):
    df = pd.read_csv(os.path.join(DATA, f"{name}.csv.gz"))
    y = df.pop("target").values.astype(int)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
        else:
            df[c] = df[c].astype(float)
    return df, y


def stratified_split(y, share, rng):
    """Return (idx_a, idx_b) with idx_b holding `share` of each class."""
    a, b = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = max(1, int(round(share * len(idx))))
        b.extend(idx[:k]); a.extend(idx[k:])
    return np.sort(np.array(a)), np.sort(np.array(b))


def keep_per_class(idx, y, frac_keep, rng, classes=None, min_keep=3):
    """Keep `frac_keep` of rows of the given classes (all classes if None), at least min_keep each."""
    out = []
    for c in np.unique(y[idx]):
        ic = idx[y[idx] == c]
        if classes is None or c in classes:
            k = max(min_keep, int(round(frac_keep * len(ic))))
            ic = rng.choice(ic, min(k, len(ic)), replace=False)
        out.extend(ic)
    return np.sort(np.array(out))


# ---------------------------------------------------------------- models
def make_model(name, seed, num_cols, cat_cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat_cols),
    ])
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000, C=1.0)
    elif name == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=seed)
    else:
        clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def scores(y_true, proba):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    pred = (proba >= 0.5).astype(int)
    ba = balanced_accuracy_score(y_true, pred)
    auc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) == 2 else float("nan")
    return float(ba), float(auc)


# ---------------------------------------------------------------- one condition
def run_condition(ds, seed, sev, tag):
    """sev: dict defect -> severity (0 = absent). Returns list of per-model records."""
    X, y = load(ds)
    cat_cols = [c for c in X.columns if X[c].dtype == object]
    num_cols = [c for c in X.columns if c not in cat_cols]
    split_rng = np.random.RandomState(seed)
    pool, test = stratified_split(y, TEST_SHARE, split_rng)
    rng = np.random.RandomState(10_000 + seed)  # same injection stream for every condition of a seed

    idx = pool.copy()
    # 1. reduced sample size (stratified)
    s = sev.get("sample_size", 0)
    if s > 0:
        idx = keep_per_class(idx, y, 1 - s, rng)
    # 2. class imbalance: remove share of the minority class (minority = rarer class in the pool)
    s = sev.get("imbalance", 0)
    if s > 0:
        counts = np.bincount(y[pool], minlength=2)
        minority = int(np.argmin(counts))
        idx = keep_per_class(idx, y, 1 - s, rng, classes={minority})
    Xd = X.iloc[idx].reset_index(drop=True).copy()
    yd = y[idx].copy()
    src = np.arange(len(idx))  # source row id, used to measure contamination
    # 3. label noise: flip a share of recorded labels
    s = sev.get("label_noise", 0)
    if s > 0:
        flip = rng.choice(len(yd), int(round(s * len(yd))), replace=False)
        yd[flip] = 1 - yd[flip]
    # 4. MCAR missing values in feature cells
    s = sev.get("missing", 0)
    if s > 0:
        mask = rng.rand(*Xd.shape) < s
        Xd = Xd.mask(mask)
    # 5. target leakage: a proxy column that equals the recorded label on a share q of rows and is a
    #    random draw elsewhere; at deployment the outcome is not yet known, so the proxy is random.
    q = sev.get("leakage", 0)
    Xte = X.iloc[test].reset_index(drop=True).copy()
    yte = y[test]
    ncols = list(num_cols)
    if q > 0:
        p1 = yd.mean()
        rnd = (rng.rand(len(yd)) < p1).astype(int)
        reveal = rng.rand(len(yd)) < q
        Xd["leak_proxy"] = np.where(reveal, yd, rnd).astype(float)
        Xte["leak_proxy"] = (rng.rand(len(yte)) < p1).astype(float)
        ncols = ncols + ["leak_proxy"]
    # 6. duplicates: append copies of a share of rows, so copies can land on both sides of the split
    s = sev.get("duplicates", 0)
    if s > 0:
        dup = rng.choice(len(yd), int(round(s * len(yd))), replace=False)
        Xd = pd.concat([Xd, Xd.iloc[dup]], ignore_index=True)
        yd = np.concatenate([yd, yd[dup]])
        src = np.concatenate([src, src[dup]])
    # the team's own split: train / internal validation ("reported" estimate)
    dev_rng = np.random.RandomState(20_000 + seed)
    perm = dev_rng.permutation(len(yd))
    nval = int(round(DEV_VAL_SHARE * len(yd)))
    va, tr = perm[:nval], perm[nval:]
    contamination = float(np.isin(src[va], src[tr]).mean())

    recs = []
    for m in MODELS:
        ytr = yd[tr]
        if len(np.unique(ytr)) < 2:
            p_va = np.full(len(va), float(ytr[0])); p_te = np.full(len(yte), float(ytr[0]))
        else:
            model = make_model(m, seed, ncols, cat_cols)
            model.fit(Xd.iloc[tr], ytr)
            p_va = model.predict_proba(Xd.iloc[va])[:, 1]
            p_te = model.predict_proba(Xte)[:, 1]
        rba, rauc = scores(yd[va], p_va) if len(np.unique(yd[va])) == 2 else (float("nan"), float("nan"))
        tba, tauc = scores(yte, p_te)
        recs.append(dict(dataset=ds, seed=seed, model=m, tag=tag, **{f"sev_{d}": float(sev.get(d, 0)) for d in DEFECTS},
                         n_train=int(len(tr)), contamination=contamination,
                         true_ba=tba, true_auc=tauc, reported_ba=rba, reported_auc=rauc))
    return recs


# ---------------------------------------------------------------- experiment plan
def plan():
    tasks = []
    for ds in DATASETS:
        for seed in SEEDS:
            tasks.append((ds, seed, {}, "clean"))
            for d in DEFECTS:
                for s in GRID[d]:
                    tasks.append((ds, seed, {d: s}, f"single:{d}"))
    combos = {}
    for di, ds in enumerate(DATASETS):
        crng = np.random.RandomState(777 + di)
        lst = []
        while len(lst) < N_COMBOS:
            present = crng.rand(len(DEFECTS)) < 0.5
            if present.sum() == 0:
                continue
            sev = {d: float(round(crng.uniform(0.02, max(GRID[d])), 3))
                   for d, p in zip(DEFECTS, present) if p}
            lst.append(sev)
        combos[ds] = lst
        for ci, sev in enumerate(lst):
            for seed in SEEDS:
                tasks.append((ds, seed, sev, f"combo:{ci}"))
    return tasks, combos


# ---------------------------------------------------------------- score
def curves_from(single, datasets):
    """Mean normalised loss per defect and grid level over the given datasets (all models & seeds)."""
    sub = single[single.dataset.isin(datasets)]
    out = {}
    for d in DEFECTS:
        s = sub[sub.defect == d].groupby("level").apply(pl)
        out[d] = (np.array([0.0] + list(s.index)), np.array([0.0] + list(s.values)))
    return out


def pl(g):
    """Pooled loss: share (%) of above-chance balanced accuracy lost, sum of drops / sum of headroom."""
    return float(100 * g["drop_pp"].sum() / g["hr"].sum())


def penalty(curves, sev_row, clip_neg=True):
    tot = 0.0
    for d in DEFECTS:
        lv, ls = curves[d]
        v = float(np.interp(sev_row[d], lv, ls))
        tot += max(v, 0.0) if clip_neg else v
    return tot


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def main():
    t0 = time.time()
    tasks, combos = plan()
    print(f"{len(tasks)} tasks x {len(MODELS)} models", flush=True)
    out = Parallel(n_jobs=N_JOBS, batch_size=1, verbose=0)(delayed(run_condition)(*t) for t in tasks)
    df = pd.DataFrame([r for rs in out for r in rs])
    runtime = time.time() - t0
    print(f"experiments done in {runtime:.1f} s", flush=True)

    # clean baselines per dataset/model/seed (paired)
    base = df[df.tag == "clean"].set_index(["dataset", "model", "seed"])
    df = df.join(base[["true_ba", "true_auc"]].rename(columns={"true_ba": "ba0", "true_auc": "auc0"}),
                 on=["dataset", "model", "seed"])
    df["drop_pp"] = 100 * (df.ba0 - df.true_ba)
    df["hr"] = 100 * (df.ba0 - 0.5)                                     # above-chance headroom, BA points
    df["auc_drop_pp"] = 100 * (df.auc0 - df.true_auc)
    df["inflation_pp"] = 100 * (df.reported_ba - df.true_ba)          # reported minus true

    single = df[df.tag.str.startswith("single")].copy()
    single["defect"] = single.tag.str.split(":").str[1]
    single["level"] = [r[f"sev_{r['defect']}"] for _, r in single.iterrows()]
    clean = df[df.tag == "clean"]

    R = {"meta": {
        "datasets": DATASETS, "models": MODELS, "seeds": SEEDS, "defects": DEFECTS,
        "defect_units": DEFECT_UNIT, "grid": GRID, "n_combos_per_dataset": N_COMBOS,
        "test_share": TEST_SHARE, "dev_validation_share": DEV_VAL_SHARE,
        "n_conditions": int(len(tasks)), "n_model_fits": int(len(df)),
        "runtime_s": round(runtime, 1), "n_jobs": N_JOBS,
        "metric": "balanced accuracy on a clean held-out test set (30% of each dataset, untouched)",
        "loss_definition": "100 * sum(BA_clean - BA_defect) / sum(BA_clean - 0.5) over the pooled runs (paired by dataset, model and seed): share of above-chance balanced accuracy lost, headroom-weighted",
        "sklearn_version": __import__("sklearn").__version__,
    }}
    # dataset descriptions
    R["datasets"] = {}
    for ds in DATASETS:
        X, y = load(ds)
        R["datasets"][ds] = {"rows": int(len(y)), "features": int(X.shape[1]),
                             "categorical_features": int(sum(X[c].dtype == object for c in X.columns)),
                             "positive_rate": r4(y.mean())}
    R["clean_baseline"] = {
        ds: {m: {"ba": r4(clean[(clean.dataset == ds) & (clean.model == m)].true_ba.mean()),
                 "auc": r4(clean[(clean.dataset == ds) & (clean.model == m)].true_auc.mean())}
             for m in MODELS} for ds in DATASETS}

    # ---------- RQ1 sensitivity curves
    sens = {}
    for d in DEFECTS:
        sd = single[single.defect == d]
        sens[d] = {"levels": GRID[d], "loss_mean": [], "loss_by_model": {m: [] for m in MODELS},
                   "loss_dataset_min": [], "loss_dataset_max": [], "drop_pp_mean": [], "auc_drop_pp_mean": [],
                   "inflation_pp_by_model": {m: [] for m in MODELS}, "reported_ba_by_model": {m: [] for m in MODELS},
                   "true_ba_by_model": {m: [] for m in MODELS}, "contamination_mean": [], "n_train_mean": []}
        for lv in GRID[d]:
            x = sd[sd.level == lv]
            sens[d]["loss_mean"].append(r4(pl(x)))
            per_ds = x.groupby("dataset").apply(pl)
            sens[d]["loss_dataset_min"].append(r4(per_ds.min()))
            sens[d]["loss_dataset_max"].append(r4(per_ds.max()))
            sens[d]["drop_pp_mean"].append(r4(x.drop_pp.mean()))
            sens[d]["auc_drop_pp_mean"].append(r4(x.auc_drop_pp.mean()))
            sens[d]["contamination_mean"].append(r4(x.contamination.mean()))
            sens[d]["n_train_mean"].append(r4(x.n_train.mean()))
            for m in MODELS:
                xm = x[x.model == m]
                sens[d]["loss_by_model"][m].append(r4(pl(xm)))
                sens[d]["inflation_pp_by_model"][m].append(r4(xm.inflation_pp.mean()))
                sens[d]["reported_ba_by_model"][m].append(r4(xm.reported_ba.mean()))
                sens[d]["true_ba_by_model"][m].append(r4(xm.true_ba.mean()))
    R["sensitivity"] = sens
    R["clean_inflation_pp_by_model"] = {m: r4(clean[clean.model == m].inflation_pp.mean()) for m in MODELS}
    R["clean_ba_mean_by_model"] = {m: r4(clean[clean.model == m].true_ba.mean()) for m in MODELS}

    # matched 20% comparison and ranking; per-dataset loss at 20%
    at_ref = single[np.isclose(single.level, REF_LEVEL)]
    ref = at_ref.groupby("defect").apply(pl).reindex(DEFECTS)
    R["loss_at_20pct"] = {d: r4(ref[d]) for d in DEFECTS}
    R["loss_at_20pct_by_model"] = {d: {m: r4(pl(at_ref[(at_ref.defect == d) & (at_ref.model == m)]))
                                       for m in MODELS} for d in DEFECTS}
    R["loss_at_max"] = {d: r4(pl(single[(single.defect == d) & np.isclose(single.level, max(GRID[d]))]))
                        for d in DEFECTS}
    pd_ref = at_ref.groupby(["dataset", "defect"]).apply(pl).unstack().reindex(index=DATASETS, columns=DEFECTS)
    R["loss_at_20pct_by_dataset"] = {ds: {d: r4(pd_ref.loc[ds, d]) for d in DEFECTS} for ds in DATASETS}
    # how often each defect is the most damaging at 20% (per dataset x model)
    top = at_ref.groupby(["dataset", "model", "defect"]).apply(pl).unstack().idxmax(axis=1)
    R["most_damaging_at_20pct_count"] = {d: int((top == d).sum()) for d in DEFECTS}
    R["most_damaging_at_20pct_total"] = int(len(top))
    # dataset-level significance at 20%: share of dataset x model cells with mean loss > 5
    R["share_cells_loss_gt5_at_20pct"] = {
        d: r4((at_ref[at_ref.defect == d].groupby(["dataset", "model"]).apply(pl) > 5).mean()) for d in DEFECTS}
    # linear weight per defect: least-squares slope through origin of loss vs severity, points per 0.1 severity
    lin = {}
    for d in DEFECTS:
        lv = np.array(GRID[d]); lm = np.array(sens[d]["loss_mean"], dtype=float)
        lin[d] = r4(0.1 * float((lv * lm).sum() / (lv * lv).sum()))
    R["linear_weight_per_0.1"] = lin

    # ---------- RQ2 evaluation integrity (reported vs true)
    R["integrity"] = {}
    for d in ["duplicates", "leakage", "label_noise"]:
        R["integrity"][d] = {m: {"reported_ba": sens[d]["reported_ba_by_model"][m],
                                 "true_ba": sens[d]["true_ba_by_model"][m],
                                 "inflation_pp": sens[d]["inflation_pp_by_model"][m]} for m in MODELS}

    # ---------- RQ3 score + LODO validation
    full_curves = curves_from(single, DATASETS)
    R["score_table"] = {d: {"levels": [r4(v) for v in full_curves[d][0]],
                            "penalty_points": [r4(max(v, 0.0)) for v in full_curves[d][1]]} for d in DEFECTS}
    combo = df[df.tag.str.startswith("combo")].copy()
    cagg = combo.groupby(["dataset", "tag"]).agg(
        dsum=("drop_pp", "sum"), hsum=("hr", "sum"), drop_pp=("drop_pp", "mean"), inflation_pp=("inflation_pp", "mean"),
        **{f"sev_{d}": (f"sev_{d}", "first") for d in DEFECTS}).reset_index()
    cmod = combo.groupby(["dataset", "tag", "model"]).agg(
        dsum=("drop_pp", "sum"), hsum=("hr", "sum"), drop_pp=("drop_pp", "mean"), **{f"sev_{d}": (f"sev_{d}", "first") for d in DEFECTS}).reset_index()

    cagg["loss"] = 100 * cagg.dsum / cagg.hsum
    cmod["loss"] = 100 * cmod.dsum / cmod.hsum

    def predict(frame):
        emp, emp_lin, eq, cnt = [], [], [], []
        for _, r in frame.iterrows():
            cal = [x for x in DATASETS if x != r.dataset]
            cv = cache[r.dataset]
            sv = {d: r[f"sev_{d}"] for d in DEFECTS}
            emp.append(min(100.0, penalty(cv, sv)))
            emp_lin.append(min(100.0, sum(max(lin_cache[r.dataset][d], 0) * sv[d] / 0.1 for d in DEFECTS)))
            eq.append(100.0 / len(DEFECTS) * sum(sv[d] / max(GRID[d]) for d in DEFECTS))
            cnt.append(sum(sv[d] > 0 for d in DEFECTS))
        frame = frame.copy()
        frame["pen_empirical"] = emp; frame["pen_linear"] = emp_lin
        frame["pen_equal"] = eq; frame["pen_count"] = cnt
        frame["score_empirical"] = 100 - frame.pen_empirical
        return frame

    cache, lin_cache = {}, {}
    for ds in DATASETS:
        cal = [x for x in DATASETS if x != ds]
        cache[ds] = curves_from(single, cal)
        lin_cache[ds] = {}
        for d in DEFECTS:
            lv, lm = cache[ds][d][0][1:], cache[ds][d][1][1:]
            lin_cache[ds][d] = 0.1 * float((lv * lm).sum() / (lv * lv).sum())
    cagg = predict(cagg)
    cmod = predict(cmod)

    def corr_block(frame, target="loss"):
        res = {}
        for k in ["pen_empirical", "pen_linear", "pen_equal", "pen_count"]:
            sp = stats.spearmanr(frame[k], frame[target]).correlation
            pr = stats.pearsonr(frame[k], frame[target])[0]
            res[k] = {"spearman": r4(sp), "pearson": r4(pr)}
            if k in ("pen_empirical", "pen_linear"):
                res[k]["mae_points"] = r4(np.mean(np.abs(frame[k] - frame[target].clip(upper=100))))
                res[k]["mean_actual_minus_predicted"] = r4(np.mean(frame[target].clip(upper=100) - frame[k]))
        return res

    val = {"combo_level": corr_block(cagg), "combo_x_model_level": corr_block(cmod)}
    val["combo_level_vs_drop_pp"] = {k: {"spearman": r4(stats.spearmanr(cagg[k], cagg.drop_pp).correlation),
                                         "pearson": r4(stats.pearsonr(cagg[k], cagg.drop_pp)[0])}
                                     for k in ["pen_empirical", "pen_equal", "pen_count"]}
    val["n_combos"] = int(len(cagg)); val["n_combo_model_pairs"] = int(len(cmod))
    val["by_model"] = {m: corr_block(cmod[cmod.model == m]) for m in MODELS}
    val["by_heldout_dataset_spearman"] = {
        ds: {k: r4(stats.spearmanr(cagg[cagg.dataset == ds][k], cagg[cagg.dataset == ds].loss).correlation)
             for k in ["pen_empirical", "pen_equal", "pen_count"]} for ds in DATASETS}
    # bootstrap the Spearman difference (empirical minus equal-weight), combo level
    brng = np.random.RandomState(42)
    diffs, diffs_c = [], []
    n = len(cagg)
    for _ in range(2000):
        b = cagg.iloc[brng.randint(0, n, n)]
        e = stats.spearmanr(b.pen_empirical, b.loss).correlation
        diffs.append(e - stats.spearmanr(b.pen_equal, b.loss).correlation)
        diffs_c.append(e - stats.spearmanr(b.pen_count, b.loss).correlation)
    val["spearman_diff_empirical_minus_equal_ci95"] = [r4(np.percentile(diffs, 2.5)), r4(np.percentile(diffs, 97.5))]
    val["spearman_diff_empirical_minus_count_ci95"] = [r4(np.percentile(diffs_c, 2.5)), r4(np.percentile(diffs_c, 97.5))]
    # score bands: actual loss by predicted score band
    bands = [(0, 50), (50, 70), (70, 85), (85, 101)]
    val["actual_loss_by_score_band"] = {
        f"{a}-{min(b, 100)}": {"n": int(((cagg.score_empirical >= a) & (cagg.score_empirical < b)).sum()),
                               "mean_actual_loss": r4(cagg[(cagg.score_empirical >= a) & (cagg.score_empirical < b)].loss.mean()),
                               "mean_drop_pp": r4(cagg[(cagg.score_empirical >= a) & (cagg.score_empirical < b)].drop_pp.mean())}
        for a, b in bands}
    val["actual_loss_range"] = [r4(cagg.loss.min()), r4(cagg.loss.max())]
    val["score_range"] = [r4(cagg.score_empirical.min()), r4(cagg.score_empirical.max())]
    # integrity sub-score: does an additive inflation model predict reported-minus-true in combos?
    infl_curves = {}
    for d in DEFECTS:
        s = single[single.defect == d].groupby("level").inflation_pp.mean()
        c0 = clean.inflation_pp.mean()
        infl_curves[d] = (np.array([0.0] + list(s.index)), np.array([0.0] + list(s.values - c0)))
    cagg["pred_inflation"] = [clean.inflation_pp.mean() + sum(float(np.interp(r[f"sev_{d}"], *infl_curves[d])) for d in DEFECTS)
                              for _, r in cagg.iterrows()]
    val["inflation_prediction"] = {"spearman": r4(stats.spearmanr(cagg.pred_inflation, cagg.inflation_pp).correlation),
                                   "pearson": r4(stats.pearsonr(cagg.pred_inflation, cagg.inflation_pp)[0]),
                                   "actual_inflation_range_pp": [r4(cagg.inflation_pp.min()), r4(cagg.inflation_pp.max())]}
    # leave-one-defect-out importance of the score (drop a defect from the empirical penalty)
    ablate = {}
    for d in DEFECTS:
        p = []
        for _, r in cagg.iterrows():
            sv = {k: (r[f"sev_{k}"] if k != d else 0.0) for k in DEFECTS}
            p.append(min(100.0, penalty(cache[r.dataset], sv)))
        ablate[d] = r4(stats.spearmanr(p, cagg.loss).correlation)
    val["spearman_without_defect"] = ablate
    R["validation"] = val
    R["combos"] = [{"dataset": r.dataset, "combo": r.tag, **{d: r4(r[f"sev_{d}"]) for d in DEFECTS},
                    "actual_loss": r4(r.loss), "actual_drop_pp": r4(r.drop_pp), "inflation_pp": r4(r.inflation_pp),
                    "predicted_penalty": r4(r.pen_empirical), "score": r4(r.score_empirical),
                    "equal_weight_penalty": r4(r.pen_equal), "defect_count": int(r.pen_count)}
                   for _, r in cagg.iterrows()]

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(R, f, indent=1)
    df.to_csv(os.path.join(HERE, "runs.csv.gz"), index=False, compression="gzip")
    make_charts(R, single, cagg)
    print(json.dumps({k: R[k] for k in ["loss_at_20pct", "loss_at_max", "linear_weight_per_0.1"]}, indent=1))
    print(json.dumps({k: R["validation"][k] for k in ["combo_level", "combo_x_model_level",
                                                       "spearman_diff_empirical_minus_equal_ci95",
                                                       "spearman_diff_empirical_minus_count_ci95",
                                                       "actual_loss_by_score_band", "inflation_prediction",
                                                       "spearman_without_defect"]}, indent=1))
    print(f"total {time.time() - t0:.1f} s")


# ---------------------------------------------------------------- charts
def make_charts(R, single, cagg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(CHARTS, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#555",
                         "axes.labelcolor": "#111", "xtick.color": "#111", "ytick.color": "#111",
                         "axes.spines.top": False, "axes.spines.right": False})
    MC = {"logreg": "#000000", "random_forest": "#555555", "hist_gbm": "#bbbbbb"}
    MS = {"logreg": "o", "random_forest": "s", "hist_gbm": "^"}
    LS = {"logreg": "-", "random_forest": "--", "hist_gbm": ":"}
    SRC = ("Source: our controlled defect-injection experiment, 10 public tabular datasets (scikit-learn, OpenML), "
           "3 models x 3 seeds. Code: github.com/stabilarity/hub/tree/master/research/ai-data-readiness-score")

    def finish(fig, name):
        fig.text(0.01, 0.01, SRC, fontsize=7, color="#555")
        fig.savefig(os.path.join(CHARTS, name), dpi=160, facecolor="white")
        plt.close(fig)

    # chart1: sensitivity curves
    sens = R["sensitivity"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.625), sharey=False)
    for ax, d in zip(axes.ravel(), DEFECTS):
        lv = [0] + GRID[d]
        lo = [0] + sens[d]["loss_dataset_min"]; hi = [0] + sens[d]["loss_dataset_max"]
        ax.fill_between(lv, lo, hi, color="#eeeeee", label="range across datasets")
        for m in MODELS:
            ax.plot(lv, [0] + sens[d]["loss_by_model"][m], LS[m], marker=MS[m], ms=4, color=MC[m], label=MODEL_LABEL[m])
        ax.axhline(0, color="#bbb", lw=0.8)
        ax.set_title(DEFECT_LABEL[d], fontsize=10)
        ax.set_xlabel(DEFECT_UNIT[d], fontsize=7.5)
        ax.set_ylabel("loss, % of above-chance BA", fontsize=7.5)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Chart 1. Performance loss on clean test data vs defect severity, by model", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    finish(fig, "chart1.png")

    # chart2: matched 20% comparison by model
    fig, ax = plt.subplots(figsize=(10, 5.625))
    x = np.arange(len(DEFECTS)); w = 0.26
    for i, m in enumerate(MODELS):
        vals = [R["loss_at_20pct_by_model"][d][m] for d in DEFECTS]
        b = ax.bar(x + (i - 1) * w, vals, w, color=MC[m], label=MODEL_LABEL[m], edgecolor="#111", lw=0.4)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + (0.3 if v >= 0 else -0.9), f"{v:.1f}", ha="center", fontsize=7, color="#111")
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([DEFECT_LABEL[d].replace(" / ", " /\n").replace(" (", "\n(") for d in DEFECTS], fontsize=8.5)
    ax.set_ylim(bottom=min(-2.5, ax.get_ylim()[0]))
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylabel("loss at severity 20%, % of above-chance balanced accuracy")
    ax.set_title("Chart 2. The same 20% dose of each defect: loss on clean test data (pooled over 10 datasets x 3 seeds)", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    finish(fig, "chart2.png")

    # chart3: reported vs true (duplicates, leakage)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.625))
    for ax, d in zip(axes, ["duplicates", "leakage"]):
        lv = [0] + GRID[d]
        for m in MODELS:
            c0 = R["clean_ba_mean_by_model"][m]
            ci = R["clean_inflation_pp_by_model"][m]
            rep = [100 * (c0 + ci / 100)] + [100 * v for v in sens[d]["reported_ba_by_model"][m]]
            tru = [100 * c0] + [100 * v for v in sens[d]["true_ba_by_model"][m]]
            ax.plot(lv, rep, "-", marker=MS[m], ms=4, color=MC[m], label=f"{MODEL_LABEL[m]}, reported")
            ax.plot(lv, tru, ":", marker=MS[m], ms=4, mfc="white", color=MC[m], label=f"{MODEL_LABEL[m]}, true (clean test)")
        ax.set_title(DEFECT_LABEL[d], fontsize=10)
        ax.set_xlabel(DEFECT_UNIT[d], fontsize=8)
        ax.set_ylabel("balanced accuracy, %")
    axes[1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Chart 3. What the team's own validation split reports vs deployment performance", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    finish(fig, "chart3.png")

    # chart4: per-dataset heatmap at 20%
    mat = np.array([[R["loss_at_20pct_by_dataset"][ds][d] for d in DEFECTS] for ds in DATASETS])
    fig, ax = plt.subplots(figsize=(10, 5.625))
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="Greys", vmin=min(0, np.nanmin(mat)), vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 * vmax else "#111")
    ax.set_xticks(range(len(DEFECTS))); ax.set_xticklabels([DEFECT_LABEL[d].replace(" / ", " /\n").replace(" (", "\n(") for d in DEFECTS], fontsize=8)
    ax.set_yticks(range(len(DATASETS))); ax.set_yticklabels(DATASETS, fontsize=8)
    cb = fig.colorbar(im, ax=ax); cb.set_label("loss at 20% severity, % of above-chance BA", fontsize=8)
    ax.set_title("Chart 4. Loss at 20% severity by dataset (pooled over 3 models x 3 seeds)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    finish(fig, "chart4.png")

    # chart5: validation scatter
    v = R["validation"]["combo_level"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 5.625))
    for ax, k, ttl in zip(axes, ["pen_empirical", "pen_equal", "pen_count"],
                          ["Empirical score penalty (100 - score)", "Equal-weight checklist penalty", "Checklist: number of defects present"]):
        ax.scatter(cagg[k], cagg.loss.clip(upper=100), s=12, color="#111", alpha=0.6, edgecolor="none")
        if k == "pen_empirical":
            ax.plot([0, 100], [0, 100], color="#bbb", lw=1, ls="--", label="predicted = actual")
            ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.set_xlabel(ttl, fontsize=8)
        ax.set_ylabel("actual loss, % of above-chance BA (capped at 100)", fontsize=8)
        ax.set_title(f"Spearman {v[k]['spearman']:.3f}", fontsize=10)
    fig.suptitle(f"Chart 5. Held-out validation: {len(cagg)} random defect combinations on unseen datasets (leave-one-dataset-out)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    finish(fig, "chart5.png")

    # chart6: score table (penalty points vs severity), the readiness lookup
    fig, ax = plt.subplots(figsize=(10, 5.625))
    styles = ["-", "--", ":", "-.", "-", "--"]
    cols = ["#000", "#000", "#000", "#555", "#555", "#bbb"]
    marks = ["o", "s", "^", "D", "v", "x"]
    for d, s_, c, mk in zip(DEFECTS, styles, cols, marks):
        st = R["score_table"][d]
        ax.plot([100 * l for l in st["levels"]], st["penalty_points"], s_, color=c, marker=mk, ms=5, label=DEFECT_LABEL[d])
    ax.set_xlabel("defect severity, % (units per defect as in Chart 1)")
    ax.set_ylabel("readiness penalty, points (= expected loss, % of above-chance BA)")
    ax.set_title("Chart 6. AI Data Readiness Score penalty curves, calibrated on all 10 datasets", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    finish(fig, "chart6.png")


if __name__ == "__main__":
    main()
