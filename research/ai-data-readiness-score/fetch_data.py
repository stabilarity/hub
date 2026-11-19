"""Fetch the 10 public tabular datasets once and store deterministic subsamples under data/.

analysis.py reads only data/*.csv.gz, so it runs offline. Run this script only to rebuild data/.
Sources: scikit-learn bundled/fetchable datasets and OpenML (https://www.openml.org), IDs below.
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing, fetch_openml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
MAX_ROWS = 4000
RNG = np.random.RandomState(0)

# name -> (OpenML data id, positive class label or None for "minority class")
OPENML = {
    "adult": (1590, ">50K"),
    "credit-g": (31, "bad"),
    "bank-marketing": (1461, "2"),
    "spambase": (44, "1"),
    "diabetes": (37, "tested_positive"),
    "phoneme": (1489, "2"),
    "magic-telescope": (1120, "h"),
    "blood-transfusion": (1464, "2"),
}


def save(name, X, y, source):
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y).astype(int), name="target")
    if len(X) > MAX_ROWS:
        idx = RNG.choice(len(X), MAX_ROWS, replace=False)
        idx.sort()
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    df = X.copy()
    df["target"] = y.values
    df.to_csv(os.path.join(OUT, f"{name}.csv.gz"), index=False, compression="gzip")
    print(name, df.shape, "pos rate %.3f" % y.mean(), source)


def main():
    os.makedirs(OUT, exist_ok=True)
    bc = load_breast_cancer(as_frame=True)
    # positive = malignant (target 0 in sklearn)
    save("breast-cancer", bc.data, (bc.target == 0).astype(int), "sklearn load_breast_cancer")
    ch = fetch_california_housing(as_frame=True)
    save("california-housing", ch.data, (ch.target > ch.target.median()).astype(int),
         "sklearn fetch_california_housing, target = value above median")
    for name, (did, pos) in OPENML.items():
        b = fetch_openml(data_id=did, as_frame=True, parser="auto")
        X = b.data.copy()
        for c in X.columns:
            if str(X[c].dtype) == "category":
                X[c] = X[c].astype(object)
        y = (b.target.astype(str) == pos).astype(int)
        save(name, X, y, f"OpenML data_id={did}, positive class {pos}")


if __name__ == "__main__":
    main()
