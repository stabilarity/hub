# AI Data Readiness Score: controlled defect injection

**Question.** Which of six common training-data defects actually break tabular classifiers, how much do they
distort the performance a team reports to itself, and can an empirically weighted readiness score predict the
performance loss of an unseen dataset better than an equal-weight checklist?

## Design

- **Data:** 10 public binary classification datasets, at most 4,000 rows each (deterministic subsample, seed 0),
  stored in `data/*.csv.gz` so the analysis runs offline.

  | File | Source | Positive class |
  |---|---|---|
  | breast-cancer | scikit-learn `load_breast_cancer` (UCI Wisconsin Diagnostic) | malignant |
  | california-housing | scikit-learn `fetch_california_housing` | median house value above the median |
  | adult | OpenML [1590](https://www.openml.org/d/1590) | >50K |
  | credit-g | OpenML [31](https://www.openml.org/d/31) | bad |
  | bank-marketing | OpenML [1461](https://www.openml.org/d/1461) | 2 (subscribed) |
  | spambase | OpenML [44](https://www.openml.org/d/44) | 1 (spam) |
  | diabetes | OpenML [37](https://www.openml.org/d/37) | tested_positive |
  | phoneme | OpenML [1489](https://www.openml.org/d/1489) | 2 |
  | magic-telescope | OpenML [1120](https://www.openml.org/d/1120) | h |
  | blood-transfusion | OpenML [1464](https://www.openml.org/d/1464) | 2 (donated) |

  `fetch_data.py` rebuilds `data/` from these sources.
- **Clean test set:** 30% of each dataset (stratified, per seed) is never touched. All "true" numbers are measured on it.
- **Defects** (injected into the remaining 70% only, each at 5 graded severities):
  1. `missing` – MCAR: a share of feature cells set to missing (median / most-frequent imputation in the pipeline).
  2. `label_noise` – a share of training labels flipped at random.
  3. `duplicates` – copies of a share of rows appended before the team's own train/validation split, so copies
     can land on both sides (train-test contamination).
  4. `imbalance` – a share of minority-class rows removed.
  5. `leakage` – a proxy column equal to the recorded label on a share q of rows (a random draw elsewhere);
     at deployment the outcome is not yet known, so the proxy is a random draw on the clean test set.
  6. `sample_size` – a share of training rows removed (stratified).
- **Models:** logistic regression, random forest (100 trees), HistGradientBoosting (100 iterations), each in a
  standard scikit-learn pipeline (imputation, scaling, one-hot encoding). Seeds 0, 1, 2.
- **Two estimates per run:** *reported* = balanced accuracy on the team's own 20% validation split of the
  (defective) data; *true* = balanced accuracy on the clean test set.
- **Loss:** the share of above-chance balanced accuracy lost, `100 * sum(BA_clean - BA_defect) / sum(BA_clean - 0.5)`,
  pooled over the runs being summarised and paired by dataset, model and seed. Pooling (a ratio of sums) keeps runs
  whose clean model is barely above chance (logistic regression on blood-transfusion, clean BA 0.53) from dominating.
- **Score:** `Readiness = 100 - sum_d penalty_d(severity_d)`, where `penalty_d` is the mean single-defect loss curve
  (piecewise linear between grid levels, negative values clipped to 0).
- **Validation:** 30 random defect combinations per dataset (each defect present with probability 0.5, severity
  uniform). Penalty curves are re-fitted leaving the target dataset out (leave-one-dataset-out). Compared with an
  equal-weight graded checklist and a count of defects present, using Spearman and Pearson correlation and a
  2,000-sample bootstrap.

## Run

```bash
python3 fetch_data.py      # only to rebuild data/ (needs network)
python3 analysis.py        # full run, about 3 minutes on 6 CPU cores (179.5 s measured)
python3 analysis.py --tiny # smoke test
```

## Outputs

- `results.json` – every number used in the article (sensitivity curves, matched 20% comparison, reported vs
  true estimates, score lookup table, validation statistics, all 300 combinations).
- `runs.csv.gz` – one row per model fit.
- `charts/chart1.png` – loss vs severity for each defect and model.
- `charts/chart2.png` – the same 20% dose of each defect, by model.
- `charts/chart3.png` – reported vs true balanced accuracy under duplicates and leakage.
- `charts/chart4.png` – loss at 20% severity by dataset.
- `charts/chart5.png` – held-out validation of the score against two checklist baselines.
- `charts/chart6.png` – the score's penalty curves (lookup table).

Software: Python 3.12, scikit-learn 1.8.0, numpy, pandas, scipy, matplotlib, joblib.
