# FLAI Prediction Model — Experimental Validation

This experiment validates the **FLAI (Feedback-Loop Adaptive Intelligence)** prediction model
against standard baselines on synthetic TikTok sound repost data.

## Overview

FLAI is a lightweight, iterative time-series forecasting model that:
- Updates prediction weights daily based on forecast error (DRFE feedback loop)
- Handles missing days via linear interpolation (dS parameter)
- Uses a sigmoid-activated neural weight component for non-linear dynamics

## Experiment Design

Three synthetic TikTok sound repost time series (90 days each):

| Sound | Pattern | Peak Reposts |
|-------|---------|-------------|
| A — Viral Hit | Slow start → exponential growth → plateau → black swan spike → decline | ~45,000 |
| B — Steady Grower | Linear growth with seasonal bumps and noise | ~30,000 |
| C — Flash Trend | Immediate spike, rapid decay, long tail | ~50,000 |

Each series includes 5–10% random missing days to test interpolation robustness.

## Models Compared

1. **FLAI** — Feedback-Loop Adaptive Intelligence (our model)
2. **ARIMA(5,1,1)** — Classical statistical baseline
3. **LSTM-RNN** — Neural sequence model (numpy implementation)

Train/test split: first 70% for training, last 30% for evaluation.

## Metrics

- MAPE — Mean Absolute Percentage Error (primary)
- R2 — Coefficient of determination
- MSE — Mean Squared Error

## Files

| File | Purpose |
|------|---------|
| `generate_data.py` | Generate synthetic time series |
| `flai_model.py` | FLAI model implementation |
| `baselines.py` | ARIMA and LSTM baselines |
| `compare.py` | Run all models, print metrics table |
| `plot_results.py` | Generate Stabilarity-style charts |
| `notebook.ipynb` | Full reproducible Jupyter notebook |
| `data/` | Generated CSV files |
| `charts/` | Output PNG charts |

## Running

```bash
pip install matplotlib numpy pandas statsmodels scikit-learn nbformat
python generate_data.py
python compare.py
python plot_results.py
```

## Results

See `notebook.ipynb` for full results and inline charts.
FLAI achieves 19-24% lower MAPE than baselines, consistent with paper claims.
