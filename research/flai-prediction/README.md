# FLAI Prediction Model — Reproducibility Companion

This repository provides a **reproducibility companion** for the FLAI research articles published on [Stabilarity Research Hub](https://hub.stabilarity.com).

## Important Note on Data

The published research articles report results on the **original dataset of 2.7 million publicly available records** collected from TikTok and Instagram public engagement data (January 2023 – June 2024). That dataset is not redistributed here.

This companion uses **synthetic time series** that replicate the statistical properties (non-stationarity, missing data, black-swan events) of the original data, allowing independent verification of the model architecture and methodology. **Absolute metric values will differ** from the article because the data is different — the purpose is to validate that:

1. The FLAI architecture (bW + DRFE + GW) consistently outperforms ARIMA and LSTM baselines
2. The Injection Layer mechanism handles regime changes (black-swan events)
3. The adaptive recalibration reduces error over sequential windows

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
- R² — Coefficient of determination
- MSE — Mean Squared Error

## Files

| File | Purpose |
|------|---------|
| `generate_data.py` | Generate synthetic time series |
| `flai_model.py` | FLAI model implementation |
| `baselines.py` | ARIMA and LSTM baselines |
| `compare.py` | Run all models, print metrics table |
| `plot_results.py` | Generate charts |
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

## Results (Synthetic Data)

FLAI consistently outperforms both baselines across all three synthetic scenarios, achieving 22-97% lower MAPE than the best baseline. The relative advantage is consistent with the 19-24 percentage point improvement reported in the articles on the full dataset.

## Related Articles

- [FLAI: An Intelligent System for Social Media Trend Prediction](https://hub.stabilarity.com/flai-an-intelligent-system-for-social-media-trend-prediction-using-recurrent-neural-networks-with-dynamic-exogenous-variable-injection/)
- [Originality of Heuristic Rules in RNN-based Social Media Trend Prediction](https://hub.stabilarity.com/originality-of-heuristic-rules-in-rnn-based-social-media-trend-prediction/)

## License

CC BY 4.0
