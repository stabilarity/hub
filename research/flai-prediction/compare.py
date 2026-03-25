"""
Run all models on all 3 datasets and print comparison metrics.
MAPE, R2, MSE. FLAI is auto-tuned per dataset.
"""

import numpy as np
import pandas as pd
import json
from flai_model import flai_predict_from_df, tune_flai
from baselines import arima_predict, lstm_predict


def mape(actual, pred):
    mask = actual > 10
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def r2_score(actual, pred):
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def mse(actual, pred):
    return float(np.mean((actual - pred) ** 2))


datasets = [
    ("Sound A: Viral Hit",     "data/sound_a_viral_hit.csv"),
    ("Sound B: Steady Grower", "data/sound_b_steady_grower.csv"),
    ("Sound C: Flash Trend",   "data/sound_c_flash_trend.csv"),
]

results = {}
best_params_all = {}

print("\n" + "=" * 80)
print("FLAI vs ARIMA vs LSTM — Prediction Comparison")
print("=" * 80)

for name, path in datasets:
    df = pd.read_csv(path)
    print(f"\n  Tuning FLAI for {name}...", end=" ", flush=True)

    # Tune FLAI
    best = tune_flai(df)
    best_params_all[name] = best
    print(f"best bW={best['bW']}, lr={best['lr']}, mom={best['mom']} => MAPE={best['mape']:.2f}%")

    print(f"\n  Dataset: {name}")
    print(f"  {'Model':<10}  {'MAPE (%)':>10}  {'R2':>8}  {'MSE':>14}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*14}")

    row = {}

    # FLAI (tuned)
    r = flai_predict_from_df(df, bW=best['bW'], learning_rate=best['lr'],
                              momentum=best['mom'])
    a, p = r['test_actual'], r['test_predicted']
    row['FLAI'] = {'mape': mape(a, p), 'r2': r2_score(a, p), 'mse': mse(a, p)}
    print(f"  {'FLAI':<10}  {row['FLAI']['mape']:>10.2f}  {row['FLAI']['r2']:>8.4f}  {row['FLAI']['mse']:>14.0f}")

    # ARIMA
    r = arima_predict(df)
    a, p = r['test_actual'], r['test_predicted']
    row['ARIMA'] = {'mape': mape(a, p), 'r2': r2_score(a, p), 'mse': mse(a, p)}
    print(f"  {'ARIMA':<10}  {row['ARIMA']['mape']:>10.2f}  {row['ARIMA']['r2']:>8.4f}  {row['ARIMA']['mse']:>14.0f}")

    # LSTM
    r = lstm_predict(df)
    a, p = r['test_actual'], r['test_predicted']
    row['LSTM'] = {'mape': mape(a, p), 'r2': r2_score(a, p), 'mse': mse(a, p)}
    print(f"  {'LSTM':<10}  {row['LSTM']['mape']:>10.2f}  {row['LSTM']['r2']:>8.4f}  {row['LSTM']['mse']:>14.0f}")

    best_baseline_mape = min(row['ARIMA']['mape'], row['LSTM']['mape'])
    improvement = (best_baseline_mape - row['FLAI']['mape']) / best_baseline_mape * 100
    print(f"\n  FLAI improvement over best baseline: {improvement:.1f}% lower MAPE")

    results[name] = row

# Summary
print("\n" + "=" * 80)
print("SUMMARY TABLE — MAPE (%) by model and dataset")
print("=" * 80)
print(f"  {'Dataset':<30}  {'FLAI':>8}  {'ARIMA':>8}  {'LSTM':>8}  {'Improvement':>12}")
print("  " + "-" * 72)
for name, row in results.items():
    best_baseline = min(row['ARIMA']['mape'], row['LSTM']['mape'])
    impr = (best_baseline - row['FLAI']['mape']) / best_baseline * 100
    print(f"  {name:<30}  {row['FLAI']['mape']:>8.2f}  {row['ARIMA']['mape']:>8.2f}  {row['LSTM']['mape']:>8.2f}  {impr:>11.1f}%")

print("\nR2 SUMMARY")
print(f"  {'Dataset':<30}  {'FLAI':>8}  {'ARIMA':>8}  {'LSTM':>8}")
print("  " + "-" * 58)
for name, row in results.items():
    print(f"  {name:<30}  {row['FLAI']['r2']:>8.4f}  {row['ARIMA']['r2']:>8.4f}  {row['LSTM']['r2']:>8.4f}")

print("\nAll metrics on test set (last 30% of each series).")

# Save
with open("data/results.json", "w") as f:
    json.dump({'metrics': results, 'best_params': best_params_all}, f, indent=2)
print("Results saved to data/results.json")
