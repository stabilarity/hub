"""
Generate Stabilarity-style charts comparing FLAI vs baselines.
Loads tuned params from data/results.json.
"""

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

from flai_model import flai_predict_from_df
from baselines import arima_predict, lstm_predict

os.makedirs("charts", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#fff",
    "axes.facecolor": "#fff",
    "axes.edgecolor": "#ddd",
    "axes.grid": True,
    "grid.color": "#eee",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "font.family": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "#ddd",
    "lines.linewidth": 1.0,
    "figure.dpi": 120,
})

# Load tuned params
with open("data/results.json") as f:
    saved = json.load(f)
best_params = saved.get('best_params', {})

datasets = [
    ("Sound A: Viral Hit",     "data/sound_a_viral_hit.csv",      "a"),
    ("Sound B: Steady Grower", "data/sound_b_steady_grower.csv",  "b"),
    ("Sound C: Flash Trend",   "data/sound_c_flash_trend.csv",    "c"),
]

all_mape = {}

def calc_mape(a, p):
    mask = a > 10
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)

for title, path, key in datasets:
    df = pd.read_csv(path)
    bp = best_params.get(title, {})
    bW = bp.get('bW', 1.0)
    lr = bp.get('lr', 0.08)
    mom = bp.get('mom', 0.2)

    r_flai  = flai_predict_from_df(df, bW=bW, learning_rate=lr, momentum=mom)
    r_arima = arima_predict(df)
    r_lstm  = lstm_predict(df)

    days   = r_flai['days']
    actual = r_flai['actual']
    ts     = r_flai['train_size']

    m_flai  = calc_mape(r_flai['test_actual'],  r_flai['test_predicted'])
    m_arima = calc_mape(r_arima['test_actual'], r_arima['test_predicted'])
    m_lstm  = calc_mape(r_lstm['test_actual'],  r_lstm['test_predicted'])
    all_mape[title] = {'FLAI': m_flai, 'ARIMA': m_arima, 'LSTM': m_lstm}

    # Cap ARIMA for display if wildly off (Sound C case)
    arima_test_pred = r_arima['predicted'][ts:]
    cap = actual.max() * 3
    arima_test_pred_capped = np.clip(arima_test_pred, 0, cap)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(days, actual, color="#000", linewidth=1.5, label="Actual reposts", zorder=5)
    ax.axvline(x=days[ts], color="#ddd", linewidth=0.8, linestyle="--")
    ax.text(days[ts] + 0.3, actual.max() * 0.92, "test start", fontsize=7, color="#aaa")

    ax.plot(days[ts:], arima_test_pred_capped, color="#bbb", linewidth=1.0,
            linestyle="--", label=f"ARIMA  MAPE {m_arima:.1f}%")
    ax.plot(days[ts:], r_lstm['predicted'][ts:], color="#555", linewidth=1.0,
            linestyle=":", label=f"LSTM   MAPE {m_lstm:.1f}%")
    ax.plot(days[ts:], r_flai['predicted'][ts:], color="#2e7d32", linewidth=1.5,
            linestyle="-", label=f"FLAI   MAPE {m_flai:.1f}%")

    ax.set_title(f"{title} — Prediction Comparison")
    ax.set_xlabel("Day")
    ax.set_ylabel("Reposts")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = f"charts/sound_{key}_prediction.png"
    plt.savefig(out, bbox_inches="tight", facecolor="#fff")
    plt.close()
    print(f"Saved {out}")


# Summary bar chart
fig, ax = plt.subplots(figsize=(8, 4))
models = ["FLAI", "ARIMA", "LSTM"]
colors = {"FLAI": "#2e7d32", "ARIMA": "#bbb", "LSTM": "#555"}
short_names = {
    "Sound A: Viral Hit": "Sound A",
    "Sound B: Steady Grower": "Sound B",
    "Sound C: Flash Trend": "Sound C",
}

x = np.arange(len(all_mape))
width = 0.25

for i, model in enumerate(models):
    vals = [min(all_mape[k][model], 200) for k in all_mape]  # cap at 200 for display
    bars = ax.bar(x + i * width, vals, width, label=model, color=colors[model], linewidth=0)
    for bar, v, raw_v in zip(bars, vals, [all_mape[k][model] for k in all_mape]):
        label = f"{raw_v:.1f}" if raw_v < 1000 else f"{raw_v/1000:.0f}k"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                label, ha="center", va="bottom", fontsize=6.5, color="#555")

ax.set_xticks(x + width)
ax.set_xticklabels([short_names[k] for k in all_mape])
ax.set_ylabel("MAPE (%) — capped at 200 for display")
ax.set_title("MAPE Comparison — All Models and Sounds")
ax.legend(loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("charts/summary_mape.png", bbox_inches="tight", facecolor="#fff")
plt.close()
print("Saved charts/summary_mape.png")
print("All charts saved to charts/")
