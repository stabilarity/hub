"""
Baseline models for comparison against FLAI:
  1. ARIMA(5,1,1) — classical statistical model via statsmodels
  2. LSTM-RNN — simple recurrent network implemented in numpy
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")


def interpolate_series(df: pd.DataFrame) -> np.ndarray:
    day_min, day_max = df['day'].min(), df['day'].max()
    full_days = pd.Series(range(int(day_min), int(day_max) + 1), name='day')
    df_full = full_days.to_frame().merge(df, on='day', how='left')
    return df_full['reposts'].interpolate(method='linear').fillna(0).values.astype(float)


# ─────────────────────────────────────────────────────────
# ARIMA baseline
# ─────────────────────────────────────────────────────────

def arima_predict(df: pd.DataFrame, train_frac: float = 0.7) -> dict:
    series = interpolate_series(df)
    n = len(series)
    train_size = int(n * train_frac)
    history = list(series[:train_size])
    preds_test = []

    for i in range(n - train_size):
        try:
            model = ARIMA(history, order=(5, 1, 1))
            fit = model.fit()
            yhat = float(fit.forecast(steps=1)[0])
        except Exception:
            yhat = float(history[-1])
        preds_test.append(max(0.0, yhat))
        history.append(series[train_size + i])

    predictions = np.concatenate([series[:train_size], np.array(preds_test)])
    return {
        'days': np.arange(1, n + 1),
        'actual': series,
        'predicted': predictions,
        'train_size': train_size,
        'test_actual': series[train_size:],
        'test_predicted': np.array(preds_test),
    }


# ─────────────────────────────────────────────────────────
# Simple RNN (numpy only)
# ─────────────────────────────────────────────────────────

def make_windows(series, window=5):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


def lstm_predict(df: pd.DataFrame, train_frac: float = 0.7, window: int = 5) -> dict:
    """Simple feed-forward regression with tanh hidden layer (numpy RNN approximation)."""
    np.random.seed(42)
    series = interpolate_series(df)
    n = len(series)
    train_size = int(n * train_frac)

    # Normalize
    mu = series[:train_size].mean()
    std = series[:train_size].std() + 1e-6
    s_norm = (series - mu) / std

    X_train, y_train = make_windows(s_norm[:train_size], window)

    # Network: input(5) -> hidden(16,tanh) -> output(1,linear)
    W1 = np.random.randn(window, 16) * 0.1
    b1 = np.zeros(16)
    W2 = np.random.randn(16, 1) * 0.1
    b2 = np.zeros(1)
    lr = 0.005

    for epoch in range(400):
        # Forward
        h = np.tanh(X_train @ W1 + b1)   # (N, 16)
        pred = h @ W2 + b2                 # (N, 1)
        pred = pred.ravel()
        err = pred - y_train               # (N,)
        # Backward
        dout = err / len(y_train)
        dW2 = h.T @ dout.reshape(-1, 1)
        db2 = dout.sum()
        dh = dout.reshape(-1, 1) * W2.T    # (N, 16)
        dtanh = (1 - h ** 2) * dh
        dW1 = X_train.T @ dtanh
        db1 = dtanh.sum(axis=0)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    # Walk-forward
    history = list(s_norm[:train_size])
    preds_test = []
    for i in range(n - train_size):
        x = np.array(history[-window:]).reshape(1, -1)
        h = np.tanh(x @ W1 + b1)
        p = float((h @ W2 + b2).ravel()[0])
        preds_test.append(p)
        history.append(s_norm[train_size + i])

    preds_test_denorm = np.maximum(0, np.array(preds_test) * std + mu)
    predictions = np.concatenate([series[:train_size], preds_test_denorm])

    return {
        'days': np.arange(1, n + 1),
        'actual': series,
        'predicted': predictions,
        'train_size': train_size,
        'test_actual': series[train_size:],
        'test_predicted': preds_test_denorm,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/sound_a_viral_hit.csv")
    r = arima_predict(df)
    mape = np.mean(np.abs((r['test_actual'] - r['test_predicted']) / (r['test_actual'] + 1e-6))) * 100
    print(f"ARIMA Sound A MAPE: {mape:.2f}%")
    r2 = lstm_predict(df)
    mape2 = np.mean(np.abs((r2['test_actual'] - r2['test_predicted']) / (r2['test_actual'] + 1e-6))) * 100
    print(f"LSTM  Sound A MAPE: {mape2:.2f}%")
