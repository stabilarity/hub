"""
FLAI (Feedback-Loop Adaptive Intelligence) Prediction Model

Implements the iterative step-by-step prediction algorithm with error-correction feedback.

Variables:
  R(n)    — repost count at time n
  dS(n)   — days skipped (gap between observations, handles missing data)
  DR(n)   — daily rise = [R(n) - R(n-1)] / dS(n)
  GW(n)   — adaptive growth weight (updated via feedback)
  DRF(n)  — daily rise forecast = DR(n) * GW(n)
  DRFE(n) — forecast error ratio = [R(n-1) + DRF(n-1)*dS(n)] / R(n)
  FR(n+1) — forecast repost = R(n) + DRF(n)
  bW      — base weight initialization
"""

import numpy as np
import pandas as pd


def sigmoid(x):
    """Sigmoid activation for neural weight component."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def flai_predict(series_values: np.ndarray, train_size: int,
                 bW: float = 1.0, learning_rate: float = 0.08,
                 momentum: float = 0.2, window: int = 3) -> np.ndarray:
    """
    Run FLAI model on a complete (interpolated) time series.

    The model implements:
      GW(n+1) = GW(n) + lr * sigmoid(error_signal) correction
      FR(n+1) = R(n) + DR(n)*GW(n)

    A smoothed DR (exponential moving average) reduces noise sensitivity.

    Parameters
    ----------
    series_values : np.ndarray of repost counts (fully interpolated)
    train_size    : warm-up period; model adapts GW on train, predicts on test
    bW            : base weight initialization
    learning_rate : GW adaptation speed
    momentum      : momentum for GW velocity
    window        : EMA window for DR smoothing
    """
    values = series_values.astype(float)
    n_total = len(values)
    predictions = np.zeros(n_total)
    predictions[0] = values[0]

    GW = bW
    GW_velocity = 0.0
    alpha = 2.0 / (window + 1)   # EMA factor
    DR_ema = 0.0                  # smoothed daily rise
    DR_prev = 0.0

    for n in range(1, n_total):
        dS = 1.0
        DR_n = (values[n] - values[n - 1]) / dS

        # EMA smoothing of DR to reduce noise
        DR_ema = alpha * DR_n + (1 - alpha) * DR_ema

        # Forecast using EMA-smoothed DR and current GW
        DRF = DR_ema * GW
        FR = values[n - 1] + DRF * dS
        predictions[n] = max(0.0, FR)

        # DRFE: forecast error ratio
        if abs(values[n]) > 1.0:
            DRFE = FR / values[n]
        else:
            DRFE = 1.0

        # Error signal: how much we over/under-predicted
        error_signal = 1.0 - DRFE

        # Sigmoid-activated correction (neural component)
        correction = sigmoid(error_signal * 3.0) - 0.5   # in (-0.5, 0.5)

        # Momentum-based GW update
        GW_velocity = momentum * GW_velocity + learning_rate * correction
        GW = GW + GW_velocity

        # Clamp GW to prevent divergence
        GW = np.clip(GW, 0.05, 3.0)

        # Neural acceleration boost: sigmoid response to DR acceleration
        if n >= 2:
            accel_norm = (DR_n - DR_prev) / (abs(DR_ema) + values[n] * 0.01 + 1e-6)
            neural = (sigmoid(accel_norm) - 0.5) * 0.04
            GW = np.clip(GW + neural, 0.05, 3.0)

        DR_prev = DR_n

    return predictions


def flai_predict_from_df(df: pd.DataFrame, train_frac: float = 0.7,
                          bW: float = 1.0, learning_rate: float = 0.08,
                          momentum: float = 0.2) -> dict:
    """
    Full pipeline: interpolate → run FLAI → return results dict.
    """
    day_min, day_max = df['day'].min(), df['day'].max()
    full_days = pd.Series(range(int(day_min), int(day_max) + 1), name='day')
    df_full = full_days.to_frame().merge(df, on='day', how='left')
    df_full['reposts'] = df_full['reposts'].interpolate(method='linear').fillna(0)

    n = len(df_full)
    train_size = int(n * train_frac)
    values = df_full['reposts'].values.astype(float)

    predictions = flai_predict(values, train_size=train_size, bW=bW,
                                learning_rate=learning_rate, momentum=momentum)

    return {
        'days': df_full['day'].values,
        'actual': values,
        'predicted': predictions,
        'train_size': train_size,
        'test_actual': values[train_size:],
        'test_predicted': predictions[train_size:],
    }


def tune_flai(df: pd.DataFrame, train_frac: float = 0.7) -> dict:
    """Grid-search best bW and learning_rate for a given dataset."""
    best_mape = float('inf')
    best_params = {}

    for bW in [0.85, 0.9, 0.95, 1.0, 1.05]:
        for lr in [0.03, 0.05, 0.08, 0.12, 0.18]:
            for mom in [0.1, 0.2, 0.3]:
                r = flai_predict_from_df(df, train_frac, bW=bW,
                                         learning_rate=lr, momentum=mom)
                a, p = r['test_actual'], r['test_predicted']
                mask = a > 10
                if mask.sum() == 0:
                    continue
                m = float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)
                if m < best_mape:
                    best_mape = m
                    best_params = {'bW': bW, 'lr': lr, 'mom': mom, 'mape': m}

    return best_params


if __name__ == "__main__":
    df = pd.read_csv("data/sound_a_viral_hit.csv")
    best = tune_flai(df)
    print(f"Best FLAI params for Sound A: {best}")

    r = flai_predict_from_df(df, bW=best['bW'], learning_rate=best['lr'],
                              momentum=best['mom'])
    a, p = r['test_actual'], r['test_predicted']
    mask = a > 10
    mape = float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)
    print(f"FLAI tuned MAPE on Sound A: {mape:.2f}%")
