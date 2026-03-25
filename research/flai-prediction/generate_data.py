"""
Generate synthetic TikTok sound repost time series for FLAI experiment.
Three scenarios: Viral Hit, Steady Grower, Flash Trend.
Includes 5-10% random missing days to test interpolation.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)


def add_noise(arr, scale=0.08):
    """Add multiplicative noise."""
    noise = np.random.normal(1.0, scale, size=len(arr))
    return np.maximum(0, arr * noise).astype(int)


def add_missing_days(df, frac=0.07):
    """Remove random rows to simulate missing days."""
    drop_idx = np.random.choice(df.index[1:-1], size=int(len(df) * frac), replace=False)
    return df.drop(drop_idx).reset_index(drop=True)


# ─────────────────────────────────────────────────────────
# Sound A: Viral Hit
# ─────────────────────────────────────────────────────────
days = np.arange(1, 91)

# Phase 1: slow start (days 1-9)
phase1 = 200 + np.linspace(0, 800, 9)
# Phase 2: exponential growth (days 10-30)
phase2 = 1000 * np.exp(np.linspace(0, 3.5, 21))
# Phase 3: plateau (days 31-55)
phase3 = np.full(25, 33000) + np.random.normal(0, 800, 25)
# Phase 4: black swan second spike (days 56-65)
phase4 = 33000 + np.concatenate([
    np.linspace(0, 12000, 5),
    np.linspace(12000, 0, 5)
])
# Phase 5: gradual decline (days 66-90)
phase5 = 33000 * np.exp(-np.linspace(0, 1.2, 25))

reposts_a_raw = np.concatenate([phase1, phase2, phase3, phase4, phase5])
reposts_a_raw = np.clip(reposts_a_raw, 0, 50000)
reposts_a = add_noise(reposts_a_raw, scale=0.06)

df_a = pd.DataFrame({"day": days, "reposts": reposts_a})
df_a = add_missing_days(df_a, frac=0.07)
df_a.to_csv("data/sound_a_viral_hit.csv", index=False)
print(f"Sound A: {len(df_a)} rows, max={df_a.reposts.max()}, missing={90-len(df_a)} days")


# ─────────────────────────────────────────────────────────
# Sound B: Steady Grower
# ─────────────────────────────────────────────────────────
base_b = np.linspace(500, 28000, 90)
# Add weekly seasonal bumps
seasonal = 1500 * np.sin(2 * np.pi * days / 7) + 500 * np.sin(2 * np.pi * days / 30)
reposts_b_raw = base_b + seasonal
reposts_b_raw = np.clip(reposts_b_raw, 0, 35000)
reposts_b = add_noise(reposts_b_raw, scale=0.09)

df_b = pd.DataFrame({"day": days, "reposts": reposts_b})
df_b = add_missing_days(df_b, frac=0.08)
df_b.to_csv("data/sound_b_steady_grower.csv", index=False)
print(f"Sound B: {len(df_b)} rows, max={df_b.reposts.max()}, missing={90-len(df_b)} days")


# ─────────────────────────────────────────────────────────
# Sound C: Flash Trend
# ─────────────────────────────────────────────────────────
# Immediate spike days 1-5, rapid decay, long tail
spike = 50000 * np.exp(-np.linspace(0, 2.5, 5))
decay = 50000 * np.exp(-np.linspace(2.5, 6.0, 55))
tail = np.linspace(decay[-1], 800, 30)
reposts_c_raw = np.concatenate([spike, decay, tail])
reposts_c_raw = np.clip(reposts_c_raw, 0, 50000)
reposts_c = add_noise(reposts_c_raw, scale=0.07)

df_c = pd.DataFrame({"day": days, "reposts": reposts_c})
df_c = add_missing_days(df_c, frac=0.06)
df_c.to_csv("data/sound_c_flash_trend.csv", index=False)
print(f"Sound C: {len(df_c)} rows, max={df_c.reposts.max()}, missing={90-len(df_c)} days")

print("\nData generation complete. Files saved to data/")
