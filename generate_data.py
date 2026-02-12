"""Generate synthetic XAUUSD M1 data -- optimized for SMC patterns."""
import numpy as np
import pandas as pd
import os
import sys


def generate(bars: int = 200_000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    times = pd.date_range("2024-01-02", periods=bars, freq="1min")
    hours = np.array([t.hour for t in times])
    sess = np.where((hours >= 7) & (hours < 10), 3.0,
           np.where((hours >= 10) & (hours < 13), 2.0,
           np.where((hours >= 13) & (hours < 16), 3.5,
           np.where((hours >= 16) & (hours < 21), 2.5,
           np.where((hours >= 1) & (hours < 7), 1.0, 0.4)))))
    shocks = np.random.normal(0, 0.08, bars)
    vol = np.zeros(bars)
    vol[0] = 0.50
    for i in range(1, bars):
        vol[i] = 0.80 * vol[i - 1] + 0.20 * 0.50 + shocks[i]
    vol = np.clip(vol, 0.10, 3.0)
    spike_mask = np.random.random(bars) < 0.008
    vol[spike_mask] = np.random.uniform(1.5, 4.0, spike_mask.sum())
    ev = vol * sess
    drift_target = 2050.0
    noise = np.random.normal(0, 1, bars) * 0.80 * ev
    trend = (np.sin(np.arange(bars) / 1500 * np.pi) * 0.08
             + np.sin(np.arange(bars) / 500 * np.pi) * 0.03)
    closes = np.zeros(bars)
    closes[0] = drift_target
    for i in range(1, bars):
        d = (drift_target - closes[i - 1]) * 0.00002
        closes[i] = closes[i - 1] + d + trend[i] + noise[i]
    opens = np.roll(closes, 1)
    opens[0] = drift_target
    uw = np.abs(np.random.exponential(0.45 * ev, bars))
    lw = np.abs(np.random.exponential(0.45 * ev, bars))
    highs = np.maximum(opens, closes) + uw
    lows = np.minimum(opens, closes) - lw
    sweep_mask = np.random.random(bars) < 0.05
    sweep_up = sweep_mask & (np.random.random(bars) > 0.5)
    sweep_dn = sweep_mask & ~sweep_up
    spike_sz = np.random.uniform(1.0, 6.0, bars)
    highs[sweep_up] += spike_sz[sweep_up]
    closes[sweep_up] = np.minimum(closes[sweep_up], opens[sweep_up]) - 0.3
    lows[sweep_dn] -= spike_sz[sweep_dn]
    closes[sweep_dn] = np.maximum(closes[sweep_dn], opens[sweep_dn]) + 0.3
    sweep_indices = np.where(sweep_mask)[0]
    for idx in sweep_indices:
        if idx + 1 >= bars:
            continue
        disp_idx = idx + 1
        disp_size = np.random.uniform(1.5, 5.0)
        if sweep_up[idx]:
            opens[disp_idx] = closes[idx]
            closes[disp_idx] = opens[disp_idx] - disp_size
            highs[disp_idx] = opens[disp_idx] + np.random.uniform(0.1, 0.5)
            lows[disp_idx] = closes[disp_idx] - np.random.uniform(0.1, 0.5)
        elif sweep_dn[idx]:
            opens[disp_idx] = closes[idx]
            closes[disp_idx] = opens[disp_idx] + disp_size
            lows[disp_idx] = opens[disp_idx] - np.random.uniform(0.1, 0.5)
            highs[disp_idx] = closes[disp_idx] + np.random.uniform(0.1, 0.5)
    for idx in sweep_indices:
        if idx < 5:
            continue
        if sweep_up[idx]:
            peak_offset = np.random.randint(3, min(6, idx))
            peak_idx = idx - peak_offset
            local_peak = highs[max(0, peak_idx - 2):peak_idx + 3].max()
            highs[peak_idx] = max(highs[peak_idx], local_peak + 0.2)
        elif sweep_dn[idx]:
            trough_offset = np.random.randint(3, min(6, idx))
            trough_idx = idx - trough_offset
            local_trough = lows[max(0, trough_idx - 2):trough_idx + 3].min()
            lows[trough_idx] = min(lows[trough_idx], local_trough - 0.2)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    tick_vol = np.maximum(1, np.random.poisson(50 * ev + 15).astype(int))
    df = pd.DataFrame({
        "time": times,
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "tick_volume": tick_vol,
    })
    return df


if __name__ == "__main__":
    bars = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    print(f"Generating {bars:,} XAUUSD M1 bars (SMC-optimized)...")
    df = generate(bars)
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "XAUUSD_M1.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    print(f"Range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    print(f"Price: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"Rows: {len(df):,}")
    m1_range = df["high"] - df["low"]
    print(f"Avg bar range: {m1_range.mean():.3f}")
