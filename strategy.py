# ============================================================
#  VALIDUS — Strategy (Signal Engine – Module B)
#  Smart Money Concept: Liquidity Sweep → Displacement → FVG
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from utils import log


# ── Manual indicators (no pandas_ta / numba needed) ─────────
def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range — pure pandas."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()


def _bbands(close: pd.Series, length: int = 20, std: float = 2.0):
    """Bollinger Bands — returns (upper, mid, lower) as Series."""
    mid = close.rolling(window=length).mean()
    sd = close.rolling(window=length).std()
    upper = mid + std * sd
    lower = mid - std * sd
    return upper, mid, lower


# ── Data-class for a trade signal ───────────────────────────
class Signal:
    __slots__ = ("direction", "entry", "sl", "tp", "fvg_zone", "reason")

    def __init__(
        self,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        fvg_zone: tuple[float, float],
        reason: str = "",
    ):
        self.direction = direction   # "BUY" | "SELL"
        self.entry = entry
        self.sl = sl
        self.tp = tp
        self.fvg_zone = fvg_zone
        self.reason = reason

    def __repr__(self) -> str:
        return (
            f"Signal({self.direction} @ {self.entry:.5f}  "
            f"SL={self.sl:.5f}  TP={self.tp:.5f}  [{self.reason}])"
        )


# ── Strategy class ──────────────────────────────────────────
class Strategy:
    """
    Encapsulates the full SMC entry logic.
    Call `evaluate(df_m1, df_m5)` each time a new M1 bar closes.
    Returns a `Signal` or None.

    After evaluate(), read `self.last_eval` for structured decision data:
      - atr, bb_exp, price: indicator values at evaluation time
      - fail: which condition failed (None if signal generated)
      - detail: human-readable explanation
    """

    def __init__(self):
        self.last_eval: dict = {}

    # ── Condition 1: Volatility check ───────────────────────
    @staticmethod
    def _volatility_ok(df: pd.DataFrame) -> bool:
        """Market must be volatile enough to trade."""
        atr = _atr(df["high"], df["low"], df["close"], length=config.ATR_PERIOD)
        if atr.empty or np.isnan(atr.iloc[-1]):
            log.info("[VOL] ATR data not ready -- skipping.")
            return False
        current_atr = atr.iloc[-1]
        price = df["close"].iloc[-1]
        relative_atr = current_atr / price if price > 0 else 0.0
        if relative_atr < config.ATR_THRESHOLD:
            log.info("[VOL] ATR/price %.6f < threshold %.6f -- market too quiet.",
                     relative_atr, config.ATR_THRESHOLD)
            return False

        # Bollinger Band expansion check
        bbu, _, bbl = _bbands(df["close"], length=config.BB_PERIOD, std=config.BB_STD)
        if bbu.empty or np.isnan(bbu.iloc[-1]):
            log.info("[VOL] BB data not ready -- skipping.")
            return False
        width = bbu - bbl
        avg_width = width.rolling(config.BB_PERIOD).mean()
        if avg_width.iloc[-1] == 0:
            log.info("[VOL] BB avg width is 0 -- skipping.")
            return False
        expansion = width.iloc[-1] / avg_width.iloc[-1]
        if expansion < config.BB_EXPANSION_FACTOR:
            log.info("[VOL] BB expansion %.2f < factor %.2f -- not expanding.",
                     expansion, config.BB_EXPANSION_FACTOR)
            return False
        log.info("[VOL] PASS: ATR=%.4f ATR/price=%.6f (>%.6f) BB_exp=%.2f (>%.2f)",
                 current_atr, current_atr / price if price > 0 else 0.0,
                 config.ATR_THRESHOLD, expansion, config.BB_EXPANSION_FACTOR)
        return True

    # ── Swing level helpers (proper structure detection) ─────
    @staticmethod
    def _find_swing_high(highs: pd.Series, confirm: int = 3) -> float | None:
        """Find the most recent confirmed swing high."""
        n = len(highs)
        if n < confirm * 2 + 1:
            return float(highs.max()) if not highs.empty else None
        for i in range(n - confirm - 1, confirm - 1, -1):
            left_max = highs.iloc[max(0, i - confirm):i].max()
            right_max = highs.iloc[i + 1:min(n, i + confirm + 1)].max()
            if highs.iloc[i] > left_max and highs.iloc[i] > right_max:
                return float(highs.iloc[i])
        return float(highs.max()) if not highs.empty else None

    @staticmethod
    def _find_swing_low(lows: pd.Series, confirm: int = 3) -> float | None:
        """Find the most recent confirmed swing low."""
        n = len(lows)
        if n < confirm * 2 + 1:
            return float(lows.min()) if not lows.empty else None
        for i in range(n - confirm - 1, confirm - 1, -1):
            left_min = lows.iloc[max(0, i - confirm):i].min()
            right_min = lows.iloc[i + 1:min(n, i + confirm + 1)].min()
            if lows.iloc[i] < left_min and lows.iloc[i] < right_min:
                return float(lows.iloc[i])
        return float(lows.min()) if not lows.empty else None

    # ── Condition 2: Liquidity sweep detection ──────────────
    @staticmethod
    def _find_liquidity_sweep(df: pd.DataFrame) -> dict | None:
        """
        Detect a fakeout where price wicked beyond a recent swing
        high/low then closed back inside — indicating a stop hunt.
        Sweep candle = bar[-2] so bar[-1] can be the displacement.
        """
        lb = config.SWING_LOOKBACK
        if len(df) < lb + 4:
            return None

        # bar[-2] = sweep candle, bar[-1] = displacement candle
        sweep_candle = df.iloc[-2]

        # Lookback for swing levels: exclude sweep & displacement bars
        lookback_df = df.iloc[-(lb + 4):-2]
        confirm = config.SWING_CONFIRM_BARS
        swing_high = Strategy._find_swing_high(lookback_df["high"], confirm)
        swing_low = Strategy._find_swing_low(lookback_df["low"], confirm)

        if swing_high is None or swing_low is None:
            log.info("[SWEEP] No confirmed swing levels (confirm=%d bars).", confirm)
            return None

        body = abs(sweep_candle["close"] - sweep_candle["open"])
        body = max(body, 1e-10)

        # Bearish sweep (wick above swing high, close back below)
        if sweep_candle["high"] > swing_high and sweep_candle["close"] < swing_high:
            upper_wick = sweep_candle["high"] - max(sweep_candle["close"], sweep_candle["open"])
            ratio = upper_wick / body
            if ratio >= config.SWEEP_WICK_RATIO:
                log.info(
                    "[SWEEP] BEARISH: wick=%.5f > swing_high=%.5f, ratio=%.2f (>=%.2f)",
                    sweep_candle["high"], swing_high, ratio, config.SWEEP_WICK_RATIO,
                )
                return {
                    "type": "BEARISH_SWEEP",
                    "sweep_level": swing_high,
                    "candle_high": sweep_candle["high"],
                }

        # Bullish sweep (wick below swing low, close back above)
        if sweep_candle["low"] < swing_low and sweep_candle["close"] > swing_low:
            lower_wick = min(sweep_candle["close"], sweep_candle["open"]) - sweep_candle["low"]
            ratio = lower_wick / body
            if ratio >= config.SWEEP_WICK_RATIO:
                log.info(
                    "[SWEEP] BULLISH: wick=%.5f < swing_low=%.5f, ratio=%.2f (>=%.2f)",
                    sweep_candle["low"], swing_low, ratio, config.SWEEP_WICK_RATIO,
                )
                return {
                    "type": "BULLISH_SWEEP",
                    "sweep_level": swing_low,
                    "candle_low": sweep_candle["low"],
                }

        return None

    # ── Condition 3: Displacement + FVG ─────────────────────
    @staticmethod
    def _find_displacement_fvg(
        df: pd.DataFrame, sweep: dict, atr_value: float
    ) -> dict | None:
        """
        After the sweep candle (bar[-2]), the displacement candle
        (bar[-1]) must show strong movement creating a Fair Value Gap.
        FVG = gap between bar[-3] and bar[-1], skipping bar[-2].
        """
        if len(df) < 4:
            return None

        c1 = df.iloc[-3]  # candle before sweep
        c2 = df.iloc[-2]  # sweep candle
        c3 = df.iloc[-1]  # displacement candle (current)

        min_gap = atr_value * config.FVG_MIN_SIZE_ATR

        if sweep["type"] == "BEARISH_SWEEP":
            if c3["close"] >= c3["open"]:
                log.info("[FVG] Displacement NOT bearish (close=%.5f >= open=%.5f).",
                         c3["close"], c3["open"])
                return None
            gap = c1["low"] - c3["high"]
            log.info("[FVG] Bearish gap: c1_low=%.5f - c3_high=%.5f = %.5f (min=%.5f)",
                     c1["low"], c3["high"], gap, min_gap)
            if gap >= min_gap:
                fvg_top = max(c1["low"], c3["high"])
                fvg_bottom = min(c1["low"], c3["high"])
                return {"direction": "SELL", "fvg_top": fvg_top, "fvg_bottom": fvg_bottom}
            log.info("[FVG] Gap too small -- no FVG.")

        elif sweep["type"] == "BULLISH_SWEEP":
            if c3["close"] <= c3["open"]:
                log.info("[FVG] Displacement NOT bullish (close=%.5f <= open=%.5f).",
                         c3["close"], c3["open"])
                return None
            gap = c3["low"] - c1["high"]
            log.info("[FVG] Bullish gap: c3_low=%.5f - c1_high=%.5f = %.5f (min=%.5f)",
                     c3["low"], c1["high"], gap, min_gap)
            if gap >= min_gap:
                fvg_top = max(c3["low"], c1["high"])
                fvg_bottom = min(c3["low"], c1["high"])
                return {"direction": "BUY", "fvg_top": fvg_top, "fvg_bottom": fvg_bottom}
            log.info("[FVG] Gap too small -- no FVG.")

        return None

    # ── Condition 4: HTF trend filter ───────────────────────
    @staticmethod
    def _htf_trend_ok(df_m5: pd.DataFrame, direction: str) -> bool:
        """Check if M5 trend aligns with signal direction via EMA."""
        if df_m5.empty or len(df_m5) < config.HTF_EMA_PERIOD:
            log.info("[HTF] Not enough M5 data (%d bars) for EMA%d -- allowing.",
                     len(df_m5), config.HTF_EMA_PERIOD)
            return True
        ema = df_m5["close"].ewm(span=config.HTF_EMA_PERIOD, adjust=False).mean()
        last_close = df_m5["close"].iloc[-1]
        last_ema = ema.iloc[-1]
        if direction == "BUY" and last_close > last_ema:
            log.info("[HTF] M5 BULLISH (close=%.2f > EMA%d=%.2f) -- aligned with BUY.",
                     last_close, config.HTF_EMA_PERIOD, last_ema)
            return True
        if direction == "SELL" and last_close < last_ema:
            log.info("[HTF] M5 BEARISH (close=%.2f < EMA%d=%.2f) -- aligned with SELL.",
                     last_close, config.HTF_EMA_PERIOD, last_ema)
            return True
        log.info("[HTF] M5 MISALIGN (close=%.2f, EMA%d=%.2f) -- rejecting %s.",
                 last_close, config.HTF_EMA_PERIOD, last_ema, direction)
        return False

    # ── Condition 5: Multi-timeframe FVG (M5 confirmation) ──
    @staticmethod
    def _m5_fvg_confirms(
        df_m5: pd.DataFrame, direction: str, atr_value: float
    ) -> bool:
        """Check if recent M5 bars contain an FVG aligned with signal direction."""
        if not config.MTF_FVG_ENABLED:
            return True
        if df_m5.empty or len(df_m5) < config.MTF_FVG_LOOKBACK + 3:
            log.info("[MTF] Not enough M5 data (%d bars) — allowing.", len(df_m5))
            return True

        lookback = df_m5.iloc[-config.MTF_FVG_LOOKBACK:]
        min_gap = atr_value * config.MTF_FVG_MIN_SIZE_ATR

        for i in range(2, len(lookback)):
            c1 = lookback.iloc[i - 2]
            c3 = lookback.iloc[i]

            if direction == "BUY":
                gap = c3["low"] - c1["high"]
                if gap >= min_gap:
                    log.info(
                        "[MTF] M5 bullish FVG found: gap=%.5f (min=%.5f) at bar %d",
                        gap, min_gap, i,
                    )
                    return True
            elif direction == "SELL":
                gap = c1["low"] - c3["high"]
                if gap >= min_gap:
                    log.info(
                        "[MTF] M5 bearish FVG found: gap=%.5f (min=%.5f) at bar %d",
                        gap, min_gap, i,
                    )
                    return True

        log.info("[MTF] No M5 FVG in last %d bars — rejecting %s.",
                 config.MTF_FVG_LOOKBACK, direction)
        return False

    # ── Main evaluation ─────────────────────────────────────
    def evaluate(
        self, df_m1: pd.DataFrame, df_m5: pd.DataFrame
    ) -> Signal | None:
        """
        Run full strategy pipeline on latest data.
        Returns a Signal when all conditions align, else None.
        Populates self.last_eval with decision metadata for logging.
        """
        price = float(df_m1["close"].iloc[-1]) if not df_m1.empty else 0.0

        # Pre-compute indicators for logging
        atr_series = _atr(
            df_m1["high"], df_m1["low"], df_m1["close"],
            length=config.ATR_PERIOD,
        )
        current_atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

        bb_upper, bb_mid, bb_lower = _bbands(
            df_m1["close"], config.BB_PERIOD, config.BB_STD,
        )
        if not bb_upper.empty and len(bb_upper) >= config.BB_PERIOD:
            width_series = bb_upper - bb_lower
            avg_width = width_series.rolling(config.BB_PERIOD).mean()
            bb_exp = float(width_series.iloc[-1] / avg_width.iloc[-1]) if avg_width.iloc[-1] > 0 else 0.0
        else:
            bb_exp = 0.0

        self.last_eval = {
            "atr": current_atr,
            "bb_exp": bb_exp,
            "price": price,
            "fail": None,
            "detail": "",
        }

        # --- Condition 1: Volatility ---
        if not self._volatility_ok(df_m1):
            self.last_eval["fail"] = "VOLATILITY"
            atr_rel = current_atr / price if price > 0 else 0.0
            self.last_eval["detail"] = f"ATR={current_atr:.5f} ATR/p={atr_rel:.6f} thr={config.ATR_THRESHOLD} BB_exp={bb_exp:.3f}"
            return None

        # --- Condition 2: Liquidity Sweep ---
        sweep = self._find_liquidity_sweep(df_m1)
        if sweep is None:
            self.last_eval["fail"] = "SWEEP"
            self.last_eval["detail"] = "No liquidity sweep detected"
            return None

        # --- Condition 3: Displacement + FVG ---
        fvg = self._find_displacement_fvg(df_m1, sweep, current_atr)
        if fvg is None:
            self.last_eval["fail"] = "FVG"
            self.last_eval["detail"] = f"No displacement FVG after sweep@{sweep['sweep_level']:.5f}"
            return None
        log.info("[FVG] Displacement FVG confirmed: %s", fvg)

        # --- Condition 4: HTF trend filter ---
        if not self._htf_trend_ok(df_m5, fvg["direction"]):
            self.last_eval["fail"] = "HTF_TREND"
            self.last_eval["detail"] = f"M5 EMA{config.HTF_EMA_PERIOD} vs {fvg['direction']}"
            return None

        # --- Condition 5: Multi-timeframe FVG (M5) ---
        if not self._m5_fvg_confirms(df_m5, fvg["direction"], current_atr):
            self.last_eval["fail"] = "MTF_FVG"
            self.last_eval["detail"] = f"No M5 FVG confirms {fvg['direction']}"
            return None

        # --- Build Signal ---
        direction = fvg["direction"]
        fvg_mid = (fvg["fvg_top"] + fvg["fvg_bottom"]) / 2
        entry = fvg_mid

        if direction == "BUY":
            sl = sweep.get("candle_low", sweep["sweep_level"])
            risk = entry - sl
            tp = entry + risk * config.RISK_REWARD_RATIO
        else:
            sl = sweep.get("candle_high", sweep["sweep_level"])
            risk = sl - entry
            tp = entry - risk * config.RISK_REWARD_RATIO

        if risk <= 0:
            self.last_eval["fail"] = "RISK_ZERO"
            self.last_eval["detail"] = f"risk={risk:.5f}"
            log.info("[STRATEGY] Invalid risk distance (%.5f) -- skipping.", risk)
            return None

        # --- Condition 6: Max SL distance ---
        if current_atr > 0 and risk > current_atr * config.MAX_SL_ATR_MULT:
            self.last_eval["fail"] = "SL_TOO_WIDE"
            self.last_eval["detail"] = f"risk={risk:.5f} > {config.MAX_SL_ATR_MULT}x ATR={current_atr * config.MAX_SL_ATR_MULT:.5f}"
            log.info(
                "[STRATEGY] SL too wide: %.5f > %.1fx ATR (%.5f) -- skipping.",
                risk, config.MAX_SL_ATR_MULT, current_atr * config.MAX_SL_ATR_MULT,
            )
            return None
        log.info("[STRATEGY] SL distance OK: %.5f (max=%.5f)",
                 risk, current_atr * config.MAX_SL_ATR_MULT)

        signal = Signal(
            direction=direction,
            entry=round(entry, 5),
            sl=round(sl, 5),
            tp=round(tp, 5),
            fvg_zone=(round(fvg["fvg_bottom"], 5), round(fvg["fvg_top"], 5)),
            reason=f"Sweep@{sweep['sweep_level']:.5f} -> FVG retrace",
        )
        self.last_eval["fail"] = None
        self.last_eval["detail"] = f"{direction} sweep@{sweep['sweep_level']:.5f} FVG={fvg['fvg_bottom']:.5f}-{fvg['fvg_top']:.5f}"
        log.info("[SIGNAL] Generated: %s", signal)
        return signal


# ════════════════════════════════════════════════════════════
#  Sweep Scalper Strategy
#  M1 Fractal Sweep + Wick Rejection + HTF EMA Trend Filter
#  + Quality Filters (wick size, body confirm, displacement)
# ════════════════════════════════════════════════════════════
class SweepScalperStrategy:
    """
    Lightweight scalping strategy with quality filters:
      1. HTF EMA trend filter (BUY-only or SELL-only)
      2. Lightweight ATR volatility check
      3. 3-candle fractal swing detection on M1
      4. Sweep + wick rejection on last closed M1 candle
      5. Wick quality: sweep wick ≥ X% of ATR
      6. Body confirmation: candle closes in rejection direction
      7. Displacement: next bar confirms momentum
    Call `evaluate(df_m1, df_htf)` — same interface as Strategy.
    """

    def __init__(self):
        self.last_eval: dict = {}

    # ── Fractal swing detection ─────────────────────────────
    @staticmethod
    def _find_fractal_high(df: pd.DataFrame, n: int) -> float | None:
        """Most recent confirmed fractal swing high (N left + N right)."""
        highs = df["high"]
        if len(highs) < 2 * n + 1:
            return None
        for i in range(len(highs) - n - 1, n - 1, -1):
            is_swing = True
            for j in range(1, n + 1):
                if highs.iloc[i] <= highs.iloc[i - j] or highs.iloc[i] <= highs.iloc[i + j]:
                    is_swing = False
                    break
            if is_swing:
                return float(highs.iloc[i])
        return None

    @staticmethod
    def _find_fractal_low(df: pd.DataFrame, n: int) -> float | None:
        """Most recent confirmed fractal swing low (N left + N right)."""
        lows = df["low"]
        if len(lows) < 2 * n + 1:
            return None
        for i in range(len(lows) - n - 1, n - 1, -1):
            is_swing = True
            for j in range(1, n + 1):
                if lows.iloc[i] >= lows.iloc[i - j] or lows.iloc[i] >= lows.iloc[i + j]:
                    is_swing = False
                    break
            if is_swing:
                return float(lows.iloc[i])
        return None

    # ── Main evaluation ─────────────────────────────────────
    def evaluate(
        self, df_m1: pd.DataFrame, df_htf: pd.DataFrame
    ) -> Signal | None:
        """
        Run sweep scalper pipeline with quality filters.
        Returns a Signal when all conditions align, else None.
        """
        ema_period = config.SWEEP_EMA_PERIOD
        fractal_n = config.SWEEP_FRACTAL_N
        lookback_bars = config.SWEEP_LOOKBACK_BARS

        price = float(df_m1["close"].iloc[-1]) if not df_m1.empty else 0.0

        # Pre-compute ATR for quality checks
        atr_series = _atr(
            df_m1["high"], df_m1["low"], df_m1["close"],
            length=config.SWEEP_ATR_PERIOD,
        )
        current_atr = float(atr_series.iloc[-1]) if (
            not atr_series.empty and not np.isnan(atr_series.iloc[-1])
        ) else 0.0

        self.last_eval = {
            "atr": current_atr,
            "bb_exp": 0.0,
            "price": price,
            "fail": None,
            "detail": "",
        }

        if len(df_m1) < 20 or len(df_htf) < ema_period + 5:
            self.last_eval["fail"] = "DATA"
            self.last_eval["detail"] = f"M1={len(df_m1)} HTF={len(df_htf)} bars (need more)"
            return None

        # ── Step 1: HTF EMA Trend Filter ────────────────────
        ema_htf = df_htf["close"].ewm(span=ema_period, adjust=False).mean()
        last_close_htf = df_htf["close"].iloc[-2]
        last_ema_htf = ema_htf.iloc[-2]

        if last_close_htf > last_ema_htf:
            trend = "UP"
        elif last_close_htf < last_ema_htf:
            trend = "DOWN"
        else:
            self.last_eval["fail"] = "TREND"
            self.last_eval["detail"] = "HTF close exactly on EMA — skip"
            return None

        # ── Step 2: Lightweight volatility check ────────────
        if current_atr > 0 and price > 0:
            atr_ratio = current_atr / price
            if atr_ratio < config.SWEEP_MIN_ATR_RATIO:
                self.last_eval["fail"] = "VOL_LOW"
                self.last_eval["detail"] = (
                    f"ATR/price={atr_ratio:.6f} < min={config.SWEEP_MIN_ATR_RATIO}"
                )
                return None

        # ── Step 3: Find fractal swing levels on M1 ─────────
        lookback = df_m1.iloc[-(lookback_bars + 1):-1]
        swing_high = self._find_fractal_high(lookback, fractal_n)
        swing_low = self._find_fractal_low(lookback, fractal_n)

        # ── Step 4: Check last closed candle for sweep ──────
        candle = df_m1.iloc[-2]
        min_wick = current_atr * config.SWEEP_MIN_WICK_ATR if current_atr > 0 else 0.0

        # ── BUY: sweep below swing low + reject ─────────────
        if trend == "UP" and swing_low is not None:
            swept_below = candle["low"] < swing_low
            closed_above = candle["close"] > swing_low
            if swept_below and closed_above:
                sweep_wick = swing_low - candle["low"]

                # ── Filter A: Wick quality ──────────────────
                if min_wick > 0 and sweep_wick < min_wick:
                    self.last_eval["fail"] = "WICK_SMALL"
                    self.last_eval["detail"] = (
                        f"BUY wick={sweep_wick:.5f} < min={min_wick:.5f} "
                        f"({config.SWEEP_MIN_WICK_ATR:.0%} ATR)"
                    )
                    return None

                # ── Filter B: Body confirmation ─────────────
                if config.SWEEP_BODY_CONFIRM:
                    if candle["close"] <= candle["open"]:
                        self.last_eval["fail"] = "BODY_DIR"
                        self.last_eval["detail"] = (
                            f"BUY but candle bearish (O={candle['open']:.5f} "
                            f"C={candle['close']:.5f}) — no rejection"
                        )
                        return None

                # ── Filter C: Displacement bar ──────────────
                if config.SWEEP_DISPLACEMENT:
                    disp = df_m1.iloc[-1]  # current forming / latest bar
                    if disp["close"] <= disp["open"]:
                        self.last_eval["fail"] = "DISP_FAIL"
                        self.last_eval["detail"] = (
                            f"BUY but next bar bearish (O={disp['open']:.5f} "
                            f"C={disp['close']:.5f}) — no momentum"
                        )
                        return None

                entry = float(candle["close"])
                sl_anchor = float(candle["low"])
                risk = entry - sl_anchor
                if risk <= 0:
                    self.last_eval["fail"] = "RISK_ZERO"
                    self.last_eval["detail"] = f"risk={risk:.5f}"
                    return None
                tp = entry + risk * config.RISK_REWARD_RATIO
                log.info(
                    "[SWEEP_SCALP] BUY — sweep low=%.5f wick=%.5f "
                    "(candle low=%.5f, close=%.5f) HTF UP ATR=%.5f",
                    swing_low, sweep_wick, candle["low"],
                    candle["close"], current_atr,
                )
                self.last_eval["fail"] = None
                self.last_eval["detail"] = (
                    f"BUY sweep_low={swing_low:.5f} wick={sweep_wick:.5f} "
                    f"trend={trend} ema={last_ema_htf:.5f} atr={current_atr:.5f}"
                )
                return Signal(
                    direction="BUY",
                    entry=round(entry, 5),
                    sl=round(sl_anchor, 5),
                    tp=round(tp, 5),
                    fvg_zone=(round(swing_low, 5), round(entry, 5)),
                    reason=f"Sweep low@{swing_low:.5f} wick={sweep_wick:.5f}",
                )

        # ── SELL: sweep above swing high + reject ────────────
        if trend == "DOWN" and swing_high is not None:
            swept_above = candle["high"] > swing_high
            closed_below = candle["close"] < swing_high
            if swept_above and closed_below:
                sweep_wick = candle["high"] - swing_high

                # ── Filter A: Wick quality ──────────────────
                if min_wick > 0 and sweep_wick < min_wick:
                    self.last_eval["fail"] = "WICK_SMALL"
                    self.last_eval["detail"] = (
                        f"SELL wick={sweep_wick:.5f} < min={min_wick:.5f} "
                        f"({config.SWEEP_MIN_WICK_ATR:.0%} ATR)"
                    )
                    return None

                # ── Filter B: Body confirmation ─────────────
                if config.SWEEP_BODY_CONFIRM:
                    if candle["close"] >= candle["open"]:
                        self.last_eval["fail"] = "BODY_DIR"
                        self.last_eval["detail"] = (
                            f"SELL but candle bullish (O={candle['open']:.5f} "
                            f"C={candle['close']:.5f}) — no rejection"
                        )
                        return None

                # ── Filter C: Displacement bar ──────────────
                if config.SWEEP_DISPLACEMENT:
                    disp = df_m1.iloc[-1]
                    if disp["close"] >= disp["open"]:
                        self.last_eval["fail"] = "DISP_FAIL"
                        self.last_eval["detail"] = (
                            f"SELL but next bar bullish (O={disp['open']:.5f} "
                            f"C={disp['close']:.5f}) — no momentum"
                        )
                        return None

                entry = float(candle["close"])
                sl_anchor = float(candle["high"])
                risk = sl_anchor - entry
                if risk <= 0:
                    self.last_eval["fail"] = "RISK_ZERO"
                    self.last_eval["detail"] = f"risk={risk:.5f}"
                    return None
                tp = entry - risk * config.RISK_REWARD_RATIO
                log.info(
                    "[SWEEP_SCALP] SELL — sweep high=%.5f wick=%.5f "
                    "(candle high=%.5f, close=%.5f) HTF DOWN ATR=%.5f",
                    swing_high, sweep_wick, candle["high"],
                    candle["close"], current_atr,
                )
                self.last_eval["fail"] = None
                self.last_eval["detail"] = (
                    f"SELL sweep_high={swing_high:.5f} wick={sweep_wick:.5f} "
                    f"trend={trend} ema={last_ema_htf:.5f} atr={current_atr:.5f}"
                )
                return Signal(
                    direction="SELL",
                    entry=round(entry, 5),
                    sl=round(sl_anchor, 5),
                    tp=round(tp, 5),
                    fvg_zone=(round(entry, 5), round(swing_high, 5)),
                    reason=f"Sweep high@{swing_high:.5f} wick={sweep_wick:.5f}",
                )

        # ── No signal ───────────────────────────────────────
        self.last_eval["fail"] = "NO_SWEEP"
        sh_str = f"{swing_high:.5f}" if swing_high else "None"
        sl_str = f"{swing_low:.5f}" if swing_low else "None"
        self.last_eval["detail"] = f"trend={trend} swing_H={sh_str} swing_L={sl_str}"
        return None
