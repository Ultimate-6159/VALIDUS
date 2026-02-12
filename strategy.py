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
    """

    # ── Condition 1: Volatility check ───────────────────────
    @staticmethod
    def _volatility_ok(df: pd.DataFrame) -> bool:
        """Market must be volatile enough to trade."""
        atr = _atr(df["high"], df["low"], df["close"], length=config.ATR_PERIOD)
        if atr.empty or np.isnan(atr.iloc[-1]):
            return False
        current_atr = atr.iloc[-1]
        if current_atr < config.ATR_THRESHOLD:
            return False

        # Bollinger Band expansion check
        bbu, _, bbl = _bbands(df["close"], length=config.BB_PERIOD, std=config.BB_STD)
        if bbu.empty or np.isnan(bbu.iloc[-1]):
            return False
        width = bbu - bbl
        avg_width = width.rolling(config.BB_PERIOD).mean()
        if avg_width.iloc[-1] == 0:
            return False
        expansion = width.iloc[-1] / avg_width.iloc[-1]
        return expansion >= config.BB_EXPANSION_FACTOR

    # ── Condition 2: Liquidity sweep detection ──────────────
    @staticmethod
    def _find_liquidity_sweep(df: pd.DataFrame) -> dict | None:
        """
        Detect a fakeout where price wicked beyond a recent swing
        high/low then closed back inside — indicating a stop hunt.
        Returns dict with sweep info or None.
        """
        lb = config.SWING_LOOKBACK
        if len(df) < lb + 3:
            return None

        recent = df.iloc[-(lb + 3):-1]  # exclude the very last bar
        last = df.iloc[-1]
        prev = df.iloc[-2]

        swing_high = recent["high"].max()
        swing_low = recent["low"].min()

        body = abs(last["close"] - last["open"])
        body = max(body, 1e-10)

        # Bearish sweep (wick above swing high, close back below)
        if last["high"] > swing_high and last["close"] < swing_high:
            upper_wick = last["high"] - max(last["close"], last["open"])
            if upper_wick / body >= config.SWEEP_WICK_RATIO:
                return {
                    "type": "BEARISH_SWEEP",
                    "sweep_level": swing_high,
                    "candle_high": last["high"],
                }

        # Bullish sweep (wick below swing low, close back above)
        if last["low"] < swing_low and last["close"] > swing_low:
            lower_wick = min(last["close"], last["open"]) - last["low"]
            if lower_wick / body >= config.SWEEP_WICK_RATIO:
                return {
                    "type": "BULLISH_SWEEP",
                    "sweep_level": swing_low,
                    "candle_low": last["low"],
                }

        return None

    # ── Condition 3: Displacement + FVG ─────────────────────
    @staticmethod
    def _find_displacement_fvg(
        df: pd.DataFrame, sweep: dict, atr_value: float
    ) -> dict | None:
        """
        After the sweep candle, the NEXT candle must show strong
        displacement (large body) creating a Fair Value Gap.
        """
        if len(df) < 4:
            return None

        c1 = df.iloc[-3]  # candle before sweep
        c2 = df.iloc[-2]  # sweep candle
        c3 = df.iloc[-1]  # displacement candle (current)

        min_gap = atr_value * config.FVG_MIN_SIZE_ATR

        if sweep["type"] == "BEARISH_SWEEP":
            # Bearish displacement: c3 must be strongly bearish
            if c3["close"] >= c3["open"]:
                return None
            # FVG: gap between c1.low and c3.high
            gap = c1["low"] - c3["high"]
            if gap >= min_gap:
                return {
                    "direction": "SELL",
                    "fvg_top": c1["low"],
                    "fvg_bottom": c3["high"],
                }

        elif sweep["type"] == "BULLISH_SWEEP":
            # Bullish displacement: c3 must be strongly bullish
            if c3["close"] <= c3["open"]:
                return None
            # FVG: gap between c3.low and c1.high
            gap = c3["low"] - c1["high"]
            if gap >= min_gap:
                return {
                    "direction": "BUY",
                    "fvg_top": c3["low"],
                    "fvg_bottom": c1["high"],
                }

        return None

    # ── Main evaluation ─────────────────────────────────────
    def evaluate(
        self, df_m1: pd.DataFrame, df_m5: pd.DataFrame
    ) -> Signal | None:
        """
        Run full strategy pipeline on latest data.
        Returns a Signal when all conditions align, else None.
        """
        # --- Condition 1: Volatility ---
        if not self._volatility_ok(df_m1):
            log.debug("Volatility check failed — skipping.")
            return None

        # --- Condition 2: Liquidity Sweep ---
        sweep = self._find_liquidity_sweep(df_m1)
        if sweep is None:
            log.debug("No liquidity sweep detected.")
            return None
        log.info("[SWEEP] Liquidity sweep detected: %s", sweep)

        # --- ATR for sizing ---
        atr_series = _atr(
            df_m1["high"], df_m1["low"], df_m1["close"],
            length=config.ATR_PERIOD,
        )
        current_atr = atr_series.iloc[-1] if not atr_series.empty else 0

        # --- Condition 3: Displacement + FVG ---
        fvg = self._find_displacement_fvg(df_m1, sweep, current_atr)
        if fvg is None:
            log.debug("No displacement / FVG after sweep.")
            return None
        log.info("[FVG] Displacement FVG found: %s", fvg)

        # --- Build Signal ---
        direction = fvg["direction"]
        fvg_mid = (fvg["fvg_top"] + fvg["fvg_bottom"]) / 2
        entry = fvg_mid  # enter at FVG midpoint retrace

        if direction == "BUY":
            sl = sweep.get("candle_low", sweep["sweep_level"])
            risk = entry - sl
            tp = entry + risk * config.RISK_REWARD_RATIO
        else:
            sl = sweep.get("candle_high", sweep["sweep_level"])
            risk = sl - entry
            tp = entry - risk * config.RISK_REWARD_RATIO

        if risk <= 0:
            log.debug("Invalid risk distance — skipping.")
            return None

        signal = Signal(
            direction=direction,
            entry=round(entry, 5),
            sl=round(sl, 5),
            tp=round(tp, 5),
            fvg_zone=(round(fvg["fvg_bottom"], 5), round(fvg["fvg_top"], 5)),
            reason=f"Sweep@{sweep['sweep_level']:.5f} -> FVG retrace",
        )
        log.info("[SIGNAL] %s", signal)
        return signal
