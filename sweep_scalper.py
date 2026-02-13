# ============================================================
#  SWEEP SCALPER — M1 Liquidity Sweep + M15 Trend Filter
#  High-Frequency Scalping Bot for XAUUSD on MetaTrader 5
#  Strategy: 3-Candle Fractal Sweep → Wick Rejection → Entry
# ============================================================
from __future__ import annotations

import logging
import logging.handlers
import time
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import MetaTrader5 as mt5


# ════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════
MT5_LOGIN    = 415146568
MT5_PASSWORD = "Ultimate@6159"
MT5_SERVER   = "Exness-MT5Trial14"
MT5_PATH     = r"C:\Program Files\MetaTrader 5\terminal64.exe"

SYMBOL       = "XAUUSDm"          # Broker-specific symbol name
MAGIC        = 615901             # Unique magic number (different from VALIDUS)

# Strategy
EMA_PERIOD   = 50                 # EMA period on M15 for trend filter
FRACTAL_N    = 1                  # Fractal detection: 1 left + 1 right candle
SL_PAD_PTS   = 20                 # Extra padding on SL in points (Gold)

# Money Management
RISK_PCT     = 3.0                # Risk 3% of balance per trade
RR_RATIO     = 2.0                # Risk:Reward = 1:2
MIN_LOT      = 0.01
MAX_TRADES   = 1                  # Only 1 open trade at a time

# Breakeven
BE_RATIO     = 1.0                # Move SL to BE when profit >= 1x risk
BE_OFFSET_PTS = 10                # +10 points past entry to cover commissions

# Safety Filters
MAX_SPREAD_PTS  = 35              # Max spread in points (3.5 pips)
SESSION_START_H = 8               # Server time hour (08:00)
SESSION_END_H   = 22              # Server time hour (22:00)

# Logging
LOG_FILE     = "logs/sweep_scalper.log"
LOG_LEVEL    = "INFO"


# ════════════════════════════════════════════════════════════
#  Logger
# ════════════════════════════════════════════════════════════
def _setup_logger() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("SWEEP_SCALPER")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


log = _setup_logger()


# ════════════════════════════════════════════════════════════
#  MT5 Connection
# ════════════════════════════════════════════════════════════
def connect_mt5() -> bool:
    if not mt5.initialize(path=MT5_PATH):
        log.error("[MT5] initialize failed: %s", mt5.last_error())
        return False
    if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        log.error("[MT5] login failed: %s", mt5.last_error())
        return False
    info = mt5.account_info()
    log.info("[MT5] Connected — account %d | Balance: %.2f", info.login, info.balance)
    return True


def ensure_connection() -> bool:
    info = mt5.account_info()
    if info is not None:
        return True
    log.warning("[MT5] Disconnected — reconnecting...")
    for attempt in range(1, 4):
        if connect_mt5():
            return True
        log.warning("  attempt %d/3 failed", attempt)
        time.sleep(2)
    return False


# ════════════════════════════════════════════════════════════
#  Data Fetching
# ════════════════════════════════════════════════════════════
def fetch_bars(symbol: str, timeframe: int, count: int = 200) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df


# ════════════════════════════════════════════════════════════
#  Indicator: EMA (native — no pandas_ta needed)
# ════════════════════════════════════════════════════════════
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ════════════════════════════════════════════════════════════
#  Strategy: Fractal Swing Detection (3-candle: L=1, R=1)
# ════════════════════════════════════════════════════════════
def find_recent_swing_high(df: pd.DataFrame, n: int = FRACTAL_N) -> float | None:
    """Find the most recent confirmed fractal swing high.
    A swing high at index i requires:
      high[i] > high[i-1] AND high[i] > high[i+1]
    We skip the last candle (it has no right confirmation yet).
    """
    highs = df["high"]
    if len(highs) < 2 * n + 1:
        return None
    # Scan from most recent confirmed (skip last n candles)
    for i in range(len(highs) - n - 1, n - 1, -1):
        is_swing = True
        for j in range(1, n + 1):
            if highs.iloc[i] <= highs.iloc[i - j] or highs.iloc[i] <= highs.iloc[i + j]:
                is_swing = False
                break
        if is_swing:
            return float(highs.iloc[i])
    return None


def find_recent_swing_low(df: pd.DataFrame, n: int = FRACTAL_N) -> float | None:
    """Find the most recent confirmed fractal swing low."""
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


# ════════════════════════════════════════════════════════════
#  Strategy: Check Trading Conditions
# ════════════════════════════════════════════════════════════
def check_signal(df_m1: pd.DataFrame, df_m15: pd.DataFrame) -> dict | None:
    """
    Evaluate the Sweep & Rejection strategy.
    Returns signal dict or None.

    Logic:
      1. M15 EMA50 trend filter → BUY-only or SELL-only
      2. Find 3-candle fractal swing high/low on M1
      3. Last closed M1 candle sweeps the swing level
         but closes back inside (wick rejection)
    """
    if len(df_m1) < 20 or len(df_m15) < EMA_PERIOD + 5:
        return None

    # ── Step 1: M15 Trend Filter ────────────────────────────
    ema_m15 = ema(df_m15["close"], EMA_PERIOD)
    last_close_m15 = df_m15["close"].iloc[-1]
    last_ema_m15 = ema_m15.iloc[-1]

    if last_close_m15 > last_ema_m15:
        trend = "UP"
    elif last_close_m15 < last_ema_m15:
        trend = "DOWN"
    else:
        return None  # exactly on EMA — skip

    # ── Step 2: Find fractal swing levels on M1 ─────────────
    # Use last 30 bars for swing detection (excluding current candle)
    lookback = df_m1.iloc[-30:]
    swing_high = find_recent_swing_high(lookback)
    swing_low = find_recent_swing_low(lookback)

    # ── Step 3: Check last closed candle for sweep ──────────
    candle = df_m1.iloc[-1]  # last closed M1 candle

    # ── BUY Signal: sweep below swing low + reject ──────────
    if trend == "UP" and swing_low is not None:
        swept_below = candle["low"] < swing_low
        closed_above = candle["close"] > swing_low
        if swept_below and closed_above:
            log.info(
                "[SIGNAL] BUY — sweep low=%.2f (candle low=%.2f, close=%.2f) M15 UP",
                swing_low, candle["low"], candle["close"],
            )
            return {
                "direction": "BUY",
                "sweep_level": swing_low,
                "sweep_extreme": candle["low"],  # SL anchor
            }

    # ── SELL Signal: sweep above swing high + reject ────────
    if trend == "DOWN" and swing_high is not None:
        swept_above = candle["high"] > swing_high
        closed_below = candle["close"] < swing_high
        if swept_above and closed_below:
            log.info(
                "[SIGNAL] SELL — sweep high=%.2f (candle high=%.2f, close=%.2f) M15 DOWN",
                swing_high, candle["high"], candle["close"],
            )
            return {
                "direction": "SELL",
                "sweep_level": swing_high,
                "sweep_extreme": candle["high"],  # SL anchor
            }

    return None


# ════════════════════════════════════════════════════════════
#  Lot Size Calculation
# ════════════════════════════════════════════════════════════
def calculate_lot_size(symbol: str, sl_distance_price: float) -> float:
    """
    Lot = (Balance * RiskPct) / (SL_distance_points * Tick_Value)
    """
    info = mt5.account_info()
    sym = mt5.symbol_info(symbol)
    if info is None or sym is None or sl_distance_price <= 0:
        return MIN_LOT

    risk_money = info.balance * (RISK_PCT / 100.0)
    tick_value = sym.trade_tick_value
    tick_size = sym.trade_tick_size

    if tick_value == 0 or tick_size == 0:
        return MIN_LOT

    sl_ticks = sl_distance_price / tick_size
    lot = risk_money / (sl_ticks * tick_value)
    lot = max(sym.volume_min, min(lot, sym.volume_max))
    lot = round(lot, 2)
    return max(lot, MIN_LOT)


# ════════════════════════════════════════════════════════════
#  Trade Execution
# ════════════════════════════════════════════════════════════
def execute_trade(symbol: str, signal: dict) -> bool:
    """Open a market order based on the signal."""
    sym = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if sym is None or tick is None:
        log.error("[EXEC] Symbol info unavailable for %s", symbol)
        return False

    # ── Spread check ────────────────────────────────────────
    spread_pts = (tick.ask - tick.bid) / sym.point
    if spread_pts > MAX_SPREAD_PTS:
        log.warning("[EXEC] Spread %.1f > max %d pts — skipping", spread_pts, MAX_SPREAD_PTS)
        return False

    direction = signal["direction"]
    pad = SL_PAD_PTS * sym.point  # convert points to price

    if direction == "BUY":
        price = tick.ask
        sl = signal["sweep_extreme"] - pad      # below the sweep wick
        sl_dist = price - sl
        tp = price + sl_dist * RR_RATIO          # 1:2 TP
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        sl = signal["sweep_extreme"] + pad      # above the sweep wick
        sl_dist = sl - price
        tp = price - sl_dist * RR_RATIO          # 1:2 TP
        order_type = mt5.ORDER_TYPE_SELL

    if sl_dist <= 0:
        log.warning("[EXEC] Invalid SL distance %.5f — skipping", sl_dist)
        return False

    lot = calculate_lot_size(symbol, sl_dist)

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         order_type,
        "price":        round(price, sym.digits),
        "sl":           round(sl, sym.digits),
        "tp":           round(tp, sym.digits),
        "deviation":    20,
        "magic":        MAGIC,
        "comment":      "SWEEP_SCALPER",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        log.error("[EXEC] order_send returned None: %s", mt5.last_error())
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error("[EXEC] Order failed [%d]: %s", result.retcode, result.comment)
        return False

    log.info(
        "[OPEN] %s %s %.2f lot @ %.2f | SL=%.2f TP=%.2f | risk=$%.2f | ticket=%d",
        direction, symbol, lot, price, sl, tp,
        mt5.account_info().balance * RISK_PCT / 100 if mt5.account_info() else 0,
        result.order,
    )
    return True


# ════════════════════════════════════════════════════════════
#  Position Management: Breakeven
# ════════════════════════════════════════════════════════════
def manage_breakeven() -> None:
    """Move SL to breakeven (+10 pts) when profit >= 1:1 of risk."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return

    for pos in positions:
        if pos.magic != MAGIC:
            continue

        sym = mt5.symbol_info(pos.symbol)
        tick = mt5.symbol_info_tick(pos.symbol)
        if sym is None or tick is None:
            continue

        offset = BE_OFFSET_PTS * sym.point

        if pos.type == mt5.ORDER_TYPE_BUY:
            risk_dist = pos.price_open - pos.sl
            current_profit = tick.bid - pos.price_open
            be_price = pos.price_open + offset
            # Already at or past breakeven?
            if pos.sl >= be_price:
                continue
            if risk_dist > 0 and current_profit >= risk_dist * BE_RATIO:
                _move_sl(pos, be_price, sym)

        elif pos.type == mt5.ORDER_TYPE_SELL:
            risk_dist = pos.sl - pos.price_open
            current_profit = pos.price_open - tick.ask
            be_price = pos.price_open - offset
            if pos.sl <= be_price and pos.sl > 0:
                continue
            if risk_dist > 0 and current_profit >= risk_dist * BE_RATIO:
                _move_sl(pos, be_price, sym)


def _move_sl(pos, new_sl: float, sym) -> None:
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": pos.ticket,
        "symbol":   pos.symbol,
        "sl":       round(new_sl, sym.digits),
        "tp":       pos.tp,
        "magic":    MAGIC,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info("[BE] SL → %.2f for ticket=%d", new_sl, pos.ticket)
    else:
        comment = result.comment if result else str(mt5.last_error())
        log.warning("[BE] Move failed ticket=%d: %s", pos.ticket, comment)


# ════════════════════════════════════════════════════════════
#  Safety: Session Filter (Server Time)
# ════════════════════════════════════════════════════════════
def is_in_session() -> bool:
    """Check if current server time is within trading hours."""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return False
    # MT5 tick.time is server time as unix timestamp
    server_time = dt.datetime.utcfromtimestamp(tick.time)
    hour = server_time.hour
    return SESSION_START_H <= hour < SESSION_END_H


# ════════════════════════════════════════════════════════════
#  Safety: Max Open Trades
# ════════════════════════════════════════════════════════════
def count_my_positions() -> int:
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return 0
    return sum(1 for p in positions if p.magic == MAGIC)


# ════════════════════════════════════════════════════════════
#  Main Loop
# ════════════════════════════════════════════════════════════
def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print(
        "\n"
        "╔══════════════════════════════════════════════╗\n"
        "║     SWEEP SCALPER v1.0 — M1 Liquidity       ║\n"
        "║     XAUUSD · M15 Trend · Wick Rejection     ║\n"
        "╚══════════════════════════════════════════════╝\n"
    )

    if not connect_mt5():
        log.error("Cannot start — MT5 connection failed.")
        return

    info = mt5.account_info()
    log.info("[START] Balance: $%.2f | Risk: %.1f%% | RR: 1:%.1f | Max trades: %d",
             info.balance, RISK_PCT, RR_RATIO, MAX_TRADES)

    last_m1_time = None
    scan_count = 0
    last_heartbeat = time.time()

    try:
        while True:
            # ── Connection check ────────────────────────────
            if not ensure_connection():
                time.sleep(5)
                continue

            # ── Position management (every tick) ────────────
            manage_breakeven()

            # ── Fetch M1 data ───────────────────────────────
            df_m1 = fetch_bars(SYMBOL, mt5.TIMEFRAME_M1, 200)
            if df_m1.empty:
                time.sleep(1)
                continue

            # ── Wait for new M1 bar ─────────────────────────
            current_m1_time = df_m1.index[-1]
            if current_m1_time == last_m1_time:
                # Heartbeat every 60s
                now = time.time()
                if now - last_heartbeat >= 60:
                    acct = mt5.account_info()
                    eq = acct.equity if acct else 0
                    n_pos = count_my_positions()
                    log.info(
                        "[HEARTBEAT] Equity: %.2f | Positions: %d | Scans: %d",
                        eq, n_pos, scan_count,
                    )
                    scan_count = 0
                    last_heartbeat = now
                time.sleep(0.5)
                continue

            last_m1_time = current_m1_time
            scan_count += 1

            tick = mt5.symbol_info_tick(SYMBOL)
            bid = tick.bid if tick else 0
            log.info("[SCAN] New M1 bar %s | Bid: %.2f", current_m1_time, bid)

            # ── Session filter ──────────────────────────────
            if not is_in_session():
                log.info("[SKIP] Outside session (%02d:00-%02d:00)", SESSION_START_H, SESSION_END_H)
                continue

            # ── Max trades filter ───────────────────────────
            if count_my_positions() >= MAX_TRADES:
                log.info("[SKIP] Max trades (%d) reached", MAX_TRADES)
                continue

            # ── Fetch M15 data & evaluate ───────────────────
            df_m15 = fetch_bars(SYMBOL, mt5.TIMEFRAME_M15, 200)
            signal = check_signal(df_m1, df_m15)

            if signal is not None:
                log.info("[SIGNAL] %s detected — executing...", signal["direction"])
                if execute_trade(SYMBOL, signal):
                    log.info("[OK] Trade opened successfully.")
                else:
                    log.warning("[FAIL] Trade execution failed.")
            else:
                # Compact log: show trend + nearest levels
                ema_val = ema(df_m15["close"], EMA_PERIOD).iloc[-1] if not df_m15.empty else 0
                trend_dir = "UP" if bid > ema_val else "DOWN"
                sh = find_recent_swing_high(df_m1.iloc[-30:])
                sl = find_recent_swing_low(df_m1.iloc[-30:])
                log.info(
                    "[NO_SIGNAL] trend=%s ema15=%.2f | swing_H=%.2f swing_L=%.2f",
                    trend_dir, ema_val,
                    sh if sh else 0, sl if sl else 0,
                )

            time.sleep(0.5)

    except KeyboardInterrupt:
        log.info("[STOP] User interrupted — shutting down.")
    finally:
        mt5.shutdown()
        log.info("[SHUTDOWN] Complete.")


if __name__ == "__main__":
    main()
