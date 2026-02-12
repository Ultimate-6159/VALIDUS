# ============================================================
#  VALIDUS — Backtest Engine
#  Usage:
#    python backtest.py --symbol XAUUSD --bars 50000
#    python backtest.py --csv data/XAUUSD_M1.csv
#    python backtest.py --symbol XAUUSD --bars 50000 --no-limit
# ============================================================
from __future__ import annotations

import argparse
import logging
import os
import sys
import datetime as dt

import numpy as np
import pandas as pd

import config
from strategy import Strategy, Signal, _atr


# Suppress noisy strategy logs during backtest
logging.getLogger("VALIDUS").setLevel(logging.WARNING)


# ────────────────────────────────────────────────────────────
#  Data structures
# ────────────────────────────────────────────────────────────
class BacktestPosition:
    """Tracks one simulated position."""

    def __init__(
        self,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        lot: float,
        entry_idx: int,
        entry_time: pd.Timestamp,
    ):
        self.direction = direction
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp
        self.lot = lot
        self.original_lot = lot
        self.entry_idx = entry_idx
        self.entry_time = entry_time
        self.exit_price: float = 0.0
        self.exit_time: pd.Timestamp | None = None
        self.pnl: float = 0.0
        self.exit_reason: str = ""
        self.is_at_be: bool = False
        self.partial_closed: bool = False
        self.partial_pnl: float = 0.0


class PendingOrder:
    """Pending limit order in backtest."""

    def __init__(
        self,
        signal: Signal,
        lot: float,
        created_idx: int,
        created_time: pd.Timestamp,
    ):
        self.signal = signal
        self.lot = lot
        self.created_idx = created_idx
        self.created_time = created_time


# ────────────────────────────────────────────────────────────
#  Backtest Engine
# ────────────────────────────────────────────────────────────
class BacktestEngine:
    """Bar-by-bar backtester that mirrors live execution logic."""

    WARMUP = 200  # bars needed before first signal

    def __init__(
        self,
        df_m1: pd.DataFrame,
        symbol: str = "XAUUSD",
        initial_balance: float | None = None,
        contract_size: float | None = None,
        tick_size: float | None = None,
    ):
        self.df_m1 = df_m1
        self.symbol = symbol
        self.initial_balance = initial_balance or config.BACKTEST_INITIAL_BAL
        self.contract_size = contract_size or config.BACKTEST_CONTRACT_SIZE
        self.tick_size = tick_size or config.BACKTEST_TICK_SIZE
        self.slippage = config.BACKTEST_SLIPPAGE_PTS * self.tick_size
        self.commission = config.BACKTEST_COMMISSION

        # Resample M1 → M5
        self.df_m5 = self._resample_m5(df_m1)

    # ── Helpers ─────────────────────────────────────────────
    @staticmethod
    def _resample_m5(df_m1: pd.DataFrame) -> pd.DataFrame:
        return df_m1.resample("5min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
        }).dropna()

    def _calc_pnl(self, pos: BacktestPosition) -> float:
        if pos.direction == "BUY":
            raw = (pos.exit_price - pos.entry_price) * pos.lot * self.contract_size
        else:
            raw = (pos.entry_price - pos.exit_price) * pos.lot * self.contract_size
        raw += pos.partial_pnl
        return raw - self.commission * pos.original_lot * 2  # round-trip commission

    def _calc_lot(self, balance: float, sl_dist: float) -> float:
        if config.RISK_PCT <= 0 or sl_dist <= 0:
            return config.LOT_SIZE
        risk_money = balance * (config.RISK_PCT / 100.0)
        ticks = sl_dist / self.tick_size
        if ticks == 0:
            return config.LOT_SIZE
        tick_value = 1.0  # $1 per tick per lot for XAUUSD
        lot = risk_money / (ticks * tick_value)
        lot = max(0.01, min(lot, 100.0))
        return round(lot, 2)

    def _get_atr(self, idx: int) -> float:
        start = max(0, idx - config.ATR_PERIOD - 5)
        sl = self.df_m1.iloc[start : idx + 1]
        if len(sl) < config.ATR_PERIOD:
            return 0.0
        atr = _atr(sl["high"], sl["low"], sl["close"], config.ATR_PERIOD)
        val = atr.iloc[-1]
        return 0.0 if np.isnan(val) else float(val)

    # ── Position management (BE + trailing) per bar ─────────
    def _manage_position(self, pos: BacktestPosition, bar, idx: int) -> None:
        """Simulate breakeven + trailing stop management on open position."""
        if pos.direction == "BUY":
            profit_dist = bar["close"] - pos.entry_price
            tp_dist = pos.tp - pos.entry_price if pos.tp > 0 else 0
            be_price = pos.entry_price + self.slippage
        else:
            profit_dist = pos.entry_price - bar["close"]
            tp_dist = pos.entry_price - pos.tp if pos.tp > 0 else 0
            be_price = pos.entry_price - self.slippage

        if tp_dist <= 0:
            return

        # Phase 1: Breakeven
        if not pos.is_at_be and profit_dist >= tp_dist * config.BREAKEVEN_PCT:
            if pos.direction == "BUY" and pos.sl < be_price:
                pos.sl = be_price
                pos.is_at_be = True
            elif pos.direction == "SELL" and (pos.sl > be_price or pos.sl == 0):
                pos.sl = be_price
                pos.is_at_be = True

        # Phase 2: Trailing stop
        if config.TRAILING_STOP_ENABLED and pos.is_at_be:
            atr_val = self._get_atr(idx)
            if atr_val > 0:
                trail_dist = atr_val * config.TRAILING_ATR_MULT
                if pos.direction == "BUY":
                    candidate = bar["close"] - trail_dist
                    if candidate > pos.sl:
                        pos.sl = candidate
                else:
                    candidate = bar["close"] + trail_dist
                    if candidate < pos.sl:
                        pos.sl = candidate

    # ── Partial TP ──────────────────────────────────────────
    def _check_partial_tp(self, pos: BacktestPosition, bar) -> None:
        """Close partial lot at RR 1:1 level and move SL to BE."""
        if not config.PARTIAL_TP_ENABLED or pos.partial_closed:
            return
        risk_dist = abs(pos.entry_price - pos.sl)
        if risk_dist <= 0:
            return

        partial_dist = risk_dist * config.PARTIAL_TP_RR
        if pos.direction == "BUY":
            partial_price = pos.entry_price + partial_dist
            if bar["high"] >= partial_price:
                close_lot = pos.lot * config.PARTIAL_TP_RATIO
                pos.partial_pnl += (partial_price - pos.entry_price) * close_lot * self.contract_size
                pos.lot = round(max(0.01, pos.lot - close_lot), 2)
                pos.partial_closed = True
                pos.sl = pos.entry_price + self.slippage
                pos.is_at_be = True
        else:
            partial_price = pos.entry_price - partial_dist
            if bar["low"] <= partial_price:
                close_lot = pos.lot * config.PARTIAL_TP_RATIO
                pos.partial_pnl += (pos.entry_price - partial_price) * close_lot * self.contract_size
                pos.lot = round(max(0.01, pos.lot - close_lot), 2)
                pos.partial_closed = True
                pos.sl = pos.entry_price - self.slippage
                pos.is_at_be = True

    # ── Check SL / TP hit ───────────────────────────────────
    def _check_sl_tp(self, pos: BacktestPosition, bar, idx: int) -> bool:
        """Return True if position closed by SL or TP."""
        bar_time = self.df_m1.index[idx]

        if pos.direction == "BUY":
            # Conservative: check SL first
            if bar["low"] <= pos.sl:
                pos.exit_price = pos.sl
                pos.exit_time = bar_time
                pos.exit_reason = "SL"
                pos.pnl = self._calc_pnl(pos)
                return True
            if pos.tp > 0 and bar["high"] >= pos.tp:
                pos.exit_price = pos.tp
                pos.exit_time = bar_time
                pos.exit_reason = "TP"
                pos.pnl = self._calc_pnl(pos)
                return True
        else:
            if bar["high"] >= pos.sl:
                pos.exit_price = pos.sl
                pos.exit_time = bar_time
                pos.exit_reason = "SL"
                pos.pnl = self._calc_pnl(pos)
                return True
            if pos.tp > 0 and bar["low"] <= pos.tp:
                pos.exit_price = pos.tp
                pos.exit_time = bar_time
                pos.exit_reason = "TP"
                pos.pnl = self._calc_pnl(pos)
                return True
        return False

    # ── Main run ────────────────────────────────────────────
    def run(self) -> dict:
        config.apply_tier(self.initial_balance)
        strategy = Strategy()

        positions: list[BacktestPosition] = []
        closed_trades: list[BacktestPosition] = []
        pending_orders: list[PendingOrder] = []
        equity_curve: list[float] = [self.initial_balance]
        balance = self.initial_balance
        last_signal_bar = -999

        total_bars = len(self.df_m1)

        for i in range(self.WARMUP, total_bars):
            bar = self.df_m1.iloc[i]
            bar_time = self.df_m1.index[i]

            # ── Session filter ──────────────────────────────
            if config.SESSION_FILTER_ENABLED:
                hour = bar_time.hour
                s, e = config.SESSION_START_UTC, config.SESSION_END_UTC
                in_session = (s <= hour < e) if s <= e else (hour >= s or hour < e)
                if not in_session:
                    equity_curve.append(equity_curve[-1])
                    continue

            # ── Check pending limit orders for fill ─────────
            filled_pending: list[PendingOrder] = []
            for po in pending_orders:
                filled = False
                if po.signal.direction == "BUY" and bar["low"] <= po.signal.entry:
                    ep = po.signal.entry + self.slippage
                    sd = abs(ep - po.signal.sl)
                    tp = ep + sd * config.RISK_REWARD_RATIO
                    positions.append(BacktestPosition(
                        "BUY", ep, po.signal.sl, tp, po.lot, i, bar_time,
                    ))
                    filled = True
                elif po.signal.direction == "SELL" and bar["high"] >= po.signal.entry:
                    ep = po.signal.entry - self.slippage
                    sd = abs(po.signal.sl - ep)
                    tp = ep - sd * config.RISK_REWARD_RATIO
                    positions.append(BacktestPosition(
                        "SELL", ep, po.signal.sl, tp, po.lot, i, bar_time,
                    ))
                    filled = True

                if filled:
                    filled_pending.append(po)
                elif (bar_time - po.created_time).total_seconds() > config.LIMIT_ORDER_EXPIRY_SEC:
                    filled_pending.append(po)  # expired

            for po in filled_pending:
                pending_orders.remove(po)

            # ── Check open positions for SL / TP ────────────
            still_open: list[BacktestPosition] = []
            for pos in positions:
                self._check_partial_tp(pos, bar)
                if self._check_sl_tp(pos, bar, i):
                    closed_trades.append(pos)
                    balance += pos.pnl
                else:
                    self._manage_position(pos, bar, i)
                    still_open.append(pos)
            positions = still_open

            # ── Equity snapshot ─────────────────────────────
            unrealized = 0.0
            for pos in positions:
                if pos.direction == "BUY":
                    unrealized += (bar["close"] - pos.entry_price) * pos.lot * self.contract_size
                else:
                    unrealized += (pos.entry_price - bar["close"]) * pos.lot * self.contract_size
            equity_curve.append(balance + unrealized)

            # ── Signal evaluation ───────────────────────────
            if len(positions) + len(pending_orders) >= config.MAX_POSITIONS:
                continue
            if i - last_signal_bar < config.SIGNAL_COOLDOWN_BARS:
                continue

            start_idx = max(0, i - 199)
            df_slice = self.df_m1.iloc[start_idx : i + 1]
            df_m5_slice = self.df_m5[self.df_m5.index <= bar_time].iloc[-200:]

            signal = strategy.evaluate(df_slice, df_m5_slice)
            if signal is None:
                continue

            last_signal_bar = i
            sl_dist = abs(signal.entry - signal.sl)
            lot = self._calc_lot(balance, sl_dist)

            if config.USE_LIMIT_ORDER:
                pending_orders.append(PendingOrder(signal, lot, i, bar_time))
            else:
                # Market fill at bar close ± slippage
                if signal.direction == "BUY":
                    ep = bar["close"] + self.slippage
                else:
                    ep = bar["close"] - self.slippage
                sd = abs(ep - signal.sl)
                if signal.direction == "BUY":
                    tp = ep + sd * config.RISK_REWARD_RATIO
                else:
                    tp = ep - sd * config.RISK_REWARD_RATIO
                positions.append(BacktestPosition(
                    signal.direction, ep, signal.sl, tp, lot, i, bar_time,
                ))

            # ── Progress (every 10 %) ───────────────────────
            pct = (i - self.WARMUP) / max(1, total_bars - self.WARMUP) * 100
            if i % max(1, (total_bars // 10)) == 0:
                print(f"  ... {pct:5.1f}%  trades={len(closed_trades)}  "
                      f"balance={balance:,.2f}", flush=True)

        # ── Close remaining positions at last close ─────────
        last_bar = self.df_m1.iloc[-1]
        last_time = self.df_m1.index[-1]
        for pos in positions:
            pos.exit_price = last_bar["close"]
            pos.exit_time = last_time
            pos.exit_reason = "END"
            pos.pnl = self._calc_pnl(pos)
            closed_trades.append(pos)
            balance += pos.pnl

        return self._compute_metrics(closed_trades, equity_curve, balance)

    # ── Metrics ─────────────────────────────────────────────
    def _compute_metrics(
        self,
        trades: list[BacktestPosition],
        equity_curve: list[float],
        final_balance: float,
    ) -> dict:
        if not trades:
            return {"total_trades": 0, "equity_curve": equity_curve, "trades": []}

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0

        # Max drawdown
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = peak - eq
        max_dd = float(dd.max())
        max_dd_pct = float((dd / np.where(peak > 0, peak, 1)).max() * 100)

        # Sharpe (annualized, assuming M1 bars ~252*14*60 bars/year)
        returns = np.diff(eq) / eq[:-1]
        sharpe = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(252 * 14 * 60))

        # Trade durations in bars
        durations = [
            (t.exit_time - t.entry_time).total_seconds() / 60
            for t in trades if t.exit_time is not None
        ]
        partial_count = sum(1 for t in trades if t.partial_closed)
        tp_count = sum(1 for t in trades if t.exit_reason == "TP")
        sl_count = sum(1 for t in trades if t.exit_reason == "SL")
        be_count = sum(1 for t in trades if t.exit_reason == "SL" and t.is_at_be)

        # Build trades list for CSV export
        trade_records = []
        for t in trades:
            trade_records.append({
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "sl": t.sl,
                "tp": t.tp,
                "lot": t.original_lot,
                "remaining_lot": t.lot,
                "pnl": t.pnl,
                "partial_pnl": t.partial_pnl,
                "partial_closed": t.partial_closed,
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "exit_reason": t.exit_reason,
            })

        return {
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "expectancy": np.mean(pnls),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_profit": final_balance - self.initial_balance,
            "net_return_pct": (final_balance - self.initial_balance) / self.initial_balance * 100,
            "max_dd": max_dd,
            "max_dd_pct": max_dd_pct,
            "sharpe": sharpe,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
            "avg_duration_min": np.mean(durations) if durations else 0,
            "partial_tp_count": partial_count,
            "tp_count": tp_count,
            "sl_count": sl_count,
            "be_count": be_count,
            "initial_balance": self.initial_balance,
            "final_balance": final_balance,
            "equity_curve": equity_curve,
            "trades": trade_records,
            "pnls": pnls,
        }


# ────────────────────────────────────────────────────────────
#  Data loaders
# ────────────────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    """Load M1 OHLCV from CSV (columns: time,open,high,low,close,tick_volume)."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            print(f"[ERROR] CSV missing required column: {col}")
            sys.exit(1)
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0
    return df


def load_mt5(symbol: str, bars: int) -> pd.DataFrame:
    """Fetch M1 data from a running MT5 terminal."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("[ERROR] MetaTrader5 package not installed.")
        sys.exit(1)
    if not mt5.initialize(path=config.MT5_PATH):
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        sys.exit(1)
    mt5.login(config.MT5_LOGIN, password=config.MT5_PASSWORD, server=config.MT5_SERVER)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, bars)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        print(f"[ERROR] No data for {symbol} M1 ({bars} bars).")
        sys.exit(1)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df


# ────────────────────────────────────────────────────────────
#  Report printer
# ────────────────────────────────────────────────────────────
def print_report(r: dict, symbol: str) -> None:
    if r["total_trades"] == 0:
        print("\n  ⚠  No trades generated during backtest period.\n")
        return

    print(f"""
════════════════════════════════════════════════════════════
 VALIDUS BACKTEST RESULTS
════════════════════════════════════════════════════════════
 Symbol          : {symbol}
 M1 Bars         : {len(r['equity_curve']):,}
 Initial Balance : ${r['initial_balance']:,.2f}
────────────────────────────────────────────────────────────
 Total Trades    : {r['total_trades']:>8}
   TP Wins       : {r['tp_count']:>8}
   SL Losses     : {r['sl_count']:>8}
   BE Exits      : {r['be_count']:>8}
   Partial TP    : {r['partial_tp_count']:>8}
 Win Rate        : {r['win_rate']:>8.2f}%
 Profit Factor   : {r['profit_factor']:>8.2f}
 Expectancy      :  ${r['expectancy']:>+10.2f} / trade
────────────────────────────────────────────────────────────
 Gross Profit    :  ${r['gross_profit']:>12,.2f}
 Gross Loss      : -${r['gross_loss']:>12,.2f}
 Net Profit      :  ${r['net_profit']:>+12,.2f}
 Net Return      : {r['net_return_pct']:>+8.2f}%
────────────────────────────────────────────────────────────
 Max Drawdown    :  ${r['max_dd']:>10,.2f} ({r['max_dd_pct']:.2f}%)
 Sharpe Ratio    : {r['sharpe']:>8.2f}
 Avg Win         :  ${r['avg_win']:>+10.2f}
 Avg Loss        :  ${r['avg_loss']:>+10.2f}
 Best Trade      :  ${r['best_trade']:>+10.2f}
 Worst Trade     :  ${r['worst_trade']:>+10.2f}
 Avg Duration    : {r['avg_duration_min']:>8.1f} min
════════════════════════════════════════════════════════════""")


def save_results(r: dict, symbol: str) -> tuple[str, str]:
    out_dir = "backtest_results"
    os.makedirs(out_dir, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    trades_path = os.path.join(out_dir, f"trades_{symbol}_{ts}.csv")
    pd.DataFrame(r["trades"]).to_csv(trades_path, index=False)

    eq_path = os.path.join(out_dir, f"equity_{symbol}_{ts}.csv")
    pd.DataFrame({"equity": r["equity_curve"]}).to_csv(eq_path, index=False)

    print(f" Trades saved to : {trades_path}")
    print(f" Equity saved to : {eq_path}")
    return trades_path, eq_path


# ────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VALIDUS — Backtest Engine")
    p.add_argument("--symbol", default="XAUUSD", help="Trading symbol (default: XAUUSD)")
    p.add_argument("--bars", type=int, default=50_000, help="M1 bars to fetch from MT5")
    p.add_argument("--csv", default="", help="Path to M1 CSV file (overrides --bars)")
    p.add_argument("--balance", type=float, default=0, help="Initial balance (0 = use config)")
    p.add_argument("--no-limit", action="store_true", help="Force market orders instead of limit")
    p.add_argument("--no-trail", action="store_true", help="Disable trailing stop")
    p.add_argument("--no-mtf", action="store_true", help="Disable multi-timeframe FVG filter")
    p.add_argument("--no-partial", action="store_true", help="Disable partial take-profit")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Override config for backtest flags
    if args.no_limit:
        config.USE_LIMIT_ORDER = False
    if args.no_trail:
        config.TRAILING_STOP_ENABLED = False
    if args.no_mtf:
        config.MTF_FVG_ENABLED = False
    if args.no_partial:
        config.PARTIAL_TP_ENABLED = False

    print(f"\n{'═' * 60}")
    print(f" VALIDUS BACKTEST — {args.symbol}")
    print(f"{'═' * 60}")

    if args.csv:
        print(f" Loading CSV: {args.csv}")
        df = load_csv(args.csv)
    else:
        print(f" Fetching {args.bars:,} M1 bars from MT5...")
        df = load_mt5(args.symbol, args.bars)

    print(f" Data range: {df.index[0]} → {df.index[-1]}")
    print(f" Total bars: {len(df):,}")
    print(f" Limit Orders: {'ON' if config.USE_LIMIT_ORDER else 'OFF'}")
    print(f" Trailing Stop: {'ON' if config.TRAILING_STOP_ENABLED else 'OFF'}")
    print(f" MTF FVG:       {'ON' if config.MTF_FVG_ENABLED else 'OFF'}")
    print(f" Partial TP:    {'ON' if config.PARTIAL_TP_ENABLED else 'OFF'}")
    print(f"{'─' * 60}")
    print(" Running backtest...\n")

    engine = BacktestEngine(
        df_m1=df,
        symbol=args.symbol,
        initial_balance=args.balance if args.balance > 0 else None,
    )
    results = engine.run()
    print_report(results, args.symbol)

    if results["total_trades"] > 0:
        trades_csv, _ = save_results(results, args.symbol)
        print(f"\n TIP: Run Monte Carlo analysis:")
        print(f"   python montecarlo.py --trades {trades_csv}")
    print()


if __name__ == "__main__":
    main()
