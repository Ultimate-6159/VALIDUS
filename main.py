# ============================================================
#  VALIDUS — Main Entry Point
#  Modules: Data Streamer (A), Execution Master (C),
#           Guardian (D), CLI Dashboard
# ============================================================
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import threading
import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import config
import trade_logger
from strategy import Strategy, SweepScalperStrategy, Signal
from utils import log, line_notify, is_news_window, is_in_session


# ════════════════════════════════════════════════════════════
#  Module D — Guardian
# ════════════════════════════════════════════════════════════
class Guardian:
    """Connection watchdog & daily drawdown limiter."""

    def __init__(self):
        self._start_balance: float = 0.0
        self._force_stopped: bool = False
        self._last_reset_date: dt.date | None = None

    # ── MT5 connect / reconnect ─────────────────────────────
    @staticmethod
    def connect() -> bool:
        if mt5.initialize(path=config.MT5_PATH):
            auth = mt5.login(
                config.MT5_LOGIN,
                password=config.MT5_PASSWORD,
                server=config.MT5_SERVER,
            )
            if auth:
                log.info("[OK] MT5 connected -- account %s", config.MT5_LOGIN)
                return True
        err = mt5.last_error()
        log.error("[FAIL] MT5 connection failed: %s", err)
        return False

    def ensure_connection(self) -> bool:
        info = mt5.account_info()
        if info is not None:
            return True
        log.warning("[WARN] MT5 disconnected -- attempting reconnect...")
        for attempt in range(1, 6):
            if self.connect():
                return True
            log.warning("  reconnect attempt %d/5 failed", attempt)
            time.sleep(2)
        log.error("[FAIL] Could not reconnect to MT5.")
        return False

    # ── Daily drawdown check ────────────────────────────────
    def snapshot_balance(self) -> None:
        info = mt5.account_info()
        if info:
            self._start_balance = info.balance
            self._last_reset_date = dt.datetime.now(dt.timezone.utc).date()
            # Auto-apply account tier based on current balance
            tier_name = config.apply_tier(info.balance)
            log.info("[BALANCE] Start-of-day: %.2f", self._start_balance)
            log.info(
                "[TIER] %s | Risk: %.2f%% | MaxPos: %d | DD Limit: %.1f%%",
                tier_name, config.RISK_PCT,
                config.MAX_POSITIONS, config.DAILY_DD_LIMIT_PCT,
            )
            line_notify(
                f"[TIER] {tier_name} | Balance: ${info.balance:,.2f} "
                f"| Risk: {config.RISK_PCT}% | DD Limit: {config.DAILY_DD_LIMIT_PCT}%"
            )

    def check_daily_reset(self) -> None:
        """Re-snapshot balance & tier at DAILY_RESET_HOUR_UTC each day."""
        now_utc = dt.datetime.now(dt.timezone.utc)
        today = now_utc.date()
        if (
            self._last_reset_date != today
            and now_utc.hour >= config.DAILY_RESET_HOUR_UTC
        ):
            log.info("[RESET] Daily reset -- re-snapshot balance & tier.")
            self._force_stopped = False
            self.snapshot_balance()

    def check_drawdown(self) -> bool:
        """Return True if drawdown limit is breached."""
        if self._force_stopped:
            return True
        info = mt5.account_info()
        if info is None or self._start_balance == 0:
            return False
        dd_pct = (self._start_balance - info.equity) / self._start_balance * 100
        if dd_pct >= config.DAILY_DD_LIMIT_PCT:
            log.error(
                "[STOP] Daily DD %.2f%% >= limit %.2f%% -- FORCE STOP",
                dd_pct, config.DAILY_DD_LIMIT_PCT,
            )
            trade_logger.log_decision(
                symbol="ALL", event="DD_STOP",
                pnl=info.equity - self._start_balance,
                detail=f"DD {dd_pct:.2f}% >= limit {config.DAILY_DD_LIMIT_PCT:.1f}%",
            )
            line_notify(
                f"[STOP] Daily drawdown limit hit ({dd_pct:.2f}%). System stopped."
            )
            self._force_stopped = True
            return True
        return False

    @property
    def is_stopped(self) -> bool:
        return self._force_stopped

    def reset(self) -> None:
        self._force_stopped = False


# ════════════════════════════════════════════════════════════
#  Module A — Data Streamer
# ════════════════════════════════════════════════════════════
_TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
}


def fetch_ohlc(symbol: str, timeframe: str, bars: int = 200) -> pd.DataFrame:
    """Fetch OHLC data from MT5 as a pandas DataFrame."""
    tf = _TF_MAP.get(timeframe, mt5.TIMEFRAME_M1)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df


def get_tick(symbol: str):
    """Return latest tick (bid/ask)."""
    return mt5.symbol_info_tick(symbol)


# ════════════════════════════════════════════════════════════
#  Module C — Execution Master
# ════════════════════════════════════════════════════════════
class ExecutionMaster:
    """Handles order execution and position management."""

    # ── Lot calculation ─────────────────────────────────────
    @staticmethod
    def calc_lot(symbol: str, sl_distance: float) -> float:
        if config.RISK_PCT <= 0:
            return config.LOT_SIZE
        info = mt5.account_info()
        sym = mt5.symbol_info(symbol)
        if info is None or sym is None or sl_distance <= 0:
            return config.LOT_SIZE
        risk_money = info.balance * (config.RISK_PCT / 100.0)
        tick_value = sym.trade_tick_value
        tick_size = sym.trade_tick_size
        if tick_value == 0 or tick_size == 0:
            return config.LOT_SIZE
        lot = risk_money / (sl_distance / tick_size * tick_value)
        lot = max(sym.volume_min, min(lot, sym.volume_max))
        lot = round(lot, 2)
        return lot

    # ── ATR cache for trailing stop ───────────────────────
    _atr_cache: dict = {}

    @staticmethod
    def _get_current_atr(symbol: str) -> float:
        """Fetch current M1 ATR for trailing stop, cached for 60 s."""
        now = time.time()
        if symbol in ExecutionMaster._atr_cache:
            val, ts = ExecutionMaster._atr_cache[symbol]
            if now - ts < 60:
                return val
        rates = mt5.copy_rates_from_pos(
            symbol, mt5.TIMEFRAME_M1, 0, config.TRAILING_ATR_PERIOD + 5,
        )
        if rates is None or len(rates) < config.TRAILING_ATR_PERIOD:
            return 0.0
        df = pd.DataFrame(rates)
        prev_c = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_c).abs(),
            (df["low"] - prev_c).abs(),
        ], axis=1).max(axis=1)
        atr_val = float(tr.rolling(config.TRAILING_ATR_PERIOD).mean().iloc[-1])
        if np.isnan(atr_val):
            return 0.0
        ExecutionMaster._atr_cache[symbol] = (atr_val, now)
        return atr_val

    # ── Send order (limit or market) ───────────────────────
    @staticmethod
    def open_order(symbol: str, signal: Signal) -> bool:
        sym = mt5.symbol_info(symbol)
        if sym is None:
            log.error("Symbol %s not found", symbol)
            return False
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        spread = tick.ask - tick.bid
        max_spread = config.MAX_SPREAD_POINTS * sym.point
        if spread > max_spread:
            log.warning("Spread %.1f > max %d pts — skipping", spread / sym.point, config.MAX_SPREAD_POINTS)
            return False

        # ── Decide: limit or market ─────────────────────
        use_limit = config.USE_LIMIT_ORDER
        if use_limit:
            if signal.direction == "BUY" and tick.ask <= signal.entry:
                log.info("[ORDER] Ask %.5f <= entry %.5f — fallback to market.",
                         tick.ask, signal.entry)
                use_limit = False
            elif signal.direction == "SELL" and tick.bid >= signal.entry:
                log.info("[ORDER] Bid %.5f >= entry %.5f — fallback to market.",
                         tick.bid, signal.entry)
                use_limit = False

        if use_limit:
            # ── Limit Order at FVG mid ──────────────────────
            price = signal.entry
            sl_dist = abs(price - signal.sl)
            lot = ExecutionMaster.calc_lot(symbol, sl_dist)
            if signal.direction == "BUY":
                order_type = mt5.ORDER_TYPE_BUY_LIMIT
            else:
                order_type = mt5.ORDER_TYPE_SELL_LIMIT

            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": lot,
                "type": order_type,
                "price": round(price, sym.digits),
                "sl": signal.sl,
                "tp": signal.tp,
                "deviation": 20,
                "magic": config.ORDER_MAGIC,
                "comment": config.ORDER_COMMENT,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            result = mt5.order_send(request)
            if result is None:
                log.error("[LIMIT] order_send returned None -- %s", mt5.last_error())
                return False
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                log.error("[LIMIT] Failed [%d]: %s", result.retcode, result.comment)
                return False
            log.info(
                "[LIMIT] %s %s %.2f lot @ %.5f  SL=%.5f  TP=%.5f  ticket=%d",
                signal.direction, symbol, lot, price,
                signal.sl, signal.tp, result.order,
            )
            trade_logger.log_decision(
                symbol=symbol, event="LIMIT_PLACED", direction=signal.direction,
                price=price, entry=signal.entry, sl=signal.sl, tp=signal.tp,
                lot=lot, ticket=result.order,
            )
            line_notify(
                f"[LIMIT] {signal.direction} {symbol} {lot} lot @ {price:.5f}"
            )
            return True

        # ── Market Order (original logic) ───────────────
        if signal.direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        sl_dist = abs(price - signal.sl)
        lot = ExecutionMaster.calc_lot(symbol, sl_dist)

        if signal.direction == "BUY":
            actual_tp = round(price + sl_dist * config.RISK_REWARD_RATIO, 5)
        else:
            actual_tp = round(price - sl_dist * config.RISK_REWARD_RATIO, 5)
        log.info(
            "[ORDER] TP recalc: signal_tp=%.5f -> actual_tp=%.5f (fill=%.5f SL=%.5f)",
            signal.tp, actual_tp, price, signal.sl,
        )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": signal.sl,
            "tp": actual_tp,
            "deviation": 20,
            "magic": config.ORDER_MAGIC,
            "comment": config.ORDER_COMMENT,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            log.error("[ORDER] order_send returned None -- %s", mt5.last_error())
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("[ORDER] Failed [%d]: %s", result.retcode, result.comment)
            return False

        log.info(
            "[OPEN] %s %s %.2f lot @ %.5f  SL=%.5f  TP=%.5f  ticket=%d",
            signal.direction, symbol, lot, price,
            signal.sl, actual_tp, result.order,
        )
        trade_logger.log_decision(
            symbol=symbol, event="MARKET_FILL", direction=signal.direction,
            price=price, entry=signal.entry, sl=signal.sl, tp=actual_tp,
            lot=lot, ticket=result.order,
        )
        line_notify(
            f"[OPEN] {signal.direction} {symbol} {lot} lot @ {price:.5f}"
        )
        return True

    # ── Breakeven management ────────────────────────────────
    @staticmethod
    def manage_positions() -> None:
        """Move SL to breakeven, then trail using ATR."""
        positions = mt5.positions_get()
        if positions is None:
            return
        for pos in positions:
            if pos.magic != config.ORDER_MAGIC:
                continue

            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                continue

            sym_info = mt5.symbol_info(pos.symbol)
            if sym_info is None:
                continue

            # Calculate distances
            if pos.type == mt5.ORDER_TYPE_BUY:
                current_profit_dist = tick.bid - pos.price_open
                tp_dist = pos.tp - pos.price_open if pos.tp > 0 else 0
                be_price = pos.price_open + sym_info.spread * sym_info.point
            else:
                current_profit_dist = pos.price_open - tick.ask
                tp_dist = pos.price_open - pos.tp if pos.tp > 0 else 0
                be_price = pos.price_open - sym_info.spread * sym_info.point

            if tp_dist <= 0:
                continue

            # ── Phase 1: Breakeven ────────────────────────────
            if current_profit_dist >= tp_dist * config.BREAKEVEN_PCT:
                should_move = False
                if pos.type == mt5.ORDER_TYPE_BUY and pos.sl < be_price:
                    should_move = True
                elif pos.type == mt5.ORDER_TYPE_SELL and (pos.sl > be_price or pos.sl == 0):
                    should_move = True

                if should_move:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "symbol": pos.symbol,
                        "sl": round(be_price, 5),
                        "tp": pos.tp,
                        "magic": config.ORDER_MAGIC,
                    }
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        log.info(
                            "[BE] SL moved ticket=%d  new SL=%.5f",
                            pos.ticket, be_price,
                        )
                        trade_logger.log_decision(
                            symbol=pos.symbol, event="BE_MOVE",
                            price=tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask,
                            sl=be_price, ticket=pos.ticket,
                            detail=f"profit_dist={current_profit_dist:.5f}",
                        )
                    else:
                        log.warning(
                            "BE move failed ticket=%d: %s",
                            pos.ticket,
                            result.comment if result else mt5.last_error(),
                        )

            # ── Phase 2: Trailing Stop (ATR-based) ────────────
            if config.TRAILING_STOP_ENABLED:
                sl_past_be = False
                if pos.type == mt5.ORDER_TYPE_BUY and pos.sl >= be_price:
                    sl_past_be = True
                elif pos.type == mt5.ORDER_TYPE_SELL and 0 < pos.sl <= be_price:
                    sl_past_be = True

                if sl_past_be:
                    atr_val = ExecutionMaster._get_current_atr(pos.symbol)
                    if atr_val > 0:
                        trail_dist = atr_val * config.TRAILING_ATR_MULT
                        new_sl = None
                        if pos.type == mt5.ORDER_TYPE_BUY:
                            candidate = round(tick.bid - trail_dist, 5)
                            if candidate > pos.sl:
                                new_sl = candidate
                        else:
                            candidate = round(tick.ask + trail_dist, 5)
                            if candidate < pos.sl:
                                new_sl = candidate

                        if new_sl is not None:
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos.ticket,
                                "symbol": pos.symbol,
                                "sl": new_sl,
                                "tp": pos.tp,
                                "magic": config.ORDER_MAGIC,
                            }
                            result = mt5.order_send(request)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                log.info(
                                    "[TRAIL] SL trailed ticket=%d  new SL=%.5f (ATR=%.5f)",
                                    pos.ticket, new_sl, atr_val,
                                )
                                trade_logger.log_decision(
                                    symbol=pos.symbol, event="TRAIL_MOVE",
                                    sl=new_sl, atr=atr_val, ticket=pos.ticket,
                                    detail=f"old_sl={pos.sl:.5f} trail_dist={trail_dist:.5f}",
                                )
                            else:
                                log.warning(
                                    "[TRAIL] Failed ticket=%d: %s",
                                    pos.ticket,
                                    result.comment if result else mt5.last_error(),
                                )

    # ── Manage pending limit orders ─────────────────────
    @staticmethod
    def manage_pending_orders() -> None:
        """Cancel expired pending limit orders."""
        if not config.USE_LIMIT_ORDER:
            return
        orders = mt5.orders_get()
        if orders is None:
            return
        now = time.time()
        for order in orders:
            if order.magic != config.ORDER_MAGIC:
                continue
            elapsed = now - order.time_setup
            if elapsed > config.LIMIT_ORDER_EXPIRY_SEC:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    log.info(
                        "[LIMIT] Expired order cancelled: ticket=%d (%.0fs old)",
                        order.ticket, elapsed,
                    )
                    trade_logger.log_decision(
                        symbol=order.symbol, event="LIMIT_EXPIRED",
                        ticket=order.ticket,
                        detail=f"elapsed={elapsed:.0f}s limit={config.LIMIT_ORDER_EXPIRY_SEC}s",
                    )
                else:
                    log.warning(
                        "[LIMIT] Cancel failed ticket=%d: %s",
                        order.ticket,
                        result.comment if result else mt5.last_error(),
                    )

    # ── Panic close all ─────────────────────────────────────
    @staticmethod
    def panic_close() -> int:
        """Close ALL open positions immediately. Returns count closed."""
        positions = mt5.positions_get()
        if not positions:
            return 0
        closed = 0
        for pos in positions:
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                continue
            close_type = (
                mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY
                else mt5.ORDER_TYPE_BUY
            )
            price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "price": price,
                "deviation": 30,
                "magic": config.ORDER_MAGIC,
                "comment": "VALIDUS_PANIC",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1
                log.info("[PANIC] Closed ticket=%d", pos.ticket)
        return closed


# ════════════════════════════════════════════════════════════
#  CLI Dashboard
# ════════════════════════════════════════════════════════════
class Dashboard:
    """Lightweight CLI dashboard with hotkey support."""

    def __init__(self):
        self.status = "STOPPED"
        self._pnl_today: float = 0.0
        # Live activity fields (updated by ValidusBot)
        self.total_scans: int = 0
        self.last_scan_time: str = "---"
        self.last_scan_symbol: str = "---"
        self.last_reject: str = "---"
        self.session_active: bool = False
        self.signals_today: int = 0

    def render(self) -> str:
        positions = mt5.positions_get() or []
        my_pos = [p for p in positions if p.magic == config.ORDER_MAGIC]
        self._pnl_today = sum(p.profit for p in my_pos)

        acct = mt5.account_info()
        balance = acct.balance if acct else 0
        equity = acct.equity if acct else 0
        tier_name = config.get_tier_name(balance) if balance > 0 else "---"

        session_str = "IN SESSION" if self.session_active else "OUT OF SESSION"
        now_str = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S UTC")

        lines = [
            "",
            "+----------------------------------------------+",
            "|           V A L I D U S   v1.0               |",
            "|       Smart Money Sniper - MT5               |",
            "+----------------------------------------------+",
            f"|  Strategy: {config.STRATEGY_MODE:<34}|",
            f"|  Status  : {self.status:<34}|",
            f"|  Tier    : {tier_name:<8}  Risk: {config.RISK_PCT:.2f}%  DD: {config.DAILY_DD_LIMIT_PCT:.1f}%    |",
            f"|  Balance : {balance:>12,.2f}                       |",
            f"|  Equity  : {equity:>12,.2f}                       |",
            f"|  PnL     : {self._pnl_today:>+12,.2f}                       |",
            f"|  Positions: {len(my_pos)}/{config.MAX_POSITIONS:<31}|",
            "+---------- Activity -------------------------+",
            f"|  Clock   : {now_str:<34}|",
            f"|  Session : {session_str:<34}|",
            f"|  Scans   : {self.total_scans:<8} Signals: {self.signals_today:<16}|",
            f"|  Last scan: {self.last_scan_symbol:<6} {self.last_scan_time:<24}|",
            f"|  Reject  : {self.last_reject:<34}|",
            "+----------------------------------------------+",
        ]
        for p in my_pos:
            side = "BUY " if p.type == 0 else "SELL"
            lines.append(
                f"|  {side} {p.symbol:<10} {p.volume:.2f}  "
                f"PnL {p.profit:>+8.2f}  tk#{p.ticket:<10}|"
            )
        if not my_pos:
            lines.append("|  (no active positions)                       |")
        lines += [
            "+----------------------------------------------+",
            "|  [S] Start   [Q] Stop   [P] Panic Close     |",
            "+----------------------------------------------+",
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════
#  Main async loop
# ════════════════════════════════════════════════════════════
class ValidusBot:
    def __init__(self):
        self.guardian = Guardian()
        self.executor = ExecutionMaster()
        if config.STRATEGY_MODE == "SWEEP":
            self.strategy = SweepScalperStrategy()
        else:
            self.strategy = Strategy()
        self.dashboard = Dashboard()
        self._running = False
        self._last_m1_time: dict[str, dt.datetime] = {}
        self._last_signal_time: dict[str, dt.datetime] = {}
        self._heartbeat_interval = 60   # log heartbeat every 60 seconds
        self._last_heartbeat: float = 0
        self._scan_count: int = 0
        self._total_scans: int = 0
        self._signals_today: int = 0
        self._loop_started_logged: bool = False

    # ── Start ───────────────────────────────────────────────
    async def start(self) -> None:
        if not self.guardian.connect():
            log.error("Cannot start — MT5 connection failed.")
            return
        self.guardian.snapshot_balance()
        self._running = True
        self.dashboard.status = "RUNNING"
        log.info("[START] VALIDUS started.")
        line_notify("[START] VALIDUS system started.")
        await self._main_loop()

    # ── Stop ────────────────────────────────────────────────
    def stop(self) -> None:
        self._running = False
        self.dashboard.status = "STOPPED"
        log.info("[STOP] VALIDUS stopped.")
        line_notify("[STOP] VALIDUS system stopped.")

    # ── Main loop ───────────────────────────────────────────
    async def _main_loop(self) -> None:
        self._last_heartbeat = time.time()
        while self._running:
            try:
                # Log once that the loop is alive
                if not self._loop_started_logged:
                    log.info("[LOOP] Main loop running. Symbols: %s | Checking every %.1fs",
                             config.SYMBOLS, config.POSITION_CHECK_SEC)
                    self._loop_started_logged = True

                # Guardian checks
                if not self.guardian.ensure_connection():
                    await asyncio.sleep(5)
                    continue
                self.guardian.check_daily_reset()
                if self.guardian.check_drawdown():
                    self.stop()
                    break

                # Position management (every tick)
                self.executor.manage_positions()
                self.executor.manage_pending_orders()

                # Signal evaluation per symbol
                for symbol in config.SYMBOLS:
                    await self._process_symbol(symbol)

                # Heartbeat log (every 60 seconds)
                now = time.time()
                if now - self._last_heartbeat >= self._heartbeat_interval:
                    acct = mt5.account_info()
                    eq = acct.equity if acct else 0
                    positions = mt5.positions_get() or []
                    my_pos = [p for p in positions if p.magic == config.ORDER_MAGIC]
                    pnl = sum(p.profit for p in my_pos)
                    log.info(
                        "[HEARTBEAT] Alive | Equity: %.2f | Positions: %d | PnL: %+.2f | Scans: %d",
                        eq, len(my_pos), pnl, self._scan_count,
                    )
                    self._scan_count = 0
                    self._last_heartbeat = now

                # Dashboard refresh
                if not config.HEADLESS:
                    if os.name == "nt":
                        os.system("cls")
                    else:
                        sys.stdout.write("\033[2J\033[H")
                    print(self.dashboard.render())

            except Exception as exc:
                log.exception("Main loop error: %s", exc)

            await asyncio.sleep(config.POSITION_CHECK_SEC)

    # ── Per-symbol processing ───────────────────────────────
    async def _process_symbol(self, symbol: str) -> None:
        df_m1 = fetch_ohlc(symbol, config.TIMEFRAME_ENTRY, bars=200)
        if df_m1.empty:
            log.warning("[%s] M1 data empty -- check Market Watch or symbol name.", symbol)
            return

        # Only evaluate when a new M1 bar closes
        last_time = df_m1.index[-1]
        if self._last_m1_time.get(symbol) == last_time:
            return  # same bar, waiting for next M1 close (normal)
        self._last_m1_time[symbol] = last_time
        self._scan_count += 1
        self._total_scans += 1

        tick = mt5.symbol_info_tick(symbol)
        bid = tick.bid if tick else 0
        log.info("[%s] New M1 bar %s | Bid: %.2f -- scanning...", symbol, last_time, bid)

        # Update dashboard activity
        self.dashboard.total_scans = self._total_scans
        self.dashboard.last_scan_time = last_time.strftime("%H:%M:%S")
        self.dashboard.last_scan_symbol = symbol

        # Session filter
        self.dashboard.session_active = is_in_session()
        if not self.dashboard.session_active:
            self.dashboard.last_reject = f"{symbol}: OUT_OF_SESSION"
            log.info("[%s] Outside session (%02d:00-%02d:00 UTC) -- skipping.",
                     symbol, config.SESSION_START_UTC, config.SESSION_END_UTC)
            trade_logger.log_decision(
                symbol=symbol, event="SKIP_SESSION", price=bid,
                detail=f"hour outside {config.SESSION_START_UTC}-{config.SESSION_END_UTC} UTC",
            )
            return

        # News filter
        if is_news_window():
            self.dashboard.last_reject = f"{symbol}: NEWS_WINDOW"
            log.info("[%s] News window active -- skipping.", symbol)
            trade_logger.log_decision(
                symbol=symbol, event="SKIP_NEWS", price=bid,
                detail=f"buffer={config.NEWS_BUFFER_MIN}min",
            )
            return

        # Signal cooldown
        last_sig_time = self._last_signal_time.get(symbol)
        if last_sig_time is not None:
            bars_elapsed = (last_time - last_sig_time).total_seconds() / 60
            if bars_elapsed < config.SIGNAL_COOLDOWN_BARS:
                self.dashboard.last_reject = f"{symbol}: COOLDOWN {bars_elapsed:.0f}/{config.SIGNAL_COOLDOWN_BARS}"
                log.info("[%s] Signal cooldown: %.0f/%d bars -- skipping.",
                         symbol, bars_elapsed, config.SIGNAL_COOLDOWN_BARS)
                trade_logger.log_decision(
                    symbol=symbol, event="SKIP_COOLDOWN", price=bid,
                    detail=f"bars={bars_elapsed:.0f}/{config.SIGNAL_COOLDOWN_BARS}",
                )
                return

        # Check max positions (include pending orders)
        positions = mt5.positions_get(symbol=symbol) or []
        my_pos = [p for p in positions if p.magic == config.ORDER_MAGIC]
        pending = mt5.orders_get(symbol=symbol) or []
        my_pending = [o for o in pending if o.magic == config.ORDER_MAGIC]
        if len(my_pos) + len(my_pending) >= config.MAX_POSITIONS:
            self.dashboard.last_reject = f"{symbol}: MAX_POS ({len(my_pos)}+{len(my_pending)})"
            log.info("[%s] Max positions (%d) reached -- skipping.", symbol, config.MAX_POSITIONS)
            trade_logger.log_decision(
                symbol=symbol, event="SKIP_MAXPOS", price=bid,
                detail=f"pos={len(my_pos)} pending={len(my_pending)} max={config.MAX_POSITIONS}",
            )
            return

        df_m5 = fetch_ohlc(symbol, config.TIMEFRAME_HTF, bars=200)
        signal = self.strategy.evaluate(df_m1, df_m5)
        ev = self.strategy.last_eval
        if signal is not None:
            self._signals_today += 1
            self.dashboard.signals_today = self._signals_today
            log.info("[%s] >> SIGNAL FOUND: %s", symbol, signal)
            trade_logger.log_decision(
                symbol=symbol, event="SIGNAL", direction=signal.direction,
                price=bid, entry=signal.entry, sl=signal.sl, tp=signal.tp,
                atr=ev.get("atr", 0), bb_exp=ev.get("bb_exp", 0),
                detail=ev.get("detail", ""),
            )
            if self.executor.open_order(symbol, signal):
                self._last_signal_time[symbol] = last_time
                log.info("[%s] Signal cooldown set for %d bars.",
                         symbol, config.SIGNAL_COOLDOWN_BARS)
        else:
            fail_reason = ev.get('fail', '?')
            self.dashboard.last_reject = f"{symbol}: {fail_reason}"
            trade_logger.log_decision(
                symbol=symbol, event="COND_FAIL", price=bid,
                atr=ev.get("atr", 0), bb_exp=ev.get("bb_exp", 0),
                detail=f"{fail_reason}: {ev.get('detail', '')}",
            )
            log.info("[%s] No signal (conditions not met).", symbol)


# ════════════════════════════════════════════════════════════
#  Hotkey listener (runs in background thread)
# ════════════════════════════════════════════════════════════
def hotkey_listener(bot: ValidusBot, loop: asyncio.AbstractEventLoop) -> None:
    """Listen for single-key commands on stdin."""
    try:
        import msvcrt  # Windows only
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode("utf-8", errors="ignore").upper()
                if key == "S" and not bot._running:
                    asyncio.run_coroutine_threadsafe(bot.start(), loop)
                elif key == "Q":
                    bot.stop()
                elif key == "P":
                    count = ExecutionMaster.panic_close()
                    log.info("[PANIC] %d positions closed.", count)
                    line_notify(f"[PANIC] {count} positions closed.")
            time.sleep(0.1)
    except ImportError:
        # Fallback for non-Windows (input based)
        while True:
            key = input().strip().upper()
            if key == "S" and not bot._running:
                asyncio.run_coroutine_threadsafe(bot.start(), loop)
            elif key == "Q":
                bot.stop()
            elif key == "P":
                count = ExecutionMaster.panic_close()
                log.info("[PANIC] %d positions closed.", count)


# ════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VALIDUS — Smart Money Sniper MT5")
    parser.add_argument(
        "--autostart", action="store_true", default=None,
        help="Start trading immediately without waiting for [S] key",
    )
    parser.add_argument(
        "--headless", action="store_true", default=None,
        help="Disable dashboard clearing (better for log files / VPS)",
    )
    return parser.parse_args()


def main() -> None:
    # ── CLI overrides ───────────────────────────────────────
    args = parse_args()
    if args.autostart is not None:
        config.AUTO_START = args.autostart
    if args.headless is not None:
        config.HEADLESS = args.headless

    # Ensure working directory is script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print(
        "\n"
        "+----------------------------------------------+\n"
        "|           V A L I D U S   v1.0               |\n"
        "|       Smart Money Sniper - MT5               |\n"
        "+----------------------------------------------+\n"
        "|  Mode: {:<38}|\n"
        "+----------------------------------------------+\n".format(
            "AUTO-START (VPS)" if config.AUTO_START else "[S] Start  [Q] Quit"
        )
    )

    bot = ValidusBot()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Start hotkey thread (even in autostart — allows [Q] & [P])
    hk_thread = threading.Thread(
        target=hotkey_listener, args=(bot, loop), daemon=True
    )
    hk_thread.start()

    try:
        if config.AUTO_START:
            loop.run_until_complete(bot.start())
        else:
            # Wait for [S] key from hotkey_listener
            while not bot._running:
                time.sleep(0.2)
            loop.run_until_complete(bot._main_loop())
    except KeyboardInterrupt:
        bot.stop()
    finally:
        trade_logger.close()
        mt5.shutdown()
        loop.close()
        log.info("System shutdown complete.")


if __name__ == "__main__":
    main()
