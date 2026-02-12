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
from strategy import Strategy, Signal
from utils import log, line_notify, is_news_window


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
                log.info("✅ MT5 connected – account %s", config.MT5_LOGIN)
                return True
        err = mt5.last_error()
        log.error("❌ MT5 connection failed: %s", err)
        return False

    def ensure_connection(self) -> bool:
        info = mt5.account_info()
        if info is not None:
            return True
        log.warning("⚠ MT5 disconnected — attempting reconnect …")
        for attempt in range(1, 6):
            if self.connect():
                return True
            log.warning("  reconnect attempt %d/5 failed", attempt)
            time.sleep(2)
        log.error("❌ Could not reconnect to MT5.")
        return False

    # ── Daily drawdown check ────────────────────────────────
    def snapshot_balance(self) -> None:
        info = mt5.account_info()
        if info:
            self._start_balance = info.balance
            self._last_reset_date = dt.datetime.now(dt.timezone.utc).date()
            # Auto-apply account tier based on current balance
            tier_name = config.apply_tier(info.balance)
            log.info("📊 Start-of-day balance: %.2f", self._start_balance)
            log.info(
                "🏷  Tier: %s | Risk: %.2f%% | MaxPos: %d | DD Limit: %.1f%%",
                tier_name, config.RISK_PCT,
                config.MAX_POSITIONS, config.DAILY_DD_LIMIT_PCT,
            )
            line_notify(
                f"🏷 Tier: {tier_name} | Balance: ${info.balance:,.2f} "
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
            log.info("🔄 Daily reset — re-snapshot balance & tier.")
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
                "🛑 Daily DD %.2f%% >= limit %.2f%% — FORCE STOP",
                dd_pct, config.DAILY_DD_LIMIT_PCT,
            )
            line_notify(
                f"🛑 Daily drawdown limit hit ({dd_pct:.2f}%). System stopped."
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

    # ── Send market order ───────────────────────────────────
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

        if signal.direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        sl_dist = abs(price - signal.sl)
        lot = ExecutionMaster.calc_lot(symbol, sl_dist)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": signal.sl,
            "tp": signal.tp,
            "deviation": 20,
            "magic": config.ORDER_MAGIC,
            "comment": config.ORDER_COMMENT,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            log.error("order_send returned None – %s", mt5.last_error())
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("Order failed [%d]: %s", result.retcode, result.comment)
            return False

        log.info(
            "📈 Opened %s %s %.2f lot @ %.5f  SL=%.5f  TP=%.5f  ticket=%d",
            signal.direction, symbol, lot, price,
            signal.sl, signal.tp, result.order,
        )
        line_notify(
            f"📈 {signal.direction} {symbol} {lot} lot @ {price:.5f}"
        )
        return True

    # ── Breakeven management ────────────────────────────────
    @staticmethod
    def manage_positions() -> None:
        """Move SL to breakeven when profit >= BREAKEVEN_PCT of TP distance."""
        positions = mt5.positions_get()
        if positions is None:
            return
        for pos in positions:
            if pos.magic != config.ORDER_MAGIC:
                continue

            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                continue

            # Calculate distances
            if pos.type == mt5.ORDER_TYPE_BUY:
                current_profit_dist = tick.bid - pos.price_open
                tp_dist = pos.tp - pos.price_open if pos.tp > 0 else 0
                be_price = pos.price_open + mt5.symbol_info(pos.symbol).spread * mt5.symbol_info(pos.symbol).point
            else:
                current_profit_dist = pos.price_open - tick.ask
                tp_dist = pos.price_open - pos.tp if pos.tp > 0 else 0
                be_price = pos.price_open - mt5.symbol_info(pos.symbol).spread * mt5.symbol_info(pos.symbol).point

            if tp_dist <= 0:
                continue

            # Check if profit reached breakeven threshold
            if current_profit_dist >= tp_dist * config.BREAKEVEN_PCT:
                # Only move if SL is not already at or beyond breakeven
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
                            "🔒 BE moved ticket=%d  new SL=%.5f",
                            pos.ticket, be_price,
                        )
                    else:
                        log.warning(
                            "BE move failed ticket=%d: %s",
                            pos.ticket,
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
                log.info("🚨 Panic closed ticket=%d", pos.ticket)
        return closed


# ════════════════════════════════════════════════════════════
#  CLI Dashboard
# ════════════════════════════════════════════════════════════
class Dashboard:
    """Lightweight CLI dashboard with hotkey support."""

    def __init__(self):
        self.status = "STOPPED"
        self._pnl_today: float = 0.0

    def render(self) -> str:
        positions = mt5.positions_get() or []
        my_pos = [p for p in positions if p.magic == config.ORDER_MAGIC]
        self._pnl_today = sum(p.profit for p in my_pos)

        acct = mt5.account_info()
        balance = acct.balance if acct else 0
        equity = acct.equity if acct else 0
        tier_name = config.get_tier_name(balance) if balance > 0 else "---"

        lines = [
            "",
            "╔══════════════════════════════════════════════╗",
            "║           V A L I D U S   v1.0               ║",
            "║       Smart Money Sniper — MT5                ║",
            "╠══════════════════════════════════════════════╣",
            f"║  Status  : {self.status:<34}║",
            f"║  Tier    : {tier_name:<8}  Risk: {config.RISK_PCT:.2f}%  DD: {config.DAILY_DD_LIMIT_PCT:.1f}%    ║",
            f"║  Balance : {balance:>12,.2f}                       ║",
            f"║  Equity  : {equity:>12,.2f}                       ║",
            f"║  PnL     : {self._pnl_today:>+12,.2f}                       ║",
            f"║  Positions: {len(my_pos)}/{config.MAX_POSITIONS:<31}║",
            "╠══════════════════════════════════════════════╣",
        ]
        for p in my_pos:
            side = "BUY " if p.type == 0 else "SELL"
            lines.append(
                f"║  {side} {p.symbol:<10} {p.volume:.2f}  "
                f"PnL {p.profit:>+8.2f}  tk#{p.ticket:<10}║"
            )
        if not my_pos:
            lines.append("║  (no active positions)                       ║")
        lines += [
            "╠══════════════════════════════════════════════╣",
            "║  [S] Start   [Q] Stop   [P] Panic Close     ║",
            "╚══════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════
#  Main async loop
# ════════════════════════════════════════════════════════════
class ValidusBot:
    def __init__(self):
        self.guardian = Guardian()
        self.executor = ExecutionMaster()
        self.strategy = Strategy()
        self.dashboard = Dashboard()
        self._running = False
        self._last_m1_time: dict[str, dt.datetime] = {}

    # ── Start ───────────────────────────────────────────────
    async def start(self) -> None:
        if not self.guardian.connect():
            log.error("Cannot start — MT5 connection failed.")
            return
        self.guardian.snapshot_balance()
        self._running = True
        self.dashboard.status = "🟢 RUNNING"
        log.info("🚀 VALIDUS started.")
        line_notify("🚀 System started.")
        await self._main_loop()

    # ── Stop ────────────────────────────────────────────────
    def stop(self) -> None:
        self._running = False
        self.dashboard.status = "🔴 STOPPED"
        log.info("⏹ VALIDUS stopped.")
        line_notify("⏹ System stopped.")

    # ── Main loop ───────────────────────────────────────────
    async def _main_loop(self) -> None:
        while self._running:
            try:
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

                # Signal evaluation per symbol
                for symbol in config.SYMBOLS:
                    await self._process_symbol(symbol)

                # Dashboard refresh
                if not config.HEADLESS:
                    sys.stdout.write("\033[2J\033[H")  # clear terminal
                    print(self.dashboard.render())

            except Exception as exc:
                log.exception("Main loop error: %s", exc)

            await asyncio.sleep(config.POSITION_CHECK_SEC)

    # ── Per-symbol processing ───────────────────────────────
    async def _process_symbol(self, symbol: str) -> None:
        df_m1 = fetch_ohlc(symbol, config.TIMEFRAME_ENTRY, bars=200)
        if df_m1.empty:
            return

        # Only evaluate when a new M1 bar closes
        last_time = df_m1.index[-1]
        if self._last_m1_time.get(symbol) == last_time:
            return
        self._last_m1_time[symbol] = last_time

        # News filter
        if is_news_window():
            log.info("📰 News window — skipping %s", symbol)
            return

        # Check max positions
        positions = mt5.positions_get(symbol=symbol) or []
        my_pos = [p for p in positions if p.magic == config.ORDER_MAGIC]
        if len(my_pos) >= config.MAX_POSITIONS:
            return

        df_m5 = fetch_ohlc(symbol, config.TIMEFRAME_HTF, bars=200)
        signal = self.strategy.evaluate(df_m1, df_m5)
        if signal is not None:
            self.executor.open_order(symbol, signal)


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
                    log.info("🚨 Panic close: %d positions closed.", count)
                    line_notify(f"🚨 Panic close: {count} positions closed.")
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
                log.info("🚨 Panic close: %d positions closed.", count)


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
        "╔══════════════════════════════════════════════╗\n"
        "║           V A L I D U S   v1.0               ║\n"
        "║       Smart Money Sniper — MT5                ║\n"
        "╠══════════════════════════════════════════════╣\n"
        "║  Mode: {:<38}║\n"
        "╚══════════════════════════════════════════════╝\n".format(
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
        mt5.shutdown()
        loop.close()
        log.info("System shutdown complete.")


if __name__ == "__main__":
    main()
