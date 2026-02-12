# ============================================================
#  VALIDUS — Trade Decision Logger
#  Structured CSV logging for post-analysis & review
#  ทุก decision จะถูกเก็บไว้ใน logs/decisions_YYYY-MM-DD.csv
# ============================================================
from __future__ import annotations

import csv
import os
import datetime as dt

import config


_writer = None
_current_date: dt.date | None = None
_file = None

HEADERS = [
    "timestamp", "symbol", "event", "direction",
    "price", "entry", "sl", "tp", "lot", "ticket",
    "pnl", "atr", "bb_exp", "detail",
]


def _ensure_writer() -> None:
    """Open or rotate daily CSV file."""
    global _writer, _current_date, _file

    if not config.DECISION_LOG_ENABLED:
        return

    today = dt.datetime.now(dt.timezone.utc).date()
    if _current_date == today and _writer is not None:
        return

    if _file is not None:
        _file.close()

    log_dir = config.DECISION_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"decisions_{today.isoformat()}.csv")
    is_new = not os.path.exists(path)

    _file = open(path, "a", newline="", encoding="utf-8")
    _writer = csv.writer(_file)
    _current_date = today

    if is_new:
        _writer.writerow(HEADERS)
        _file.flush()


def log_decision(
    symbol: str,
    event: str,
    direction: str = "",
    price: float = 0.0,
    entry: float = 0.0,
    sl: float = 0.0,
    tp: float = 0.0,
    lot: float = 0.0,
    ticket: int = 0,
    pnl: float = 0.0,
    atr: float = 0.0,
    bb_exp: float = 0.0,
    detail: str = "",
) -> None:
    """
    Append one decision row to the daily CSV.

    Event types:
      SCAN           — new M1 bar, starting evaluation
      COND_FAIL      — strategy condition failed (detail = which)
      SIGNAL         — signal generated (all conditions passed)
      LIMIT_PLACED   — limit order placed on MT5
      MARKET_FILL    — market order filled
      LIMIT_EXPIRED  — pending limit order cancelled (timeout)
      BE_MOVE        — SL moved to breakeven
      TRAIL_MOVE     — trailing stop adjusted
      PARTIAL_TP     — partial take-profit closed (backtest only)
      SL_HIT         — stop loss hit (backtest only)
      TP_HIT         — take profit hit (backtest only)
      DD_STOP        — daily drawdown limit triggered
      SKIP_SESSION   — outside trading session hours
      SKIP_NEWS      — news window active
      SKIP_COOLDOWN  — signal cooldown period
      SKIP_MAXPOS    — max positions reached
    """
    if not config.DECISION_LOG_ENABLED:
        return

    _ensure_writer()
    if _writer is None:
        return

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    _writer.writerow([
        now,
        symbol,
        event,
        direction,
        f"{price:.5f}" if price else "",
        f"{entry:.5f}" if entry else "",
        f"{sl:.5f}" if sl else "",
        f"{tp:.5f}" if tp else "",
        f"{lot:.2f}" if lot else "",
        ticket or "",
        f"{pnl:.2f}" if pnl else "",
        f"{atr:.5f}" if atr else "",
        f"{bb_exp:.3f}" if bb_exp else "",
        detail,
    ])
    _file.flush()


def close() -> None:
    """Flush and close the current log file."""
    global _file, _writer, _current_date
    if _file:
        _file.close()
        _file = None
        _writer = None
        _current_date = None
