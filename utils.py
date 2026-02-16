# ============================================================
#  VALIDUS — Utility helpers (logging, notifications, news)
# ============================================================
from __future__ import annotations

import logging
import logging.handlers
import datetime as dt
import requests
import config


# ── Logger Setup ────────────────────────────────────────────
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("VALIDUS")
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Rotating file handler (auto-rotate when file exceeds limit)
    fh = logging.handlers.RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


log = setup_logger()


# ── Line Notify ─────────────────────────────────────────────
def line_notify(message: str) -> None:
    if not config.LINE_NOTIFY_TOKEN:
        return
    try:
        requests.post(
            config.LINE_NOTIFY_URL,
            headers={"Authorization": f"Bearer {config.LINE_NOTIFY_TOKEN}"},
            data={"message": f"\n[VALIDUS] {message}"},
            timeout=5,
        )
    except Exception as exc:
        log.warning("Line Notify failed: %s", exc)


# ── News Filter (Forex Factory) ─────────────────────────────
_news_cache: list[dict] = []
_news_cache_date: dt.date | None = None
_news_fetch_failed_until: float = 0.0  # backoff timestamp


def _fetch_news() -> list[dict]:
    """Fetch today's high-impact events from Forex Factory calendar."""
    global _news_cache, _news_cache_date, _news_fetch_failed_until
    today = dt.date.today()
    if _news_cache_date == today and _news_cache is not None:
        return _news_cache

    # Backoff: don't retry for 300s after a failure
    import time as _time
    if _time.time() < _news_fetch_failed_until:
        return _news_cache

    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        events = resp.json()
        _news_cache = [
            e for e in events
            if e.get("impact", "").lower() == "high"
            and _parse_ff_date(e.get("date", "")) == today
            and e.get("country", "").upper() in config.NEWS_CURRENCIES
        ]
        _news_cache_date = today
        log.info("[NEWS] Fetched %d high-impact events for today.", len(_news_cache))
    except Exception as exc:
        log.warning("News fetch failed: %s — retrying in 5 min.", exc)
        _news_fetch_failed_until = _time.time() + 300  # 5 min backoff
        # Keep existing cache, don't clear
    return _news_cache


def _parse_ff_date(date_str: str) -> dt.date | None:
    """Parse ISO date string from Forex Factory feed."""
    try:
        return dt.datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
    except Exception:
        return None


def is_news_window() -> bool:
    """Return True if we are within NEWS_BUFFER_MIN of a high-impact event."""
    if not config.NEWS_FILTER_ENABLED:
        return False
    now_utc = dt.datetime.now(dt.timezone.utc)
    events = _fetch_news()
    buf = dt.timedelta(minutes=config.NEWS_BUFFER_MIN)
    for ev in events:
        try:
            ev_time = dt.datetime.fromisoformat(
                ev.get("date", "").replace("Z", "+00:00")
            )
            if abs(now_utc - ev_time) <= buf:
                log.info("[NEWS] Window active: %s", ev.get("title", ""))
                return True
        except Exception:
            continue
    return False


# ── Session Filter ──────────────────────────────────────────
def is_in_session() -> bool:
    """Return True if current UTC hour is within the configured trading session."""
    if not config.SESSION_FILTER_ENABLED:
        return True
    hour = dt.datetime.now(dt.timezone.utc).hour
    start = config.SESSION_START_UTC
    end = config.SESSION_END_UTC
    if start <= end:
        return start <= hour < end
    else:
        return hour >= start or hour < end


# ── Misc helpers ─────────────────────────────────────────────
def pts_to_price(symbol_info, points: int) -> float:
    """Convert integer points to price distance."""
    return points * symbol_info.point


def timestamp_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
