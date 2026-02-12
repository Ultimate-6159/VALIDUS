# ============================================================
#  VALIDUS — Configuration
#  Adaptive for account sizes $500 – $5,000,000
# ============================================================

# ── MT5 Connection ──────────────────────────────────────────
MT5_LOGIN    = 0            # MT5 account number
MT5_PASSWORD = ""           # MT5 account password
MT5_SERVER   = ""           # e.g. "Exness-MT5Real"
MT5_PATH     = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# ── Trading Pairs ───────────────────────────────────────────
SYMBOLS = [
    "XAUUSD",
    # "EURUSD",
    # "GBPUSD",
]

# ── Timeframes ──────────────────────────────────────────────
TIMEFRAME_ENTRY  = "M1"     # Entry signal timeframe
TIMEFRAME_HTF    = "M5"     # Higher-timeframe confirmation

# ════════════════════════════════════════════════════════════
#  Account Tier System (Auto-Adaptive)
# ════════════════════════════════════════════════════════════
#  ระบบจะ detect balance แล้วเลือก tier อัตโนมัติตอน start
#  หรือจะ force tier ด้วย FORCE_TIER = "MICRO" / "SMALL" / ...
#  ถ้าไม่ force → ระบบเลือกเอง
# ────────────────────────────────────────────────────────────
FORCE_TIER = ""  # "" = auto-detect | "MICRO" | "SMALL" | "MEDIUM" | "LARGE" | "WHALE"

ACCOUNT_TIERS = {
    # ── MICRO : $500 – $2,000 ───────────────────────────────
    #   พอร์ตเล็ก ต้องระวังสูงสุด เปิดทีละ 1 ออเดอร์
    #   Lot เล็ก แต่ Risk % พอเหมาะเพื่อให้พอร์ตโตได้
    "MICRO": {
        "min_balance":        500,
        "max_balance":      2_000,
        "risk_pct":           1.0,      # 1% ต่อไม้ ($5–$20 ต่อเทรด)
        "lot_size":          0.01,      # fallback lot
        "max_positions":        1,      # เปิดทีละ 1 เท่านั้น — ห้ามเสี่ยง
        "daily_dd_limit_pct":  3.0,     # หยุดเมื่อเสีย 3% ($15–$60)
        "risk_reward_ratio":   1.5,     # RR 1:1.5
        "breakeven_pct":      0.50,     # เลื่อน BE ที่ 50% ของ TP
        "max_spread_points":    35,     # พอร์ตเล็กมักใช้ broker spread สูง
    },
    # ── SMALL : $2,001 – $10,000 ────────────────────────────
    #   เริ่มมี room เปิดได้ 2 ออเดอร์ ยังคง conservative
    "SMALL": {
        "min_balance":      2_001,
        "max_balance":     10_000,
        "risk_pct":           1.0,      # 1% ต่อไม้ ($20–$100)
        "lot_size":          0.02,
        "max_positions":        2,
        "daily_dd_limit_pct":  3.0,     # หยุดเมื่อเสีย 3% ($60–$300)
        "risk_reward_ratio":   1.5,
        "breakeven_pct":      0.50,
        "max_spread_points":    30,
    },
    # ── MEDIUM : $10,001 – $100,000 ─────────────────────────
    #   พอร์ตกลาง ลด risk % เพื่อความเสถียร
    "MEDIUM": {
        "min_balance":     10_001,
        "max_balance":    100_000,
        "risk_pct":          0.75,      # 0.75% ต่อไม้ ($75–$750)
        "lot_size":          0.05,
        "max_positions":        3,
        "daily_dd_limit_pct":  2.5,     # หยุดเมื่อเสีย 2.5%
        "risk_reward_ratio":   1.5,
        "breakeven_pct":      0.50,
        "max_spread_points":    25,
    },
    # ── LARGE : $100,001 – $1,000,000 ───────────────────────
    #   พอร์ตใหญ่ เน้นรักษาทุน ลด risk, DD limit เข้ม
    "LARGE": {
        "min_balance":    100_001,
        "max_balance":  1_000_000,
        "risk_pct":          0.50,      # 0.5% ต่อไม้ ($500–$5,000)
        "lot_size":          0.10,
        "max_positions":        3,
        "daily_dd_limit_pct":  2.0,     # หยุดเมื่อเสีย 2%
        "risk_reward_ratio":   1.5,
        "breakeven_pct":      0.45,     # เลื่อน BE เร็วขึ้นเล็กน้อย
        "max_spread_points":    20,     # เข้มเรื่อง spread (ควรใช้ ECN)
    },
    # ── WHALE : $1,000,001 – $5,000,000 ─────────────────────
    #   Ultra-conservative ระวัง slippage & liquidity
    #   ลด max_positions เพราะ lot ใหญ่จะกระทบ market
    "WHALE": {
        "min_balance":  1_000_001,
        "max_balance":  5_000_000,
        "risk_pct":          0.25,      # 0.25% ต่อไม้ ($2,500–$12,500)
        "lot_size":          0.50,
        "max_positions":        2,      # ลดจำนวน เพราะ lot ใหญ่ ≈ slippage
        "daily_dd_limit_pct":  1.5,     # หยุดเมื่อเสีย 1.5%
        "risk_reward_ratio":   1.5,
        "breakeven_pct":      0.40,     # เลื่อน BE เร็ว ปกป้องทุน
        "max_spread_points":    15,     # ECN/Raw spread เท่านั้น
    },
}


# ════════════════════════════════════════════════════════════
#  Tier auto-detection helper
# ════════════════════════════════════════════════════════════
def get_tier(balance: float) -> dict:
    """Return the tier config dict that matches the given balance."""
    if FORCE_TIER and FORCE_TIER in ACCOUNT_TIERS:
        return ACCOUNT_TIERS[FORCE_TIER]
    for tier in ACCOUNT_TIERS.values():
        if tier["min_balance"] <= balance <= tier["max_balance"]:
            return tier
    # Fallback: if balance < 500 → MICRO, if > 5M → WHALE
    if balance < 500:
        return ACCOUNT_TIERS["MICRO"]
    return ACCOUNT_TIERS["WHALE"]


def get_tier_name(balance: float) -> str:
    """Return the tier name string for the given balance."""
    if FORCE_TIER and FORCE_TIER in ACCOUNT_TIERS:
        return FORCE_TIER
    for name, tier in ACCOUNT_TIERS.items():
        if tier["min_balance"] <= balance <= tier["max_balance"]:
            return name
    if balance < 500:
        return "MICRO"
    return "WHALE"


# ════════════════════════════════════════════════════════════
#  Active parameters (defaults — overridden at runtime by tier)
# ════════════════════════════════════════════════════════════
RISK_PCT            = 1.0
LOT_SIZE            = 0.01
MAX_SPREAD_POINTS   = 30
RISK_REWARD_RATIO   = 1.5
BREAKEVEN_PCT       = 0.50
MAX_POSITIONS       = 1
DAILY_DD_LIMIT_PCT  = 3.0


def apply_tier(balance: float) -> str:
    """
    Apply tier-based parameters to module-level config variables.
    Call once on startup & optionally on daily reset.
    Returns tier name for logging.
    """
    global RISK_PCT, LOT_SIZE, MAX_SPREAD_POINTS, RISK_REWARD_RATIO
    global BREAKEVEN_PCT, MAX_POSITIONS, DAILY_DD_LIMIT_PCT

    tier = get_tier(balance)
    name = get_tier_name(balance)

    RISK_PCT           = tier["risk_pct"]
    LOT_SIZE           = tier["lot_size"]
    MAX_SPREAD_POINTS  = tier["max_spread_points"]
    RISK_REWARD_RATIO  = tier["risk_reward_ratio"]
    BREAKEVEN_PCT      = tier["breakeven_pct"]
    MAX_POSITIONS      = tier["max_positions"]
    DAILY_DD_LIMIT_PCT = tier["daily_dd_limit_pct"]

    return name


# ── Volatility Filter (XAUUSD-optimized for M1) ────────────
ATR_PERIOD          = 14
ATR_THRESHOLD       = 0.80      # XAUUSD M1 ATR ปกติ ~0.5–3.0 กรองตลาดนิ่งออก
BB_PERIOD           = 20
BB_STD              = 2.0
BB_EXPANSION_FACTOR = 1.3       # BB width ต้อง > ค่าเฉลี่ย × 1.3

# ── Liquidity Sweep ────────────────────────────────────────
SWING_LOOKBACK      = 20        # Bars to look back for swing H/L
SWEEP_WICK_RATIO    = 0.6       # Min wick-to-body ratio for sweep candle
FVG_MIN_SIZE_ATR    = 0.5       # FVG gap must be >= 0.5 × ATR
MAX_SL_ATR_MULT     = 3.0       # SL distance > 3x ATR = skip (wick ยาวเกิน)
SWING_CONFIRM_BARS  = 3         # Bars each side to confirm swing H/L structure

# ── Execution ──────────────────────────────────────────────
ORDER_MAGIC         = 615900
ORDER_COMMENT       = "VALIDUS"
POSITION_CHECK_SEC  = 0.5       # Seconds between position management ticks

# ── News Filter ─────────────────────────────────────────────
NEWS_FILTER_ENABLED = True
NEWS_BUFFER_MIN     = 15        # Minutes before/after red news to pause
NEWS_CURRENCIES     = ["USD"]   # Only filter these currencies (relevant to XAUUSD)

# ── Notifications (Line Notify) ─────────────────────────────
LINE_NOTIFY_TOKEN   = ""        # Leave blank to disable
LINE_NOTIFY_URL     = "https://notify-api.line.me/api/notify"

# ── VPS / Deployment ─────────────────────────────────────────
AUTO_START           = True      # True = start trading immediately (VPS mode)
HEADLESS             = False     # True = no dashboard clear (better for log files)
AUTO_RESTART_DELAY   = 10        # Seconds to wait before auto-restart after crash
DAILY_RESET_HOUR_UTC = 0         # Hour (UTC) to re-snapshot balance & re-apply tier

# ── Session Filter ──────────────────────────────────────────
SESSION_FILTER_ENABLED = True
SESSION_START_UTC      = 7       # London pre-open (07:00 UTC)
SESSION_END_UTC        = 21      # NY close (21:00 UTC)

# ── Signal Cooldown ─────────────────────────────────────────
SIGNAL_COOLDOWN_BARS   = 5       # Min M1 bars between signals on same symbol

# ── HTF Trend Filter (M5 EMA) ──────────────────────────────
HTF_EMA_PERIOD         = 50      # EMA period on M5 for trend direction

# ── Logging ─────────────────────────────────────────────────
LOG_FILE             = "validus.log"
LOG_LEVEL            = "INFO"    # DEBUG | INFO | WARNING | ERROR
LOG_MAX_BYTES        = 5_000_000 # 5 MB per log file
LOG_BACKUP_COUNT     = 5         # Keep 5 rotated files (25 MB total max)
