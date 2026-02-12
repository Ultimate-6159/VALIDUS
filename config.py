# ============================================================
#  VALIDUS — Configuration
#  Adaptive for account sizes $500 – $5,000,000
# ============================================================

# ── MT5 Connection ──────────────────────────────────────────
MT5_LOGIN=415146568
MT5_PASSWORD="Ultimate@6159"
MT5_SERVER="Exness-MT5Trial14"
MT5_PATH     = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# ── Trading Pairs ───────────────────────────────────────────
SYMBOLS = [
    "XAUUSDm",
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
    #   Ultra-Aggressive Scalping — เป้า 15%/วัน, 200%/เดือน
    #   Risk 5% + RR 1:2.5 + Partial TP + Compounding
    "MICRO": {
        "min_balance":        500,
        "max_balance":      2_000,
        "risk_pct":           5.0,      # 5% ต่อไม้ ($25 ที่ $500)
        "lot_size":          0.01,      # fallback lot
        "max_positions":        5,      # เปิดได้ 5 ออเดอร์พร้อมกัน
        "daily_dd_limit_pct": 18.0,     # DD limit 18% — ultra-aggressive
        "risk_reward_ratio":   2.5,     # RR 1:2.5 — กำไรสูงเมื่อชนะ
        "breakeven_pct":      0.35,     # เลื่อน BE เร็วมากที่ 35% TP
        "max_spread_points":    45,
    },
    # ── SMALL : $2,001 – $10,000 ────────────────────────────
    #   ยังคง aggressive — ต่อยอด compounding จาก MICRO
    "SMALL": {
        "min_balance":      2_001,
        "max_balance":     10_000,
        "risk_pct":           4.0,      # 4% ต่อไม้ ($80–$400)
        "lot_size":          0.02,
        "max_positions":        5,
        "daily_dd_limit_pct": 15.0,     # DD limit 15%
        "risk_reward_ratio":   2.5,     # RR 1:2.5
        "breakeven_pct":      0.35,
        "max_spread_points":    35,
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
RISK_PCT            = 5.0
LOT_SIZE            = 0.01
MAX_SPREAD_POINTS   = 45
RISK_REWARD_RATIO   = 2.5
BREAKEVEN_PCT       = 0.35
MAX_POSITIONS       = 5
DAILY_DD_LIMIT_PCT  = 18.0


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
ATR_THRESHOLD       = 0.25      # ลดเกณฑ์ให้เทรดได้ตลาด vol ปานกลาง
BB_PERIOD           = 20
BB_STD              = 2.0
BB_EXPANSION_FACTOR = 1.05      # BB ขยายเล็กน้อยก็เข้าได้

# ── Liquidity Sweep ────────────────────────────────────────
SWING_LOOKBACK      = 10        # lookback สั้นขึ้น — จับ sweep เยอะขึ้น
SWEEP_WICK_RATIO    = 0.20      # wick/body ต่ำก็ยอมรับ — sweep เบาๆ
FVG_MIN_SIZE_ATR    = 0.10      # FVG เล็กก็เข้า — scalping mode
MAX_SL_ATR_MULT     = 5.0       # SL กว้างขึ้นได้ (RR 1:2 ชดเชย)
SWING_CONFIRM_BARS  = 2         # ยืนยัน swing แค่ 2 bars

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
SESSION_START_UTC      = 1       # ตั้งแต่ Asian session (01:00 UTC)
SESSION_END_UTC        = 23      # ถึง NY close (23:00 UTC)

# ── Signal Cooldown ─────────────────────────────────────────
SIGNAL_COOLDOWN_BARS   = 1       # แทบไม่มี cooldown — scalp ทุกสัญญาณ

# ── HTF Trend Filter (M5 EMA) ──────────────────────────────
HTF_EMA_PERIOD         = 20      # EMA20 ไว — trend flip เร็วขึ้น บล็อคสัญญาณน้อยลง

# ── Trailing Stop (ATR-based) ───────────────────────────────
TRAILING_STOP_ENABLED  = True
TRAILING_ATR_PERIOD    = 14       # ATR period for trail distance calculation
TRAILING_ATR_MULT      = 2.0     # Trail SL at price ± ATR × mult (ให้ room วิ่ง)

# ── Limit Order Mode ────────────────────────────────────────
USE_LIMIT_ORDER        = True     # True = limit at FVG mid | False = market order
LIMIT_ORDER_EXPIRY_SEC = 600      # Cancel unfilled limit after 10 min (600s)

# ── Multi-Timeframe FVG (M5 confirmation) ───────────────────
MTF_FVG_ENABLED        = False    # ปิดไว้ก่อน — เพิ่มสัญญาณ (เปิดเมื่อพอร์ตใหญ่)
MTF_FVG_LOOKBACK       = 20      # M5 bars to scan for recent FVG zones
MTF_FVG_MIN_SIZE_ATR   = 0.3     # M5 FVG min gap as ATR multiplier

# ── Backtest ────────────────────────────────────────────────
BACKTEST_DATA_DIR      = "data"   # Directory for historical CSV files
BACKTEST_INITIAL_BAL   = 10_000.0 # Starting balance for backtest
BACKTEST_COMMISSION    = 0.0      # Commission per lot per side ($)
BACKTEST_SLIPPAGE_PTS  = 2        # Simulated slippage in points
BACKTEST_CONTRACT_SIZE = 100      # XAUUSD: 100 oz per standard lot
BACKTEST_TICK_SIZE     = 0.01     # XAUUSD minimum price increment

# ── Monte Carlo ─────────────────────────────────────────────
MC_SIMULATIONS         = 1_000    # Number of Monte Carlo shuffles
MC_CONFIDENCE_PCT      = 95       # Confidence interval percentage

# ── Partial Take-Profit ─────────────────────────────────────
PARTIAL_TP_ENABLED     = True     # ปิด 50% ที่ RR 1:1, ปล่อย 50% วิ่งถึง TP
PARTIAL_TP_RATIO       = 0.50     # สัดส่วน lot ที่ปิด (50%)
PARTIAL_TP_RR          = 1.0      # ปิดครึ่งแรกที่ RR 1:1

# ── Logging ─────────────────────────────────────────────────
LOG_FILE             = "validus.log"
LOG_LEVEL            = "INFO"    # DEBUG | INFO | WARNING | ERROR
LOG_MAX_BYTES        = 5_000_000 # 5 MB per log file
LOG_BACKUP_COUNT     = 5         # Keep 5 rotated files (25 MB total max)

# ── Decision Log (CSV สำหรับวิเคราะห์ย้อนหลัง) ─────────────
DECISION_LOG_ENABLED = True      # เก็บ log ทุก decision ลง CSV
DECISION_LOG_DIR     = "logs"    # โฟลเดอร์เก็บ decisions_YYYY-MM-DD.csv
