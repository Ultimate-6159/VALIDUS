# ============================================================
#  SWEEP SCALPER — M1 Liquidity Sweep + M15 Trend Filter
#  High-Frequency Scalping Bot for XAUUSD on MetaTrader 5
#  Strategy: 3-Candle Fractal Sweep → Wick Rejection → Entry
#
#  This script is a thin launcher that activates SWEEP mode
#  in the VALIDUS modular system. For configuration, edit
#  config.py (symbols, tiers, sweep settings).
# ============================================================
from __future__ import annotations

import config

# Activate Sweep Scalper strategy mode
config.STRATEGY_MODE = "SWEEP"

# Apply multi-symbol overrides from sweep_scalper defaults
config.SYMBOLS = ["XAUUSDm", "EURUSDm", "GBPUSDm"]
config.TIMEFRAME_HTF = "M15"

# SWEEP uses market orders (entry = current close, no pullback expected)
config.USE_LIMIT_ORDER = False

# Auto-start for convenience
config.AUTO_START = True

from main import main  # noqa: E402

if __name__ == "__main__":
    main()
