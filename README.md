# VALIDUS — Smart Money Sniper MT5

Automated trading bot for MetaTrader 5 based on **Smart Money Concepts** (Liquidity Sweep → Displacement → FVG Retrace).

## Architecture

| Module | File | Role |
|--------|------|------|
| **Config** | `config.py` | All settings (pairs, risk, tiers, thresholds) |
| **Strategy** (B) | `strategy.py` | SMC entry logic — edit this most often |
| **Utilities** | `utils.py` | Logging (auto-rotate), Line Notify, news filter |
| **Core** (A/C/D) | `main.py` | Data streamer, execution, guardian, dashboard |

## Quick Start

```
1. Install:   double-click  setup.bat
2. Configure: edit          config.py   (MT5 login, password, server, pairs)
3. Run:       double-click  run_bot.bat
```

## VPS Deployment (Unattended)

```
1. setup.bat                      ← install dependencies
2. Edit config.py                 ← MT5 credentials + AUTO_START = True
3. install_service.bat            ← register auto-start on boot (run as Admin)
4. run_bot.bat                    ← start now (auto-restarts on crash)
```

### File roles on VPS

| File | Purpose |
|------|---------|
| `run_bot.bat` | Production runner — auto-start + crash restart loop |
| `run_interactive.bat` | Manual mode — press [S] to start, has dashboard |
| `install_service.bat` | Register Windows Task Scheduler (auto-start on logon) |

### CLI flags

```
python main.py --autostart --headless
```

| Flag | Effect |
|------|--------|
| `--autostart` | Skip waiting for [S] — start trading immediately |
| `--headless` | Disable terminal clearing — output goes to log file |

## Account Tier System

Risk parameters auto-adapt based on account balance:

| Tier | Balance | Risk/trade | Max Pos | DD Limit |
|------|---------|-----------|---------|----------|
| MICRO | $500 – $2K | 1.00% | 1 | 3.0% |
| SMALL | $2K – $10K | 1.00% | 2 | 3.0% |
| MEDIUM | $10K – $100K | 0.75% | 3 | 2.5% |
| LARGE | $100K – $1M | 0.50% | 3 | 2.0% |
| WHALE | $1M – $5M | 0.25% | 2 | 1.5% |

Override with `FORCE_TIER = "MEDIUM"` in `config.py`.

## Hotkeys (CLI Dashboard)

| Key | Action |
|-----|--------|
| `S` | Start trading |
| `Q` | Stop trading |
| `P` | Panic — close all positions immediately |

## Resilience Features

- **Auto-restart**: `run_bot.bat` restarts after crash (10s delay)
- **Daily reset**: Re-snapshots balance & tier at midnight UTC
- **Connection watchdog**: Auto-reconnects MT5 (5 retries)
- **Log rotation**: 5 × 5 MB files (25 MB max)
- **Daily drawdown guard**: Force-stops when limit is hit, resets next day

## Requirements

- Python 3.10+
- MetaTrader 5 Terminal (Hedge Account)
- Windows (VPS recommended: 2 vCPU / 4 GB RAM / NVMe SSD)

## License

MIT