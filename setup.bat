@echo off
cd /d "%~dp0"
echo ══════════════════════════════════════════════
echo        VALIDUS — Smart Money Sniper MT5
echo        Installation Script
echo ══════════════════════════════════════════════
echo.

REM ── Check Python ──────────────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Install Python 3.10+ first.
    echo    https://www.python.org/downloads/
    pause
    exit /b 1
)
echo ✅ Python found:
python --version
echo.

REM ── Install dependencies ──────────────────────
echo Installing Dependencies...
pip install --upgrade pip
pip install MetaTrader5 pandas pandas_ta numpy requests schedule
if %errorlevel% neq 0 (
    echo ❌ pip install failed!
    pause
    exit /b 1
)
echo.
echo ✅ All Python packages installed.
echo.

REM ── Verify config ─────────────────────────────
if exist config.py (
    echo ✅ config.py found.
) else (
    echo ❌ config.py not found! Make sure all files are in this folder.
    pause
    exit /b 1
)
echo.

echo ══════════════════════════════════════════════
echo  ✅  Installation complete!
echo.
echo  Next steps:
echo    1. Edit config.py — set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
echo    2. Run: run_bot.bat (auto-start + crash recovery)
echo    3. Optional: run install_service.bat (auto-start on VPS boot)
echo ══════════════════════════════════════════════
pause
