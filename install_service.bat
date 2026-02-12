@echo off
cd /d "%~dp0"
echo ══════════════════════════════════════════════
echo  VALIDUS — Register Windows Task Scheduler
echo  (Auto-start on boot / user logon)
echo ══════════════════════════════════════════════
echo.

REM ── Must run as Administrator ─────────────────
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Please run this script as Administrator!
    echo    Right-click → Run as administrator
    pause
    exit /b 1
)

set BOT_DIR=%~dp0
set TASK_NAME=VALIDUS_TradingBot

REM ── Remove old task if exists ─────────────────
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

REM ── Create scheduled task ─────────────────────
REM  Trigger: at system startup + on user logon
REM  Action : run run_bot.bat in this directory
schtasks /create ^
    /tn "%TASK_NAME%" ^
    /tr "\"%BOT_DIR%run_bot.bat\"" ^
    /sc onlogon ^
    /rl highest ^
    /f

if %errorlevel% equ 0 (
    echo.
    echo ══════════════════════════════════════════════
    echo  ✅ Task "%TASK_NAME%" registered!
    echo.
    echo  The bot will auto-start when you log in.
    echo  To remove:  schtasks /delete /tn "%TASK_NAME%" /f
    echo  To run now: schtasks /run /tn "%TASK_NAME%"
    echo ══════════════════════════════════════════════
) else (
    echo.
    echo ❌ Failed to create scheduled task.
)
pause
