@echo off
chcp 65001 >nul
title VALIDUS - Smart Money Sniper MT5
cd /d "%~dp0"

:LOOP
echo ==============================================
echo  VALIDUS starting...  %date% %time%
echo ==============================================
python main.py --autostart --headless

echo.
echo [!] VALIDUS exited. Auto-restart in 10 seconds...
echo     Press Ctrl+C to cancel.
timeout /t 10 /nobreak >nul
goto LOOP
