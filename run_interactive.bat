@echo off
title VALIDUS — Interactive Mode
cd /d "%~dp0"
echo Starting VALIDUS in interactive mode...
echo Press [S] to Start, [Q] to Stop, [P] to Panic Close
echo.
python main.py
pause
