@echo off
title Cognitive LLM Launcher
color 0B

echo ========================================================
echo   Cognitive LLM System Launcher
echo ========================================================
echo.

:: Start Web Server in a new window
echo [Launcher] Starting Web Server in background...
start "Cognitive Web Server" cmd /k "python web.py"

:: Give it a few seconds
timeout /t 3 /nobreak > nul

:: Start CLI in the current window
echo [Launcher] Starting Interactive CLI...
python main.py

echo.
echo [Launcher] CLI session ended.
pause
