@echo off
title Customer Churn & Retention Engine
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Setting up Python virtual environment...
  py -3 -m venv .venv
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
".venv\Scripts\python.exe" app.py
echo.
pause
