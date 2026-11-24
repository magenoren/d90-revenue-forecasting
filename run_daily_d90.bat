@echo off
REM Run from the repository root; install dependencies into .venv first.
cd /d "%~dp0"
if not exist "outputs\logs" mkdir "outputs\logs"
".venv\Scripts\python.exe" "scripts\daily_d90_forecast.py" >> "outputs\logs\daily_d90.log" 2>&1
REM To opt into BigQuery forecast-table writes, add --write-bigquery above.
exit /b %errorlevel%
