@echo off
cd /d "%~dp0"
venv\Scripts\python.exe hive_worker.py
pause
