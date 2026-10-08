@echo off
chcp 65001 >nul
cd /d "%~dp0"
set GRADIO_ANALYTICS_ENABLED=False
set PYTHONUTF8=1
python\python.exe app.py --inbrowser
pause
