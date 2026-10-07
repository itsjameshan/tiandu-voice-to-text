@echo off
chcp 65001 >nul
rem 联网安装（备用）：建虚拟环境、用清华镜像装依赖、下载模型
cd /d "%~dp0\..\.."
py -3.11 -m venv .venv 2>nul || python -m venv .venv
.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
.venv\Scripts\python.exe models\download_models.py
pause
