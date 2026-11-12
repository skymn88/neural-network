@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo First launch: creating Python 3.12 environment...
  py -3.12 -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import torch, numpy, scipy, pandas, openpyxl, fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
  echo First launch: downloading CPU dependencies. Internet is needed for installation only.
  ".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 goto failed
  ".venv\Scripts\python.exe" -m pip install -r python\requirements.txt
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" python\app.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Setup or startup failed. Python 3.12 must be installed. Read the error above.
pause
exit /b 1
