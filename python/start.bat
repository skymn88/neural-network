@echo off
setlocal
cd /d "%~dp0.."
if defined ACTUATOR_PYTHON (
  set "ACTUATOR_RUN_PYTHON=%ACTUATOR_PYTHON%"
) else (
  set "ACTUATOR_RUN_PYTHON=%CD%\.venv39\Scripts\python.exe"
)
if not exist "%ACTUATOR_RUN_PYTHON%" (
  echo Python 3.9 environment not found. No download will be attempted.
  echo First run: python\setup_offline.bat D:\path\to\wheelhouse
  echo Or set ACTUATOR_PYTHON to your existing Python 3.9 python.exe.
  goto failed
)
"%ACTUATOR_RUN_PYTHON%" python\check_environment.py
if errorlevel 1 goto failed
"%ACTUATOR_RUN_PYTHON%" python\app.py
if errorlevel 1 goto failed
exit /b 0
:failed
pause
exit /b 1
