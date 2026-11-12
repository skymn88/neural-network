@echo off
setlocal
cd /d "%~dp0.."
set "ACTUATOR_WHEELS=%~1"
if not defined ACTUATOR_WHEELS set "ACTUATOR_WHEELS=%CD%\wheelhouse"
if not exist "%ACTUATOR_WHEELS%" (
  echo Local wheel folder not found: %ACTUATOR_WHEELS%
  echo Usage: python\setup_offline.bat D:\path\to\wheelhouse
  goto failed
)
if not exist ".venv39\Scripts\python.exe" (
  py -3.9 -m venv .venv39
  if errorlevel 1 goto failed
)
.venv39\Scripts\python.exe -c "import sys,struct;sys.exit(0 if sys.version_info[:2]==(3,9) and struct.calcsize('P')==8 else 1)"
if errorlevel 1 (
  echo .venv39 must contain 64-bit Python 3.9.
  goto failed
)
.venv39\Scripts\python.exe -m pip --isolated install --no-index --find-links "%ACTUATOR_WHEELS%" --only-binary=:all: --disable-pip-version-check -r python\requirements-cpu.txt
if errorlevel 1 goto failed
.venv39\Scripts\python.exe python\check_environment.py
if errorlevel 1 goto failed
echo Offline setup complete. Run python\start.bat.
exit /b 0
:failed
echo Offline setup failed. See the missing package or interpreter above.
echo No online package source is used by this installer.
pause
exit /b 1
