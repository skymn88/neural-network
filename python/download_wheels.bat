@echo off
setlocal
cd /d "%~dp0.."
echo ONLINE PREPARATION ONLY: run on an internet-connected Windows x64 PC with Python 3.9.
echo The offline training PC should run setup_offline.bat, not this script.
set "ACTUATOR_WHEELS=%~1"
if not defined ACTUATOR_WHEELS set "ACTUATOR_WHEELS=%CD%\wheelhouse"
py -3.9 -c "import struct,sys;sys.exit(0 if struct.calcsize('P')==8 else 1)"
if errorlevel 1 goto failed
py -3.9 -m pip --isolated download --only-binary=:all: --dest "%ACTUATOR_WHEELS%" -r python\requirements.txt
if errorlevel 1 goto failed
py -3.9 -m pip --isolated download --only-binary=:all: --dest "%ACTUATOR_WHEELS%" --index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu
if errorlevel 1 goto failed
echo Copy the entire wheelhouse folder to the offline PC, including transitive dependencies.
exit /b 0
:failed
echo Download failed. Read the error above. Use Python 3.9 on Windows x64.
pause
exit /b 1
