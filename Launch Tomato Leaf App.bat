@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating the app's Python environment...
    call :find_python
    if errorlevel 1 goto failed
    !PYTHON_CMD! -m venv .venv
    if errorlevel 1 goto failed
)

if not exist ".venv\Scripts\streamlit.exe" (
    echo Installing app requirements. This can take a few minutes on first launch...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto failed
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto failed
)

echo Starting Tomato Leaf Decision Support. Close this window to stop the app.
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo The app could not start.
if not defined PYTHON_CMD if not exist ".venv\Scripts\python.exe" echo Install Python 3.10, 3.11, 3.12, or 3.14 from python.org, then run this shortcut again.
if exist ".venv\Scripts\python.exe" echo Check your internet connection and the error above, then run this shortcut again.
pause
exit /b 1

:find_python
where py >nul 2>&1
if not errorlevel 1 (
    py -3.14 -c "import sys" >nul 2>&1
    if not errorlevel 1 (set "PYTHON_CMD=py -3.14"&exit /b 0)
    py -3.12 -c "import sys" >nul 2>&1
    if not errorlevel 1 (set "PYTHON_CMD=py -3.12"&exit /b 0)
    py -3.11 -c "import sys" >nul 2>&1
    if not errorlevel 1 (set "PYTHON_CMD=py -3.11"&exit /b 0)
    py -3.10 -c "import sys" >nul 2>&1
    if not errorlevel 1 (set "PYTHON_CMD=py -3.10"&exit /b 0)
)
where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 14)" >nul 2>&1
    if not errorlevel 1 (set "PYTHON_CMD=python"&exit /b 0)
)
exit /b 1
