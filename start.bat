@echo off
setlocal
cd /d "%~dp0"

if not exist "backend\.venv\Scripts\python.exe" (
    echo [INFO] Backend virtual environment not detected. Running initial setup...
    call "%~dp0setup_and_run.bat" %*
    exit /b %ERRORLEVEL%
)

if not exist "frontend\dist\gate\browser\index.html" (
    echo [INFO] Frontend build not detected. Running initial setup...
    call "%~dp0setup_and_run.bat" %*
    exit /b %ERRORLEVEL%
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
