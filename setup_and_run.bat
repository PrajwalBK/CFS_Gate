@echo off
setlocal EnableDelayedExpansion
title GateVision AI - Automated Setup and Launcher

echo ======================================================================
echo           GateVision AI - Automated Setup and Launcher
echo ======================================================================
echo.

:: Detect root directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if exist "backend\app\main.py" (
    set "PROJECT_ROOT=%SCRIPT_DIR%"
) else if exist "ai-video-based-inventory\backend\app\main.py" (
    set "PROJECT_ROOT=%SCRIPT_DIR%ai-video-based-inventory\"
    cd /d "%PROJECT_ROOT%"
) else if exist "..\backend\app\main.py" (
    set "PROJECT_ROOT=%SCRIPT_DIR%..\"
    cd /d "%PROJECT_ROOT%"
) else (
    echo [ERROR] Cannot locate GateVision project root.
    pause
    exit /b 1
)

set "BACKEND_DIR=%PROJECT_ROOT%backend"
set "FRONTEND_DIR=%PROJECT_ROOT%frontend"
set "VENV_DIR=%BACKEND_DIR%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
set "PORT=8001"

echo [1/7] Granting permissions and unblocking files...
:: Remove read-only attributes
attrib -R "%PROJECT_ROOT%*.*" /S /D >nul 2>&1

:: Create directories and grant full permissions to current user
if not exist "%BACKEND_DIR%\data" mkdir "%BACKEND_DIR%\data"
if not exist "%BACKEND_DIR%\snapshots" mkdir "%BACKEND_DIR%\snapshots"
if not exist "%BACKEND_DIR%\logs" mkdir "%BACKEND_DIR%\logs"
if not exist "%BACKEND_DIR%\models" mkdir "%BACKEND_DIR%\models"
if not exist "%BACKEND_DIR%\uploads\ocr-spool" mkdir "%BACKEND_DIR%\uploads\ocr-spool"

icacls "%BACKEND_DIR%\data" /grant "%USERNAME%:(OI)(CI)F" /T /Q /C >nul 2>&1
icacls "%BACKEND_DIR%\snapshots" /grant "%USERNAME%:(OI)(CI)F" /T /Q /C >nul 2>&1
icacls "%BACKEND_DIR%\logs" /grant "%USERNAME%:(OI)(CI)F" /T /Q /C >nul 2>&1
icacls "%BACKEND_DIR%\models" /grant "%USERNAME%:(OI)(CI)F" /T /Q /C >nul 2>&1
icacls "%BACKEND_DIR%\uploads" /grant "%USERNAME%:(OI)(CI)F" /T /Q /C >nul 2>&1
echo   Full file permissions granted for user %USERNAME%.

net session >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo   Administrator privileges detected. Configuring Windows Firewall for Port %PORT%...
    netsh advfirewall firewall delete rule name="GateVision AI Server (Port %PORT%)" >nul 2>&1
    netsh advfirewall firewall add rule name="GateVision AI Server (Port %PORT%)" dir=in action=allow protocol=TCP localport=%PORT% profile=any >nul 2>&1
    echo   Windows Firewall rule registered.
)

echo.
echo [2/7] Checking Python installation...
set "SYSTEM_PYTHON="
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "SYSTEM_PYTHON=python"
) else (
    where py >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "SYSTEM_PYTHON=py -3"
    )
)

if "%SYSTEM_PYTHON%"=="" (
    echo [ERROR] Python was not found in your system PATH.
    echo Please download and install Python 3.10+ from https://www.python.org/
    echo Be sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('%SYSTEM_PYTHON% --version 2^>^&1') do set "PY_VER=%%v"
echo   Found: !PY_VER!

echo.
echo [3/7] Verifying Python virtual environment...
if not exist "%VENV_PYTHON%" (
    echo   Creating virtual environment in %VENV_DIR%...
    %SYSTEM_PYTHON% -m venv "%VENV_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo   Virtual environment created successfully.
) else (
    echo   Virtual environment ready at %VENV_DIR%.
)

echo.
echo [4/7] Installing all backend dependencies...
echo   Upgrading pip, setuptools, and wheel...
"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel --quiet

echo   Installing core packages from requirements.txt...
"%VENV_PIP%" install -r "%BACKEND_DIR%\requirements.txt"

if exist "%BACKEND_DIR%\requirements-ai.txt" (
    echo   Installing AI packages: Ultralytics, EasyOCR, PaddleOCR...
    "%VENV_PIP%" install -r "%BACKEND_DIR%\requirements-ai.txt"
)

"%VENV_PIP%" install "safetensors>=0.8.0" --quiet
echo   Backend dependencies verified.

echo.
echo [5/7] Verifying configuration and storage directories...
if not exist "%BACKEND_DIR%\.env" (
    if exist "%BACKEND_DIR%\.env.example" (
        echo   Creating .env from .env.example...
        copy "%BACKEND_DIR%\.env.example" "%BACKEND_DIR%\.env" >nul
    )
) else (
    echo   Config file .env present.
)
echo   Storage directories verified.

echo.
echo [6/7] Installing frontend dependencies and checking build...
set "FRONTEND_INDEX=%FRONTEND_DIR%\dist\gate\browser\index.html"
where npm >nul 2>&1
if %ERRORLEVEL% equ 0 (
    pushd "%FRONTEND_DIR%"
    if not exist "node_modules" (
        echo   Installing frontend dependencies...
        call npm install
    ) else (
        echo   Frontend node_modules present.
    )
    if not exist "%FRONTEND_INDEX%" (
        echo   Compiling Angular frontend...
        call npm run build
    ) else (
        echo   Frontend build ready.
    )
    popd
) else (
    echo   [NOTICE] npm was not found. Backend REST API and Swagger Docs will still be available at /docs.
)

echo.
echo [7/7] Launching GateVision AI...

:: Check if port is already running, find next free port without affecting running instances
:check_port_loop
netstat -aon | findstr ":!PORT! " | findstr "LISTENING" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set /a PREV_PORT=!PORT!
    set /a PORT+=1
    echo   Port !PREV_PORT! is in use by another application. Keeping it running and switching to port !PORT!...
    goto check_port_loop
)

echo ======================================================================
echo    GateVision Web Dashboard : http://localhost:%PORT%
echo    Interactive API Docs     : http://localhost:%PORT%/docs
echo ======================================================================
echo Press Ctrl+C in this window to stop the server.
echo.

:: Automatically open browser after 2 seconds in background
start "" /b cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:%PORT%"

cd /d "%BACKEND_DIR%"
"%VENV_PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port %PORT% --workers 1 --no-access-log

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application server stopped with exit code %ERRORLEVEL%.
    pause
)
