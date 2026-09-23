@echo off
setlocal EnableDelayedExpansion
title GateVision AI - Auto Setup and Launcher

echo ======================================================================
echo           GateVision AI - Automated Setup and Launcher
echo ======================================================================
echo.

:: Detect directories
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if exist "backend\app\main.py" (
    set "PROJECT_ROOT=%SCRIPT_DIR%"
) else if exist "app\main.py" (
    set "PROJECT_ROOT=%SCRIPT_DIR%..\"
    cd /d "%PROJECT_ROOT%"
) else (
    echo [ERROR] Cannot locate project root directory.
    pause
    exit /b 1
)

set "BACKEND_DIR=%PROJECT_ROOT%backend"
set "FRONTEND_DIR=%PROJECT_ROOT%frontend"
set "VENV_DIR=%BACKEND_DIR%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"

echo [1/6] Checking Python installation...
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
    echo Please install Python 3.10, 3.11, or 3.12 from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('%SYSTEM_PYTHON% --version 2^>^&1') do set "PY_VER=%%v"
echo Found: %PY_VER%

echo.
echo [2/6] Setting up Python virtual environment...
if not exist "%VENV_PYTHON%" (
    echo Creating virtual environment in %VENV_DIR%...
    %SYSTEM_PYTHON% -m venv "%VENV_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists at %VENV_DIR%.
)

echo.
echo [3/6] Installing and verifying backend dependencies...
"%VENV_PYTHON%" -m pip install --upgrade pip --quiet
echo Installing core dependencies from requirements.txt...
"%VENV_PIP%" install -r "%BACKEND_DIR%\requirements.txt" --quiet
if exist "%BACKEND_DIR%\requirements-ai.txt" (
    echo Installing AI dependencies (Ultralytics, EasyOCR, etc.)...
    "%VENV_PIP%" install -r "%BACKEND_DIR%\requirements-ai.txt" --quiet
)

echo.
echo [4/6] Verifying directory structure and environment settings...
if not exist "%BACKEND_DIR%\.env" (
    if exist "%BACKEND_DIR%\.env.example" (
        echo Copying .env.example to .env...
        copy "%BACKEND_DIR%\.env.example" "%BACKEND_DIR%\.env" >nul
    )
)

if not exist "%BACKEND_DIR%\data" mkdir "%BACKEND_DIR%\data"
if not exist "%BACKEND_DIR%\snapshots" mkdir "%BACKEND_DIR%\snapshots"
if not exist "%BACKEND_DIR%\logs" mkdir "%BACKEND_DIR%\logs"
if not exist "%BACKEND_DIR%\models" mkdir "%BACKEND_DIR%\models"
if not exist "%BACKEND_DIR%\uploads\ocr-spool" mkdir "%BACKEND_DIR%\uploads\ocr-spool"

echo.
echo [5/6] Checking Frontend build...
set "FRONTEND_INDEX=%FRONTEND_DIR%\dist\gate\browser\index.html"
if exist "%FRONTEND_INDEX%" (
    echo Frontend production build is ready.
) else (
    echo Frontend build not found. Checking Node.js / npm...
    where npm >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo Building Angular frontend (this may take 1-2 minutes)...
        pushd "%FRONTEND_DIR%"
        if not exist "node_modules" (
            echo Running npm install...
            call npm install
        )
        echo Running npm run build...
        call npm run build
        popd
        if exist "%FRONTEND_INDEX%" (
            echo Frontend built successfully!
        ) else (
            echo [WARNING] Frontend build failed. API will still be available at /docs.
        )
    ) else (
        echo [INFO] npm not found. If you want the full web UI, please install Node.js.
        echo Backend REST API and Swagger Docs will still be fully functional at /docs.
    )
)

echo.
echo [6/6] Launching GateVision AI...
echo ======================================================================
echo    GateVision Web Dashboard : http://localhost:8001
echo    Interactive API Docs     : http://localhost:8001/docs
echo ======================================================================
echo Press Ctrl+C in this window to stop the server.
echo.

cd /d "%BACKEND_DIR%"
"%VENV_PYTHON%" -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 1 --no-access-log

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application stopped unexpectedly with error code %ERRORLEVEL%.
    pause
)
