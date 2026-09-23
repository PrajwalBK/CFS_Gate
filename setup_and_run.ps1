# GateVision AI - Automated Setup and Launcher (PowerShell)
[CmdletBinding()]
param(
    [string]$HostAddress = '0.0.0.0',
    [ValidateRange(1, 65535)]
    [int]$Port = 8001,
    [switch]$ForceReinstall
)

$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot

$backendRoot = Join-Path $scriptRoot 'backend'
$frontendRoot = Join-Path $scriptRoot 'frontend'
$venvDir = Join-Path $backendRoot '.venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'
$venvPip = Join-Path $venvDir 'Scripts\pip.exe'
$frontendIndex = Join-Path $frontendRoot 'dist\gate\browser\index.html'

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "          GateVision AI - Automated Setup and Launcher" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Verify Python
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
$sysPython = (Get-Command python.exe -ErrorAction SilentlyContinue)
if (-not $sysPython) {
    $sysPython = (Get-Command py.exe -ErrorAction SilentlyContinue)
}

if (-not $sysPython) {
    Write-Error "Python 3.10+ was not found in PATH. Please install Python from https://www.python.org/ and select 'Add Python to PATH'."
    exit 1
}

$pyVersion = (& $sysPython.Source --version 2>&1)
Write-Host "Found: $pyVersion" -ForegroundColor Green

# 2. Virtual Environment
Write-Host "`n[2/6] Setting up virtual environment..." -ForegroundColor Yellow
if ($ForceReinstall -or -not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    Write-Host "Creating virtual environment at $venvDir..."
    & $sysPython.Source -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# 3. Dependencies
Write-Host "`n[3/6] Installing / Verifying backend dependencies..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip --quiet
Write-Host "Checking core requirements..."
& $venvPip install -r (Join-Path $backendRoot 'requirements.txt') --quiet

$reqAi = Join-Path $backendRoot 'requirements-ai.txt'
if (Test-Path -LiteralPath $reqAi -PathType Leaf) {
    Write-Host "Checking AI requirements (Ultralytics, EasyOCR)..."
    & $venvPip install -r $reqAi --quiet
}
Write-Host "Backend dependencies verified." -ForegroundColor Green

# 4. Folders and .env
Write-Host "`n[4/6] Verifying directory structure and configuration..." -ForegroundColor Yellow
$envFile = Join-Path $backendRoot '.env'
$envExample = Join-Path $backendRoot '.env.example'
if (-not (Test-Path -LiteralPath $envFile -PathType Leaf) -and (Test-Path -LiteralPath $envExample -PathType Leaf)) {
    Write-Host "Creating .env from .env.example..."
    Copy-Item $envExample $envFile
}

$dirs = @('data', 'snapshots', 'logs', 'models', 'uploads\ocr-spool')
foreach ($d in $dirs) {
    $targetDir = Join-Path $backendRoot $d
    if (-not (Test-Path -LiteralPath $targetDir -PathType Container)) {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    }
}
Write-Host "Directory structure verified." -ForegroundColor Green

# 5. Frontend Build
Write-Host "`n[5/6] Checking Frontend build..." -ForegroundColor Yellow
if (Test-Path -LiteralPath $frontendIndex -PathType Leaf) {
    Write-Host "Frontend build is present and ready." -ForegroundColor Green
} else {
    $npmCmd = (Get-Command npm.cmd -ErrorAction SilentlyContinue)
    if ($npmCmd) {
        Write-Host "Building Angular frontend production bundle..." -ForegroundColor Yellow
        Push-Location $frontendRoot
        try {
            $nodeModules = Join-Path $frontendRoot 'node_modules'
            if (-not (Test-Path -LiteralPath $nodeModules -PathType Container)) {
                Write-Host "Installing frontend dependencies (npm install)..."
                & npm.cmd install
            }
            Write-Host "Compiling frontend (npm run build)..."
            & npm.cmd run build
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Frontend built successfully!" -ForegroundColor Green
            } else {
                Write-Warning "Frontend build returned non-zero exit code. API will still function."
            }
        } finally {
            Pop-Location
        }
    } else {
        Write-Host "npm not found. Using API mode (Swagger docs at /docs)." -ForegroundColor Gray
    }
}

# 6. Launch Application
Write-Host "`n[6/6] Launching GateVision AI Server..." -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ("  Web Application Dashboard : http://localhost:{0}" -f $Port) -ForegroundColor Green
Write-Host ("  API Documentation (Swagger): http://localhost:{0}/docs" -f $Port) -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the application.`n" -ForegroundColor Gray

Set-Location $backendRoot
& $venvPython -m uvicorn app.main:app --host $HostAddress --port $Port --workers 1 --no-access-log
exit $LASTEXITCODE
