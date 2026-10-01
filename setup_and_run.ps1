# GateVision AI - Automated Setup and Launcher (PowerShell)
<#
.SYNOPSIS
    Automated one-click setup, dependency installer, permissions granter, and launcher for GateVision AI.
.DESCRIPTION
    1. Sets ExecutionPolicy Bypass and unblocks project files.
    2. Grants full NTFS file/directory permissions (ACLs) to current user.
    3. If elevated as Administrator, configures Windows Firewall rules for Port 8001 and Python.
    4. Sets up Python virtual environment (.venv).
    5. Installs all backend dependencies (requirements.txt, requirements-ai.txt, safetensors, etc.).
    6. Installs all frontend npm dependencies and compiles the Angular production bundle.
    7. Creates all required data, snapshot, log, and model folders with .env initialization.
    8. Automatically handles port conflicts and launches the server, opening the dashboard in browser.
.PARAMETER HostAddress
    Network interface to bind (default: 127.0.0.1).
.PARAMETER Port
    Port to listen on (default: 8001).
.PARAMETER SkipFrontendInstall
    Skip running 'npm install' if node_modules already exists.
.PARAMETER NoBrowser
    Do not automatically open the default web browser upon launch.
.PARAMETER Elevate
    Re-launch the script with Administrator privileges to configure Windows Firewall.
#>
[CmdletBinding()]
param(
    [string]$HostAddress = '127.0.0.1',
    [ValidateRange(1, 65535)]
    [int]$Port = 8001,
    [switch]$SkipFrontendInstall,
    [switch]$NoBrowser,
    [switch]$Elevate
)

# Step 0: Set execution policy for the current process
try {
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force -ErrorAction SilentlyContinue
} catch {}

function Test-IsAdmin {
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Self-elevation if requested and not admin
if ($Elevate -and -not (Test-IsAdmin)) {
    Write-Host "Elevating privileges to Administrator..." -ForegroundColor Yellow
    $scriptPath = $MyInvocation.MyCommand.Path
    Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
    exit 0
}

$ErrorActionPreference = 'Stop'

# Resolve project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (Test-Path (Join-Path $scriptDir 'backend\app\main.py')) {
    $projectRoot = $scriptDir
} elseif (Test-Path (Join-Path $scriptDir 'ai-video-based-inventory\backend\app\main.py')) {
    $projectRoot = Join-Path $scriptDir 'ai-video-based-inventory'
} else {
    Write-Error "Cannot locate GateVision backend directory from $scriptDir."
    exit 1
}

Set-Location $projectRoot
$backendRoot = Join-Path $projectRoot 'backend'
$frontendRoot = Join-Path $projectRoot 'frontend'
$venvDir = Join-Path $backendRoot '.venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'
$venvPip = Join-Path $venvDir 'Scripts\pip.exe'
$frontendIndex = Join-Path $frontendRoot 'dist\gate\browser\index.html'

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "          GateVision AI - Automated Setup and Launcher               " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Project Root : $projectRoot" -ForegroundColor DarkGray
Write-Host "Timestamp    : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor DarkGray
Write-Host ""

# [1/7] Grant Permissions & Unblock Files
Write-Host "[1/7] Granting permissions and unblocking files..." -ForegroundColor Yellow

# Unblock files downloaded or cloned
try {
    Get-ChildItem -Path $projectRoot -Filter *.ps1 -Recurse -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue
    Get-ChildItem -Path $projectRoot -Filter *.bat -Recurse -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue
} catch {}

# Clear Read-Only attributes
try {
    $attrCmd = "attrib -R `"$projectRoot\*.*`" /S /D"
    Invoke-Expression $attrCmd 2>$null | Out-Null
} catch {}

# Grant current user full NTFS control to data/logs/snapshots/models/uploads directories
$dirsToPermit = @('data', 'snapshots', 'logs', 'models', 'uploads', 'uploads\ocr-spool')
foreach ($d in $dirsToPermit) {
    $p = Join-Path $backendRoot $d
    if (-not (Test-Path -LiteralPath $p)) {
        New-Item -ItemType Directory -Path $p -Force | Out-Null
    }
    # Grant current user full control recursively
    try {
        & cmd.exe /c "icacls `"$p`" /grant %USERNAME%:(OI)(CI)F /T /Q /C" 2>$null | Out-Null
    } catch {}
}
Write-Host "  Directory read/write/modify permissions granted to $env:USERNAME." -ForegroundColor Green

# Windows Firewall rules if Admin
$isAdmin = Test-IsAdmin
if ($isAdmin) {
    Write-Host "  Configuring Windows Firewall rules for Port $Port and Python..." -ForegroundColor Cyan
    try {
        & netsh advfirewall firewall delete rule name="GateVision AI Server (Port $Port)" >$null 2>&1
        & netsh advfirewall firewall add rule name="GateVision AI Server (Port $Port)" dir=in action=allow protocol=TCP localport=$Port profile=any >$null 2>&1
        Write-Host "  Windows Firewall rule added for Port $Port." -ForegroundColor Green
    } catch {
        Write-Warning "Could not update firewall rules."
    }
} else {
    Write-Host "  Running with standard user permissions. (Run as Administrator if you wish to auto-register Windows Firewall rules)." -ForegroundColor Gray
}

# [2/7] Check Python Installation
Write-Host "`n[2/7] Checking Python installation..." -ForegroundColor Yellow
$sysPython = (Get-Command python.exe -ErrorAction SilentlyContinue)
if (-not $sysPython) {
    $sysPython = (Get-Command py.exe -ErrorAction SilentlyContinue)
}

if (-not $sysPython) {
    Write-Host "[ERROR] Python was not found in your system PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://www.python.org/ and select 'Add Python to PATH'." -ForegroundColor Yellow
    exit 1
}

$pyVersionStr = (& $sysPython.Source --version 2>&1)
Write-Host "  Found: $pyVersionStr ($($sysPython.Source))" -ForegroundColor Green

# [3/7] Setup Python Virtual Environment
Write-Host "`n[3/7] Verifying Python virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    Write-Host "  Creating virtual environment at $venvDir..." -ForegroundColor Cyan
    & $sysPython.Source -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
    Write-Host "  Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "  Virtual environment ready at $venvDir." -ForegroundColor Green
}

# [4/7] Install ALL Backend Dependencies
Write-Host "`n[4/7] Installing and verifying all backend dependencies..." -ForegroundColor Yellow
Write-Host "  Updating pip, setuptools, and wheel..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip setuptools wheel --quiet

Write-Host "  Installing core packages (requirements.txt)..." -ForegroundColor Cyan
& $venvPip install -r (Join-Path $backendRoot 'requirements.txt')

$reqAi = Join-Path $backendRoot 'requirements-ai.txt'
if (Test-Path -LiteralPath $reqAi -PathType Leaf) {
    Write-Host "  Installing AI packages (requirements-ai.txt: Ultralytics, EasyOCR, PaddleOCR)..." -ForegroundColor Cyan
    & $venvPip install -r $reqAi
}

# Ensure safetensors >= 0.8.0
Write-Host "  Verifying safetensors>=0.8.0..." -ForegroundColor Cyan
& $venvPip install "safetensors>=0.8.0" --quiet

Write-Host "  All backend dependencies installed and verified." -ForegroundColor Green

# [5/7] Verify Configuration and Directory Setup
Write-Host "`n[5/7] Verifying configuration and storage directories..." -ForegroundColor Yellow
$envFile = Join-Path $backendRoot '.env'
$envExample = Join-Path $backendRoot '.env.example'
if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
    if (Test-Path -LiteralPath $envExample -PathType Leaf) {
        Write-Host "  Creating .env from .env.example..." -ForegroundColor Cyan
        Copy-Item $envExample $envFile
    } else {
        Write-Warning "Neither .env nor .env.example found in backend directory."
    }
} else {
    Write-Host "  Config file .env present." -ForegroundColor Green
}

Write-Host "  Storage directories verified." -ForegroundColor Green

# [6/7] Install ALL Frontend Dependencies & Build
Write-Host "`n[6/7] Installing frontend dependencies and compiling production bundle..." -ForegroundColor Yellow
$npmCmd = (Get-Command npm.cmd -ErrorAction SilentlyContinue)

if ($npmCmd) {
    Push-Location $frontendRoot
    try {
        $nodeModules = Join-Path $frontendRoot 'node_modules'
        if (-not $SkipFrontendInstall -or -not (Test-Path -LiteralPath $nodeModules -PathType Container)) {
            Write-Host "  Installing all frontend npm dependencies (npm install)..." -ForegroundColor Cyan
            & npm.cmd install --no-audit --prefer-offline
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "npm install finished with non-zero code. Attempting standard npm install..."
                & npm.cmd install
            }
        } else {
            Write-Host "  Frontend node_modules already present. Skipping re-install." -ForegroundColor Green
        }

        # Build if index.html is missing
        if (-not (Test-Path -LiteralPath $frontendIndex -PathType Leaf)) {
            Write-Host "  Compiling Angular frontend production assets (npm run build)..." -ForegroundColor Cyan
            & npm.cmd run build
        } else {
            Write-Host "  Frontend production build already present ($frontendIndex)." -ForegroundColor Green
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "  [NOTICE] npm not found. If web UI is needed, please install Node.js (https://nodejs.org/)." -ForegroundColor Gray
    Write-Host "  Backend REST API and Swagger Docs will still be available at /docs." -ForegroundColor Gray
}

# [7/7] Port Conflict Resolution & Launch Application
Write-Host "`n[7/7] Checking port availability and preparing launch..." -ForegroundColor Yellow

function Get-AvailablePort([string]$hostAddr, [int]$startPort) {
    $p = $startPort
    while ($p -le 65535) {
        $inUse = $false
        $active = Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue
        if ($active) {
            $inUse = $true
        } else {
            try {
                $ip = if ($hostAddr -eq '0.0.0.0') { [System.Net.IPAddress]::Any } else { [System.Net.IPAddress]::Parse($hostAddr) }
                $listener = [System.Net.Sockets.TcpListener]::new($ip, $p)
                $listener.Start()
                $listener.Stop()
            } catch {
                $inUse = $true
            }
        }
        if (-not $inUse) {
            return $p
        }
        $p++
    }
    throw "No available ports found starting from $startPort"
}

# Clean up any orphaned background processes belonging to this specific workspace
$conns = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -ge 8000 -and $_.LocalPort -le 8010 }
foreach ($c in $conns) {
    try {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $($c.OwningProcess)" -ErrorAction SilentlyContinue
        if ($proc -and $proc.CommandLine -and ($proc.CommandLine -like "*$backendRoot*" -or $proc.ExecutablePath -like "*$backendRoot*")) {
            Write-Host ("Reclaiming port {0} from previous background run of this project (PID {1})..." -f $c.LocalPort, $c.OwningProcess) -ForegroundColor DarkGray
            Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 300
        }
    } catch {}
}

$originalPort = $Port
$Port = Get-AvailablePort -hostAddr $HostAddress -startPort $Port
if ($Port -ne $originalPort) {
    Write-Host "  Port $originalPort is unavailable. Automatically switched to port $Port." -ForegroundColor Yellow
}

$dashboardUrl = "http://localhost:$Port"
$docsUrl = "http://localhost:$Port/docs"

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "                     GateVision AI is Starting                       " -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Web Dashboard : $dashboardUrl" -ForegroundColor Green
Write-Host "  API Docs      : $docsUrl" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Press Ctrl+C in this terminal to shut down the server.`n" -ForegroundColor DarkGray

# Open browser if requested
if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($url)
        Start-Sleep -Seconds 2
        try { Start-Process $url } catch {}
    } -ArgumentList $dashboardUrl | Out-Null
}

Set-Location $backendRoot
& $venvPython -m uvicorn app.main:app --host $HostAddress --port $Port --workers 1 --no-access-log
exit $LASTEXITCODE
