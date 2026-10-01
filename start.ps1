[CmdletBinding()]
param(
    [string]$HostAddress = '127.0.0.1',
    [ValidateRange(1, 65535)]
    [int]$Port = 8001,
    [switch]$BuildFrontend,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$launcherRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendRoot = Join-Path $launcherRoot 'backend'
$frontendRoot = Join-Path $launcherRoot 'frontend'
$backendPython = Join-Path $backendRoot '.venv\Scripts\python.exe'
$frontendIndex = Join-Path $frontendRoot 'dist\gate\browser\index.html'

# Ensure root snapshots junction points to backend snapshots
$snapshotsRoot = Join-Path $launcherRoot 'snapshots'
$backendSnapshots = Join-Path $backendRoot 'snapshots'
if (-not (Test-Path -LiteralPath $snapshotsRoot)) {
    try {
        New-Item -ItemType Junction -Path $snapshotsRoot -Target $backendSnapshots -ErrorAction SilentlyContinue | Out-Null
    } catch {}
}

if (-not (Test-Path -LiteralPath $backendPython -PathType Leaf)) {
    throw 'Backend environment is missing. Create backend\.venv and install backend\requirements.txt first.'
}

if ($BuildFrontend -or -not (Test-Path -LiteralPath $frontendIndex -PathType Leaf)) {
    if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
        throw 'The Angular build is missing and npm.cmd is unavailable. Install Node.js, run npm ci in frontend, then retry.'
    }
    Push-Location $frontendRoot
    try {
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Angular production build failed.' }
    }
    finally {
        Pop-Location
    }
}

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
    Write-Host ("Port {0} is currently in use. Automatically switched to available port: {1}" -f $originalPort, $Port) -ForegroundColor Yellow
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

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($url)
        Start-Sleep -Seconds 2
        try { Start-Process $url } catch {}
    } -ArgumentList $dashboardUrl | Out-Null
}

Set-Location $backendRoot
& $backendPython -m uvicorn app.main:app --host $HostAddress --port $Port --workers 1 --no-access-log
exit $LASTEXITCODE
