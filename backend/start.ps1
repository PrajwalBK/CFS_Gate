[CmdletBinding()]
param(
    [string]$HostAddress = '127.0.0.1',
    [ValidateRange(1, 65535)]
    [int]$Port = 8001,
    [switch]$BuildFrontend
)

$targetScript = Join-Path (Split-Path -Parent $PSScriptRoot) 'start.ps1'
if (Test-Path -LiteralPath $targetScript -PathType Leaf) {
    & $targetScript @PSBoundParameters
    if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq 1 -or $LASTEXITCODE -eq 0x40010004) {
        exit 0
    }
    exit $LASTEXITCODE
} else {
    throw "Cannot find start.ps1 at $targetScript"
}
