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
    exit $LASTEXITCODE
} else {
    throw "Cannot find start.ps1 at $targetScript"
}
