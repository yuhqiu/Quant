<#
.SYNOPSIS
    Verify the local environment can run the pipeline.
.DESCRIPTION
    Checks the interpreter, the required packages, the data lake and the feature
    panel. Exits non-zero when anything essential is missing, so it is safe to use
    as a gate in front of a scheduled run.
#>
[CmdletBinding()]
param(
    [switch]$Strict
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'

$problems = New-Object System.Collections.Generic.List[string]

function Write-Check {
    param([string]$Name, [bool]$Ok, [string]$Detail = '')
    $mark = if ($Ok) { 'OK  ' } else { 'FAIL' }
    $colour = if ($Ok) { 'Green' } else { 'Red' }
    Write-Host ("[{0}] {1}{2}" -f $mark, $Name.PadRight(34), $Detail) -ForegroundColor $colour
    if (-not $Ok) { $script:problems.Add($Name) }
}

Write-Host "`n== interpreter ==" -ForegroundColor Cyan

if (-not (Test-Path $python)) {
    Write-Check 'virtual environment' $false "expected $python"
    Write-Host "`nCreate it with: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}
$version = & $python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
$major, $minor = $version.Split('.')[0..1]
Write-Check 'python >= 3.12' ([int]$major -gt 3 -or ([int]$major -eq 3 -and [int]$minor -ge 12)) "found $version"

Write-Host "`n== packages ==" -ForegroundColor Cyan
$required = @('pandas', 'pyarrow', 'duckdb', 'numpy')
$optional = @('yfinance', 'scipy', 'matplotlib', 'pytest')

foreach ($package in $required + $optional) {
    $found = & $python -c "
import importlib.metadata as m
try:
    print(m.version('$package'))
except Exception:
    print('')
"
    $ok = -not [string]::IsNullOrWhiteSpace($found)
    if ($ok) {
        Write-Check $package $true "v$found"
    }
    elseif ($required -contains $package) {
        Write-Check $package $false 'missing (pip install -e ".[all]")'
    }
    else {
        Write-Host ("[SKIP] {0}{1}" -f $package.PadRight(34), 'optional, not installed') -ForegroundColor DarkGray
    }
}

Write-Host "`n== project ==" -ForegroundColor Cyan
& $python -c "import Common, DataAcquisition, MetricsGeneration, Signals, Portfolio, Strategy, Evaluation" 2>&1 | Out-Null
Write-Check 'all modules importable' ($LASTEXITCODE -eq 0)

$summary = & $python -c @"
from Common import settings
from Common.types import Partition
from DataAcquisition import stored_symbols
from MetricsGeneration import read_manifest

resolved = settings()
partition = Partition(resolved.region, resolved.asset_class, resolved.interval)
manifest = read_manifest(partition.metrics_dir) or {}
print(resolved.data_root)
print(len(stored_symbols(partition)))
print(manifest.get('symbols', 0))
print(manifest.get('end', '-'))
"@

$lines = $summary -split "`r?`n"
Write-Check 'data root' (Test-Path $lines[0]) $lines[0]
Write-Check 'bars in the lake' ([int]$lines[1] -gt 0) "$($lines[1]) symbols"
$panelOk = [int]$lines[2] -gt 0
if ($panelOk) {
    Write-Check 'feature panel built' $true "$($lines[2]) symbols to $($lines[3])"
}
elseif ($Strict) {
    Write-Check 'feature panel built' $false 'run: python -m MetricsGeneration build'
}
else {
    Write-Host ("[SKIP] {0}{1}" -f 'feature panel built'.PadRight(34), 'not built yet') -ForegroundColor DarkGray
}

Write-Host ''
if ($problems.Count -gt 0) {
    Write-Host ("{0} check(s) failed: {1}" -f $problems.Count, ($problems -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'environment ready' -ForegroundColor Green
exit 0
