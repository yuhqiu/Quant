<#
.SYNOPSIS
    Daily pipeline: acquire -> quality -> features -> backtest -> report.
.DESCRIPTION
    Every stage must exit zero or the run stops, because a report built on a
    failed download is worse than no report at all.
.EXAMPLE
    .\scripts\run-pipeline.ps1 -Strategy momentum
.EXAMPLE
    .\scripts\run-pipeline.ps1 -SkipDownload -Rebuild
#>
[CmdletBinding()]
param(
    [string]$Strategy = 'momentum',
    [string]$Interval = '1d',
    [string]$AssetClass = 'stock',
    [switch]$SkipDownload,
    [switch]$Rebuild,
    [switch]$SkipBacktest
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
Set-Location $root

if (-not (Test-Path $python)) { throw "no virtual environment at $python" }

$stage = 0
function Invoke-Stage {
    param([string]$Name, [string[]]$Arguments)
    $script:stage++
    Write-Host ("`n=== [{0}] {1} ===" -f $script:stage, $Name) -ForegroundColor Cyan
    Write-Host ("    python {0}" -f ($Arguments -join ' ')) -ForegroundColor DarkGray

    $started = Get-Date
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("stage failed: {0} (exit {1})" -f $Name, $LASTEXITCODE) -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host ("    done in {0:N1}s" -f ((Get-Date) - $started).TotalSeconds) -ForegroundColor DarkGray
}

Write-Host "`n=== [0] environment ===" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot 'check-env.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipDownload) {
    Invoke-Stage 'universe' @('-m', 'DataAcquisition', 'universe')
    Invoke-Stage 'update bars' @('-m', 'DataAcquisition', 'update', '--interval', $Interval)
    Invoke-Stage 'quality report' @('-m', 'DataAcquisition', 'quality', '--interval', $Interval, '--asset-class', $AssetClass)
}

$buildArgs = @('-m', 'MetricsGeneration', 'build', '--interval', $Interval, '--asset-class', $AssetClass)
if (-not $Rebuild) { $buildArgs += '--incremental' }
Invoke-Stage 'feature panel' $buildArgs

if (-not $SkipBacktest) {
    Invoke-Stage 'backtest' @('-m', 'Strategy', 'backtest', '--spec', $Strategy)
    Invoke-Stage 'evaluation' @('-m', 'Evaluation', 'report', '--strategy', $Strategy, '--run', 'latest')
}

Write-Host "`npipeline complete" -ForegroundColor Green
exit 0
