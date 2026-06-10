param(
    [string]$Label = (Get-Date -Format "yyyyMMdd"),
    [string]$PreviousLabel = "",
    [int]$Simulations = 10000,
    [ValidateSet("best_default", "full_with_lineups")]
    [string]$ModelProfile = "best_default",
    [switch]$SkipDownload,
    [switch]$SkipOdds,
    [switch]$UpdateSoccerbase,
    [switch]$UseLineups,
    [switch]$Deploy,
    [switch]$PrebuiltDeploy
)

$ErrorActionPreference = "Stop"

$rebuildArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "$PSScriptRoot\rebuild_predictions.ps1",
    "-Label", $Label,
    "-Simulations", $Simulations,
    "-ModelProfile", $ModelProfile
)

if ($PreviousLabel) {
    $rebuildArgs += @("-PreviousLabel", $PreviousLabel)
}
if ($SkipDownload) {
    $rebuildArgs += "-SkipDownload"
}
if ($SkipOdds) {
    $rebuildArgs += "-SkipOdds"
}
if ($UpdateSoccerbase) {
    $rebuildArgs += "-UpdateSoccerbase"
}
if ($UseLineups) {
    $rebuildArgs += "-UseLineups"
}

& powershell.exe @rebuildArgs
if ($LASTEXITCODE -ne 0) {
    throw "Rebuild failed with exit code $LASTEXITCODE"
}

if ($Deploy) {
    $deployArgs = @("-ExecutionPolicy", "Bypass", "-File", "$PSScriptRoot\deploy_vercel.ps1")
    if ($PrebuiltDeploy) {
        $deployArgs += "-Prebuilt"
    }
    & powershell.exe @deployArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Deploy failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "Rebuild complete. Live Vercel site was not changed because -Deploy was not set."
}
