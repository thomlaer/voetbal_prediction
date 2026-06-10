param(
    [bool]$Production = $true,
    [switch]$Prebuilt,
    [switch]$SkipBuild,
    [string]$Token = $env:VERCEL_TOKEN
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

function Invoke-Native {
    $command = $args[0]
    $commandArgs = @()
    if ($args.Count -gt 1) {
        $commandArgs = $args[1..($args.Count - 1)]
    }
    & $command @commandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($args -join ' ')"
    }
}

$appDir = (Resolve-Path "$PSScriptRoot\..").Path
Push-Location $appDir
try {
    if (-not $SkipBuild) {
        if ($Prebuilt) {
            $buildArgs = @("vercel", "build")
            if ($Production) {
                $buildArgs += "--prod"
            }
            if ($Token) {
                $buildArgs += @("--token", $Token)
            }
            Invoke-Native npx.cmd @buildArgs
        }
        else {
            Invoke-Native npm.cmd "run" "build"
        }
    }

    $deployArgs = @("vercel", "deploy", "--yes")
    if ($Production) {
        $deployArgs += "--prod"
    }
    if ($Prebuilt) {
        $deployArgs += "--prebuilt"
    }
    if ($Token) {
        $deployArgs += @("--token", $Token)
    }

    Invoke-Native npx.cmd @deployArgs
}
finally {
    Pop-Location
}
