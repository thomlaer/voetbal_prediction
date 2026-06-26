param(
    [string]$ModelRoot = (Resolve-Path "$PSScriptRoot\..\..").Path,
    [string]$Label = (Get-Date -Format "yyyyMMdd"),
    [string]$PreviousLabel = "",
    [switch]$SkipDownload,
    [switch]$SkipOdds,
    [switch]$UpdateSoccerbase,
    [switch]$SkipSoccerbaseRefresh,
    [switch]$UseLineups,
    [int]$Simulations = 10000,
    [ValidateSet("best_default", "full_with_lineups")]
    [string]$ModelProfile = "best_default"
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

$python = Join-Path $ModelRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

Push-Location $ModelRoot
try {
    $prepareArgs = @("-X", "utf8", "github_vercel_app\tools\prepare_model_data.py", "--model-root", $ModelRoot)
    if ($ModelProfile -eq "full_with_lineups") {
        $prepareArgs += @("--include-sofifa", "--include-external-elo")
    }
    Invoke-Native $python @prepareArgs

    if (-not $SkipDownload) {
        Invoke-Native $python "github_vercel_app\tools\update_international_results.py" --model-root $ModelRoot
    }

    if (-not $SkipOdds) {
        Invoke-Native node "extract_oddsportal_worldcup2026_fixtures.mjs" `
            --output "data\extracted\oddsportal_worldcup2026_fixture_odds_schedule.csv" `
            --raw-output "data\extracted\oddsportal_worldcup2026_fixture_odds_raw.csv" `
            --report "outputs\oddsportal_worldcup2026_fixture_odds_report.csv"
    }

    Invoke-Native $python -X utf8 "github_vercel_app\tools\update_espn_worldcup_results.py" `
        --output "data\extracted\espn_worldcup2026_results.csv"

    if ($UpdateSoccerbase -or -not $SkipSoccerbaseRefresh) {
        Invoke-Native $python -X utf8 "extract_soccerbase_match_data.py" `
            --skip-errors `
            --incremental `
            --url "https://www.soccerbase.com/tournaments/tournament.sd?comp_id=73" `
            --max-tournaments 1 `
            --incremental-lookback-days 0 `
            --max-fetch-games 0 `
            --worldcup-date-pages "2026-06-11:2026-07-19" `
            --lineups-output "data\extracted\soccerbase_lineups_used.csv" `
            --stats-output "data\extracted\soccerbase_match_stats.csv" `
            --cards-output "data\extracted\soccerbase_cards_events.csv" `
            --report-output "outputs\soccerbase_extraction_report.csv"
    }

    $modelOut = "outputs_worldcup2026_cards_$Label"
    $modelDir = "models_worldcup2026_cards_$Label"
    $drawOut = "outputs_worldcup2026_cards_draw05_$Label"
    $trainArgs = @(
        "-X", "utf8", "train_xgboost_worldcup.py",
        "--skip-download",
        "--use-soccerbase-stat-features",
        "--use-soccerbase-card-features",
        "--future-fixtures", "data\extracted\oddsportal_worldcup2026_fixture_odds_schedule.csv",
        "--output-dir", $modelOut,
        "--model-dir", $modelDir
    )

    if ($UseLineups -or $ModelProfile -eq "full_with_lineups") {
        $trainArgs += @("--use-soccerbase-lineup-features", "--use-soccerbase-ratings")
    }
    if ($ModelProfile -eq "full_with_lineups") {
        $trainArgs += @(
            "--use-sofifa-yearly-ratings",
            "--use-squad-market-values",
            "--use-squad-sofifa-ratings",
            "--use-soccerbase-stat-recency-features",
            "--use-geo-features",
            "--use-external-elo-features"
        )
        if (Test-Path "data\extracted\oddsportal_international_closing_1x2.csv") {
            $trainArgs += @("--odds-csv", "data\extracted\oddsportal_international_closing_1x2.csv")
        }
    }

    Invoke-Native $python @trainArgs

    New-Item -ItemType Directory -Force -Path $drawOut | Out-Null
    Invoke-Native $python -X utf8 "apply_prediction_tweaks.py" `
        --predictions "$modelOut\future_predictions_xgboost.csv" `
        --schedule "data\extracted\oddsportal_worldcup2026_fixture_odds_schedule.csv" `
        --output "$drawOut\future_predictions_xgboost_draw05.csv" `
        --template-output "data\extracted\manual_prediction_tweaks_template_cards_draw05_$Label.csv" `
        --draw-multiplier 0.50 `
        --goal-multiplier 1.00

    Invoke-Native $python -X utf8 "predict_worldcup2026_montecarlo.py" `
        --model-predictions "$drawOut\future_predictions_xgboost_draw05.csv" `
        --schedule "data\extracted\oddsportal_worldcup2026_fixture_odds_schedule.csv" `
        --output-dir $drawOut `
        --simulations $Simulations `
        --seed 42

    Invoke-Native $python -X utf8 "github_vercel_app\tools\build_expected_group_tables.py" `
        --schedule "data\extracted\oddsportal_worldcup2026_fixture_odds_schedule.csv" `
        --team-probabilities "$drawOut\worldcup2026_montecarlo_team_probabilities.csv" `
        --output "$drawOut\worldcup2026_expected_group_tables.csv"

    Invoke-Native $python -X utf8 "predict_worldcup2026_bracket.py" `
        --output-dir $drawOut `
        --model-predictions "$drawOut\future_predictions_xgboost_draw05.csv" `
        --schedule "data\extracted\oddsportal_worldcup2026_fixture_odds_schedule.csv"

    Invoke-Native $python -X utf8 "make_scorito_worldcup_picks.py" `
        --group-predictions "$drawOut\worldcup2026_group_match_predictions.csv" `
        --bracket "$drawOut\worldcup2026_bracket_prediction.csv" `
        --team-probabilities "$drawOut\worldcup2026_montecarlo_team_probabilities.csv" `
        --output-dir $drawOut

    Invoke-Native $python -X utf8 "generate_worldcup_excel.py" `
        --output-dir $drawOut `
        --schedule "data\extracted\oddsportal_worldcup2026_fixture_odds_schedule.csv"

    $publishArgs = @(
        "-X", "utf8", "github_vercel_app\tools\publish_latest_outputs.py",
        "--model-root", $ModelRoot,
        "--run-dir", (Join-Path $ModelRoot $drawOut)
    )
    if ($PreviousLabel) {
        $publishArgs += @("--previous-label", $PreviousLabel)
    }
    Invoke-Native $python @publishArgs
}
finally {
    Pop-Location
}
