#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$(cd "$APP_DIR/.." && pwd)}"
LABEL="${LABEL:-$(date +%Y%m%d)}"
PREVIOUS_LABEL="${PREVIOUS_LABEL:-}"
SIMULATIONS="${SIMULATIONS:-10000}"
PYTHON="${PYTHON:-python}"
MODEL_PROFILE="${MODEL_PROFILE:-best_default}"

cd "$MODEL_ROOT"

PREPARE_ARGS=(--model-root "$MODEL_ROOT")
if [[ "$MODEL_PROFILE" == "full_with_lineups" ]]; then
  PREPARE_ARGS+=(--include-sofifa --include-external-elo)
fi
"$PYTHON" -X utf8 github_vercel_app/tools/prepare_model_data.py "${PREPARE_ARGS[@]}"

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  "$PYTHON" github_vercel_app/tools/update_international_results.py --model-root "$MODEL_ROOT"
fi

if [[ "${SKIP_ODDS:-0}" != "1" ]]; then
  if ! node extract_oddsportal_worldcup2026_fixtures.mjs \
    --output data/extracted/oddsportal_worldcup2026_fixture_odds_schedule.csv \
    --raw-output data/extracted/oddsportal_worldcup2026_fixture_odds_raw.csv \
    --report outputs/oddsportal_worldcup2026_fixture_odds_report.csv; then
    if [[ -f data/extracted/oddsportal_worldcup2026_fixture_odds_schedule.csv ]]; then
      echo "WARNING: OddsPortal refresh failed; continuing with existing fixture odds schedule."
    else
      echo "ERROR: OddsPortal refresh failed and no existing odds schedule is available."
      exit 1
    fi
  fi
fi

if [[ "${SKIP_ESPN:-0}" != "1" ]]; then
  "$PYTHON" -X utf8 github_vercel_app/tools/update_espn_worldcup_results.py \
    --output data/extracted/espn_worldcup2026_results.csv
fi

TRAINING_RESULTS="data/extracted/results_training_with_espn.csv"
"$PYTHON" -X utf8 github_vercel_app/tools/build_training_results_with_espn.py \
  --base data/results.csv \
  --espn data/extracted/espn_worldcup2026_results.csv \
  --output "$TRAINING_RESULTS"

if [[ "${UPDATE_SOCCERBASE:-1}" == "1" ]]; then
  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    SOCCERBASE_URLS="${SOCCERBASE_URLS:-https://www.soccerbase.com/tournaments/tournament.sd?comp_id=73}"
    SOCCERBASE_LOOKBACK_DAYS="${SOCCERBASE_LOOKBACK_DAYS:-14}"
    SOCCERBASE_MAX_TOURNAMENTS="${SOCCERBASE_MAX_TOURNAMENTS:-1}"
    SOCCERBASE_MAX_FETCH_GAMES="${SOCCERBASE_MAX_FETCH_GAMES:-50}"
  fi
  SOCCERBASE_ARGS=(
    --skip-errors \
    --incremental \
    --lineups-output data/extracted/soccerbase_lineups_used.csv \
    --stats-output data/extracted/soccerbase_match_stats.csv \
    --cards-output data/extracted/soccerbase_cards_events.csv \
    --report-output outputs/soccerbase_extraction_report.csv
  )
  if [[ -n "${SOCCERBASE_URLS:-}" ]]; then
    IFS='|' read -r -a SOCCERBASE_URL_LIST <<< "$SOCCERBASE_URLS"
    for url in "${SOCCERBASE_URL_LIST[@]}"; do
      SOCCERBASE_ARGS+=(--url "$url")
    done
  fi
  if [[ -n "${SOCCERBASE_LOOKBACK_DAYS:-}" ]]; then
    SOCCERBASE_ARGS+=(--incremental-lookback-days "$SOCCERBASE_LOOKBACK_DAYS")
  fi
  if [[ -n "${SOCCERBASE_MAX_TOURNAMENTS:-}" ]]; then
    SOCCERBASE_ARGS+=(--max-tournaments "$SOCCERBASE_MAX_TOURNAMENTS")
  fi
  if [[ -n "${SOCCERBASE_MAX_FETCH_GAMES:-}" ]]; then
    SOCCERBASE_ARGS+=(--max-fetch-games "$SOCCERBASE_MAX_FETCH_GAMES")
  fi
  if [[ -n "${SOCCERBASE_WORLDCUP_DATE_PAGES:-}" ]]; then
    SOCCERBASE_ARGS+=(--worldcup-date-pages "$SOCCERBASE_WORLDCUP_DATE_PAGES")
  fi
  echo "Soccerbase refresh args: ${SOCCERBASE_ARGS[*]}"
  "$PYTHON" -X utf8 extract_soccerbase_match_data.py "${SOCCERBASE_ARGS[@]}"
fi

MODEL_OUT="outputs_worldcup2026_cards_${LABEL}"
MODEL_DIR="models_worldcup2026_cards_${LABEL}"
DRAW_OUT="outputs_worldcup2026_cards_draw05_${LABEL}"
EXTRA_TRAIN_ARGS=()
if [[ "${USE_LINEUPS:-0}" == "1" || "$MODEL_PROFILE" == "full_with_lineups" ]]; then
  EXTRA_TRAIN_ARGS+=(--use-soccerbase-lineup-features --use-soccerbase-ratings)
fi
if [[ "$MODEL_PROFILE" == "full_with_lineups" ]]; then
  EXTRA_TRAIN_ARGS+=(
    --use-sofifa-yearly-ratings
    --use-squad-market-values
    --use-squad-sofifa-ratings
    --use-soccerbase-stat-recency-features
    --use-geo-features
    --use-external-elo-features
  )
  if [[ -f data/extracted/oddsportal_international_closing_1x2.csv ]]; then
    EXTRA_TRAIN_ARGS+=(--odds-csv data/extracted/oddsportal_international_closing_1x2.csv)
  fi
fi

"$PYTHON" -X utf8 train_xgboost_worldcup.py \
  --skip-download \
  --results "$TRAINING_RESULTS" \
  --use-soccerbase-stat-features \
  --use-soccerbase-card-features \
  "${EXTRA_TRAIN_ARGS[@]}" \
  --future-fixtures data/extracted/oddsportal_worldcup2026_fixture_odds_schedule.csv \
  --output-dir "$MODEL_OUT" \
  --model-dir "$MODEL_DIR"

mkdir -p "$DRAW_OUT"
"$PYTHON" -X utf8 apply_prediction_tweaks.py \
  --predictions "$MODEL_OUT/future_predictions_xgboost.csv" \
  --schedule data/extracted/oddsportal_worldcup2026_fixture_odds_schedule.csv \
  --output "$DRAW_OUT/future_predictions_xgboost_draw05.csv" \
  --template-output "data/extracted/manual_prediction_tweaks_template_cards_draw05_${LABEL}.csv" \
  --draw-multiplier 0.50 \
  --goal-multiplier 1.00

"$PYTHON" -X utf8 predict_worldcup2026_montecarlo.py \
  --model-predictions "$DRAW_OUT/future_predictions_xgboost_draw05.csv" \
  --schedule data/extracted/oddsportal_worldcup2026_fixture_odds_schedule.csv \
  --output-dir "$DRAW_OUT" \
  --simulations "$SIMULATIONS" \
  --seed 42

"$PYTHON" -X utf8 github_vercel_app/tools/build_expected_group_tables.py \
  --schedule data/extracted/oddsportal_worldcup2026_fixture_odds_schedule.csv \
  --team-probabilities "$DRAW_OUT/worldcup2026_montecarlo_team_probabilities.csv" \
  --output "$DRAW_OUT/worldcup2026_expected_group_tables.csv"

"$PYTHON" -X utf8 predict_worldcup2026_bracket.py \
  --output-dir "$DRAW_OUT" \
  --model-predictions "$DRAW_OUT/future_predictions_xgboost_draw05.csv" \
  --schedule data/extracted/oddsportal_worldcup2026_fixture_odds_schedule.csv

"$PYTHON" -X utf8 make_scorito_worldcup_picks.py \
  --group-predictions "$DRAW_OUT/worldcup2026_group_match_predictions.csv" \
  --bracket "$DRAW_OUT/worldcup2026_bracket_prediction.csv" \
  --team-probabilities "$DRAW_OUT/worldcup2026_montecarlo_team_probabilities.csv" \
  --output-dir "$DRAW_OUT"

"$PYTHON" -X utf8 generate_worldcup_excel.py \
  --output-dir "$DRAW_OUT" \
  --schedule data/extracted/oddsportal_worldcup2026_fixture_odds_schedule.csv

"$PYTHON" -X utf8 github_vercel_app/tools/publish_latest_outputs.py \
  --model-root "$MODEL_ROOT" \
  --run-dir "$MODEL_ROOT/$DRAW_OUT" \
  --previous-label "$PREVIOUS_LABEL"
