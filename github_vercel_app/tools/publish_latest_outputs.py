#!/usr/bin/env python
"""Publish the newest prediction outputs as compact files for the Vercel app."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = APP_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build public/data/dashboard.json from model outputs.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--previous-label", default="")
    return parser.parse_args()


def read_csv(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: convert_value(value) for key, value in row.items()})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def convert_value(value: str | None) -> Any:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value
    if re.fullmatch(r"-?(\d+\.\d*|\d*\.\d+)(e-?\d+)?", value, flags=re.I):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def newest_run(model_root: Path) -> Path:
    candidates = [
        p
        for p in model_root.glob("outputs_worldcup2026_cards_draw05_*")
        if p.is_dir() and re.search(r"\d{8}$", p.name)
    ]
    if not candidates:
        fallback = model_root / "outputs_worldcup2026_cards_draw05"
        if fallback.exists():
            return fallback
        raise FileNotFoundError("No outputs_worldcup2026_cards_draw05_* directory found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def file_status(name: str, path: Path, note: str = "") -> dict[str, Any]:
    if not path.exists():
        return {"name": name, "status": "missing", "note": note or str(path)}
    rows = None
    if path.suffix.lower() == ".csv":
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = max(0, sum(1 for _ in handle) - 1)
        except UnicodeDecodeError:
            rows = None
    return {
        "name": name,
        "status": "ok",
        "rows": rows,
        "last_modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "note": note,
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def has_score_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "na", "nan", "none", "null"}
    return True


def copy_if_exists(source: Path, destination: Path) -> str:
    if not source.exists():
        return ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    public_root = APP_ROOT / "public"
    return "/" + destination.relative_to(public_root).as_posix()


def copy_first_existing(sources: list[Path], destination: Path) -> str:
    for source in sources:
        if source.exists():
            return copy_if_exists(source, destination)
    return ""


def write_csv(rows: list[dict[str, Any]], destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    public_root = APP_ROOT / "public"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return "/" + destination.relative_to(public_root).as_posix()


def compact_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "match_number",
        "date",
        "stage",
        "group",
        "home_team",
        "away_team",
        "score",
        "home_score",
        "away_score",
        "predicted_winner",
        "confidence",
        "safe_score",
        "upside_score",
        "recommended_rule",
        "model_favourite_prob",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
    ]
    return [{field: row.get(field, "") for field in fields} for row in rows]


def build_changes_from_previous(
    previous_predictions: list[dict[str, Any]], current_predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not previous_predictions:
        return []
    previous_by_number = {
        str(row.get("match_number", "")): row
        for row in previous_predictions
        if str(row.get("match_number", ""))
    }
    changes: list[dict[str, Any]] = []
    for row in current_predictions:
        match_number = str(row.get("match_number", ""))
        old = previous_by_number.get(match_number)
        if not old:
            continue
        watched = ["home_team", "away_team", "score", "predicted_winner", "confidence"]
        if all(str(old.get(field, "")) == str(row.get(field, "")) for field in watched):
            continue
        changes.append(
            {
                "match_number": match_number,
                "stage": row.get("stage", ""),
                "home_team_old": old.get("home_team", ""),
                "away_team_old": old.get("away_team", ""),
                "score_old": old.get("score", ""),
                "predicted_winner_old": old.get("predicted_winner", ""),
                "home_team_new": row.get("home_team", ""),
                "away_team_new": row.get("away_team", ""),
                "score_new": row.get("score", ""),
                "predicted_winner_new": row.get("predicted_winner", ""),
            }
        )
    return changes


def same_fixture_score_changes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only score changes for the exact same displayed fixture.

    Route or schedule mapping changes can be useful for diagnostics, but they are
    noisy in the phone dashboard. For filling scores, the only useful change is:
    same match number, same teams, different score.
    """
    filtered: list[dict[str, Any]] = []
    for row in rows:
        same_home = str(row.get("home_team_old", "")) == str(row.get("home_team_new", ""))
        same_away = str(row.get("away_team_old", "")) == str(row.get("away_team_new", ""))
        score_changed = str(row.get("score_old", "")) != str(row.get("score_new", ""))
        if same_home and same_away and score_changed:
            filtered.append(row)
    return filtered


def main() -> None:
    args = parse_args()
    model_root = args.model_root.resolve()
    run_dir = (args.run_dir or newest_run(model_root)).resolve()
    public_data = APP_ROOT / "public" / "data"
    public_files = APP_ROOT / "public" / "files"
    public_data.mkdir(parents=True, exist_ok=True)
    public_files.mkdir(parents=True, exist_ok=True)

    match = re.search(r"(\d{8})$", run_dir.name)
    label = match.group(1) if match else datetime.now().strftime("%Y%m%d")

    previous_dashboard = load_json(public_data / "dashboard.json")
    predictions = compact_predictions(read_csv(run_dir / "scorito_invuladvies.csv"))
    changes = read_csv(run_dir / f"score_changes_vs_{args.previous_label}.csv")
    if not changes:
        changes = read_csv(run_dir / "score_changes_vs_20260609.csv")
    if not changes:
        changes = build_changes_from_previous(
            previous_dashboard.get("predictions", []),
            predictions,
        )
    changes = same_fixture_score_changes(changes)

    model_output_name = run_dir.name.replace(
        "outputs_worldcup2026_cards_draw05_",
        "outputs_worldcup2026_cards_",
        1,
    )
    metrics_path = model_root / model_output_name / "model_metrics.json"
    metrics = load_json(metrics_path)

    results_path = model_root / "data" / "results.csv"
    latest_match_date = ""
    row_count = None
    if results_path.exists():
        rows = read_csv(results_path)
        row_count = len(rows)
        dates = [
            str(row.get("date", ""))
            for row in rows
            if has_score_value(row.get("home_score")) and has_score_value(row.get("away_score"))
        ]
        latest_match_date = max(dates) if dates else ""

    compact_csv_source = run_dir / "scorito_scores_invullen_compact.csv"

    downloads = {
        "compact_excel": copy_first_existing(
            [
                run_dir / f"Scorito_scores_puur_{label}.xlsx",
                run_dir / "Scorito_scores_puur_latest.xlsx",
                run_dir / "WK2026_Voorspellingen.xlsx",
            ],
            public_files / "Scorito_scores_puur_latest.xlsx",
        ),
        "probabilities_excel": copy_first_existing(
            [
                run_dir / f"Scorito_scores_met_kansen_{label}.xlsx",
                run_dir / "Scorito_scores_met_kansen_latest.xlsx",
                run_dir / "WK2026_Voorspellingen.xlsx",
            ],
            public_files / "Scorito_scores_met_kansen_latest.xlsx",
        ),
        "full_excel": copy_if_exists(
            run_dir / "WK2026_Voorspellingen.xlsx",
            public_files / "WK2026_Voorspellingen_latest.xlsx",
        ),
        "compact_csv": copy_if_exists(
            compact_csv_source,
            public_files / "scorito_scores_invullen_compact_latest.csv",
        )
        if compact_csv_source.exists()
        else write_csv(
            predictions,
            public_files / "scorito_scores_invullen_compact_latest.csv",
        ),
    }

    sources = [
        file_status("International results", results_path, "martj42 international_results"),
        file_status(
            "OddsPortal WK odds",
            model_root / "data" / "extracted" / "oddsportal_worldcup2026_fixture_odds_schedule.csv",
            "group-stage 1X2 odds and fixture mapping",
        ),
        file_status(
            "Soccerbase lineups",
            model_root / "data" / "extracted" / "soccerbase_lineups_used.csv",
            "available as source; current default model does not enable lineup feature flag",
        ),
        file_status(
            "Soccerbase stats",
            model_root / "data" / "extracted" / "soccerbase_match_stats.csv",
            "rolling shots/corners/possession/fouls source",
        ),
        file_status(
            "Soccerbase cards",
            model_root / "data" / "extracted" / "soccerbase_cards_events.csv",
            "rolling cards/referee source",
        ),
    ]

    dashboard = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "source_run": run_dir.name,
            "model_accuracy": metrics.get("accuracy"),
            "exact_score_accuracy": metrics.get("score_exact_accuracy"),
            "row_count": row_count,
            "latest_match_date": latest_match_date,
            "features": metrics.get("features"),
            "lineup_features_enabled": metrics.get("soccerbase_lineup_features_enabled"),
            "stat_features_enabled": metrics.get("soccerbase_stat_features_enabled"),
            "card_features_enabled": metrics.get("soccerbase_card_features_enabled"),
        },
        "downloads": downloads,
        "predictions": predictions,
        "changes": changes,
        "champions": read_csv(run_dir / "scorito_champion_picks.csv"),
        "group_standings": read_csv(run_dir / "scorito_group_standings.csv"),
        "top_scorers": read_csv(run_dir / "scorito_topscorer_picks.csv"),
        "group_top_scorers": read_csv(run_dir / "scorito_groupstage_topscorer_picks.csv"),
        "sources": sources,
    }

    with (public_data / "dashboard.json").open("w", encoding="utf-8") as handle:
        json.dump(dashboard, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Published dashboard from {run_dir}")
    print(f"Wrote {public_data / 'dashboard.json'}")


if __name__ == "__main__":
    main()
