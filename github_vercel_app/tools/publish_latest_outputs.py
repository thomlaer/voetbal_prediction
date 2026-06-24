#!/usr/bin/env python
"""Publish the newest prediction outputs as compact files for the Vercel app."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = APP_ROOT.parent
DEFAULT_CARDS_PATH = DEFAULT_MODEL_ROOT / "data" / "extracted" / "soccerbase_cards_events.csv"
DEFAULT_ESPN_RESULTS_PATH = DEFAULT_MODEL_ROOT / "data" / "extracted" / "espn_worldcup2026_results.csv"
STAGE_ORDER = [
    "Group Stage",
    "Round of 32",
    "Round of 16",
    "Quarterfinals",
    "Semifinals",
    "Third Place Playoff",
    "Final",
]


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
        if p.is_dir()
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


def normalize_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    aliases = {
        "usa": "united states",
        "us": "united states",
        "bosnia herzegovina": "bosnia and herzegovina",
        "bosnia hz": "bosnia and herzegovina",
        "czech rep": "czech republic",
        "czechia": "czech republic",
        "cote d ivoire": "ivory coast",
        "cote divoire": "ivory coast",
        "dr congo": "congo dr",
        "korea republic": "south korea",
        "korea rep": "south korea",
        "turkiye": "turkey",
    }
    return aliases.get(text, text)


CANONICAL_TEAM_NAMES = {
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "congo dr": "DR Congo",
    "curacao": "Curaçao",
    "czech republic": "Czech Republic",
    "ivory coast": "Ivory Coast",
    "south korea": "South Korea",
    "turkey": "Turkey",
    "united states": "United States",
}


def canonical_team_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return CANONICAL_TEAM_NAMES.get(normalize_key(text), text)


def canonical_winner(value: Any, home_team: Any, away_team: Any) -> str:
    text = str(value or "").strip()
    if not text or text == "Draw":
        return text
    key = normalize_key(text)
    home = canonical_team_name(home_team)
    away = canonical_team_name(away_team)
    if key == normalize_key(home):
        return home
    if key == normalize_key(away):
        return away
    return canonical_team_name(text)


def canonicalize_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output["home_team"] = canonical_team_name(output.get("home_team", ""))
    output["away_team"] = canonical_team_name(output.get("away_team", ""))
    for field in (
        "predicted_winner",
        "model_predicted_winner",
        "filled_predicted_winner",
        "new_model_predicted_winner",
        "actual_winner",
        "pre_match_predicted_winner",
    ):
        if field in output:
            output[field] = canonical_winner(output.get(field, ""), output["home_team"], output["away_team"])
    return output


def date_key(value: Any) -> int | None:
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", str(value or ""))
    if not match:
        return None
    return int(f"{match.group(1)}{match.group(2)}{match.group(3)}")


def stage_start_keys(rows: list[dict[str, Any]]) -> dict[str, int]:
    starts: dict[str, int] = {}
    for row in rows:
        stage = str(row.get("stage", ""))
        key = date_key(row.get("date"))
        if not stage or key is None:
            continue
        if stage not in starts or key < starts[stage]:
            starts[stage] = key
    return starts


def is_stage_locked(stage: Any, starts: dict[str, int], snapshot_key: int | None) -> bool:
    stage_key = str(stage or "")
    return snapshot_key is not None and stage_key in starts and snapshot_key >= starts[stage_key]


def outcome_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def score_outcome(score: Any) -> str:
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(score or ""))
    if not match:
        return ""
    return outcome_label(int(match.group(1)), int(match.group(2)))


def score_winner(score: Any, home_team: Any, away_team: Any) -> str:
    parsed = parse_score(score)
    if parsed is None:
        return ""
    home_goals, away_goals = parsed
    outcome = outcome_label(home_goals, away_goals)
    if outcome == "home_win":
        return str(home_team or "")
    if outcome == "away_win":
        return str(away_team or "")
    return "Draw"


def parse_score(score: Any) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(score or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def is_placeholder_team(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return not text or text == "nan" or text.startswith("winner ") or text.startswith("group ")


KNOCKOUT_WINNER_LINKS = {
    89: (73, 75),
    90: (74, 77),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
    101: (97, 98),
    102: (99, 100),
    104: (101, 102),
}

KNOCKOUT_LOSER_LINKS = {
    103: (101, 102),
}


def participant_winner(row: dict[str, Any]) -> str:
    home = canonical_team_name(row.get("home_team", ""))
    away = canonical_team_name(row.get("away_team", ""))
    if row.get("actual_available") and row.get("actual_score"):
        winner = score_winner(row.get("actual_score"), home, away)
        if winner:
            return winner

    for score_field in ("filled_score", "pre_match_score", "score", "model_score"):
        winner = score_winner(row.get(score_field), home, away)
        if winner:
            if winner == "Draw":
                break
            return winner

    for winner_field in (
        "actual_winner",
        "filled_predicted_winner",
        "pre_match_predicted_winner",
        "predicted_winner",
        "model_predicted_winner",
    ):
        winner = canonical_winner(row.get(winner_field, ""), home, away)
        if normalize_key(winner) in {normalize_key(home), normalize_key(away)}:
            return winner
    return home


def participant_loser(row: dict[str, Any]) -> str:
    home = canonical_team_name(row.get("home_team", ""))
    away = canonical_team_name(row.get("away_team", ""))
    winner = participant_winner(row)
    if normalize_key(winner) == normalize_key(home):
        return away
    if normalize_key(winner) == normalize_key(away):
        return home
    return away


def normalize_row_winners(row: dict[str, Any]) -> None:
    for score_field, winner_field in (
        ("score", "predicted_winner"),
        ("model_score", "model_predicted_winner"),
        ("filled_score", "filled_predicted_winner"),
        ("pre_match_score", "pre_match_predicted_winner"),
        ("new_model_score", "new_model_predicted_winner"),
    ):
        if score_field not in row:
            continue
        winner = score_winner(row.get(score_field), row.get("home_team", ""), row.get("away_team", ""))
        if winner:
            row[winner_field] = winner


def repair_knockout_bracket(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Propagate displayed knockout winners through later displayed rounds.

    The scorer/model outputs are generated in multiple steps. If a later score
    layer changes a knockout winner, dependent rows must be refreshed too;
    otherwise a team can lose one row and still appear in the next round.
    """
    rows_by_match = {
        int(row.get("match_number")): row
        for row in predictions
        if str(row.get("match_number", "")).isdigit()
    }

    for match_number, (home_source, away_source) in KNOCKOUT_WINNER_LINKS.items():
        row = rows_by_match.get(match_number)
        home_row = rows_by_match.get(home_source)
        away_row = rows_by_match.get(away_source)
        if not row or not home_row or not away_row:
            continue

        home_team = participant_winner(home_row)
        away_team = participant_winner(away_row)
        if row.get("home_team") != home_team or row.get("away_team") != away_team:
            row["home_team"] = home_team
            row["away_team"] = away_team
            row["bracket_repaired"] = True
        normalize_row_winners(row)

    for match_number, (home_source, away_source) in KNOCKOUT_LOSER_LINKS.items():
        row = rows_by_match.get(match_number)
        home_row = rows_by_match.get(home_source)
        away_row = rows_by_match.get(away_source)
        if not row or not home_row or not away_row:
            continue

        home_team = participant_loser(home_row)
        away_team = participant_loser(away_row)
        if row.get("home_team") != home_team or row.get("away_team") != away_team:
            row["home_team"] = home_team
            row["away_team"] = away_team
            row["bracket_repaired"] = True
        normalize_row_winners(row)

    return [canonicalize_prediction_row(row) for row in predictions]


def head_to_head_stats(
    teams: set[str],
    matches: list[tuple[str, str, int, int]],
) -> dict[str, dict[str, int]]:
    stats = {
        team: {"h2h_points": 0, "h2h_gd": 0, "h2h_gf": 0}
        for team in teams
    }
    for home, away, home_goals, away_goals in matches:
        if home not in teams or away not in teams:
            continue
        stats[home]["h2h_gf"] += home_goals
        stats[away]["h2h_gf"] += away_goals
        stats[home]["h2h_gd"] += home_goals - away_goals
        stats[away]["h2h_gd"] += away_goals - home_goals
        if home_goals > away_goals:
            stats[home]["h2h_points"] += 3
        elif away_goals > home_goals:
            stats[away]["h2h_points"] += 3
        else:
            stats[home]["h2h_points"] += 1
            stats[away]["h2h_points"] += 1
    return stats


def card_conduct_points(card_type: Any) -> int:
    text = str(card_type or "").strip().lower()
    if "red" in text:
        return -4
    if "yellow" in text:
        return -1
    return 0


def load_fair_play_points(path: Path = DEFAULT_CARDS_PATH) -> dict[str, int]:
    if not path.exists():
        return {}
    points: dict[str, int] = {}
    for row in read_csv(path):
        if not str(row.get("date", "")).startswith("2026-"):
            continue
        if "world cup" not in str(row.get("competition", "")).lower():
            continue
        if "group" not in str(row.get("stage", "")).lower():
            continue
        team_key = normalize_key(row.get("team", ""))
        if not team_key:
            continue
        points[team_key] = points.get(team_key, 0) + card_conduct_points(row.get("card_type"))
    return points


def rank_group_rows(
    rows: list[dict[str, Any]],
    matches: list[tuple[str, str, int, int]],
    fair_play_points: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    fair_play_points = fair_play_points or {}
    point_buckets: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        point_buckets.setdefault(int(row["points"]), []).append(row)

    for points in sorted(point_buckets, reverse=True):
        tied = point_buckets[points]
        if len(tied) == 1:
            ranked.extend(tied)
            continue

        tied_teams = {str(row["team"]) for row in tied}
        h2h = head_to_head_stats(tied_teams, matches)
        tied.sort(
            key=lambda item: (
                -int(h2h[str(item["team"])]["h2h_points"]),
                -int(h2h[str(item["team"])]["h2h_gd"]),
                -int(h2h[str(item["team"])]["h2h_gf"]),
                -int(item["gd"]),
                -int(item["gf"]),
                -int(fair_play_points.get(normalize_key(item["team"]), 0)),
                str(item["team"]),
            )
        )
        ranked.extend(tied)
    return ranked


def load_worldcup_actuals(results_path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    actuals: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not results_path.exists():
        return actuals
    for row in read_csv(results_path):
        if str(row.get("tournament", "")) != "FIFA World Cup":
            continue
        if not has_score_value(row.get("home_score")) or not has_score_value(row.get("away_score")):
            continue
        date = str(row.get("date", ""))[:10]
        if not date:
            continue
        home_score = int(float(row.get("home_score", 0)))
        away_score = int(float(row.get("away_score", 0)))
        home_team = str(row.get("home_team", ""))
        away_team = str(row.get("away_team", ""))
        outcome = outcome_label(home_score, away_score)
        reverse_outcome = outcome_label(away_score, home_score)
        actuals[(date, normalize_key(home_team), normalize_key(away_team))] = {
            "actual_available": True,
            "actual_home_score": home_score,
            "actual_away_score": away_score,
            "actual_score": f"{home_score}-{away_score}",
            "actual_outcome": outcome,
            "actual_winner": home_team if outcome == "home_win" else away_team if outcome == "away_win" else "Draw",
            "actual_source": "results_csv",
        }
        actuals[(date, normalize_key(away_team), normalize_key(home_team))] = {
            "actual_available": True,
            "actual_home_score": away_score,
            "actual_away_score": home_score,
            "actual_score": f"{away_score}-{home_score}",
            "actual_outcome": reverse_outcome,
            "actual_winner": away_team if reverse_outcome == "home_win" else home_team if reverse_outcome == "away_win" else "Draw",
            "actual_source": "results_csv",
        }
    return actuals


def load_soccerbase_actuals(stats_path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    actuals: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not stats_path.exists():
        return actuals
    for row in read_csv(stats_path):
        if "world cup" not in str(row.get("competition", "")).lower():
            continue
        date = str(row.get("date", ""))[:10]
        if not date.startswith("2026-"):
            continue
        if not has_score_value(row.get("home_score")) or not has_score_value(row.get("away_score")):
            continue
        home_team = str(row.get("home_team", ""))
        away_team = str(row.get("away_team", ""))
        try:
            home_score = int(float(row.get("home_score", 0)))
            away_score = int(float(row.get("away_score", 0)))
        except (TypeError, ValueError):
            continue
        outcome = outcome_label(home_score, away_score)
        reverse_outcome = outcome_label(away_score, home_score)
        actuals[(date, normalize_key(home_team), normalize_key(away_team))] = {
            "actual_available": True,
            "actual_home_score": home_score,
            "actual_away_score": away_score,
            "actual_score": f"{home_score}-{away_score}",
            "actual_outcome": outcome,
            "actual_winner": home_team if outcome == "home_win" else away_team if outcome == "away_win" else "Draw",
            "actual_source": "soccerbase",
        }
        actuals[(date, normalize_key(away_team), normalize_key(home_team))] = {
            "actual_available": True,
            "actual_home_score": away_score,
            "actual_away_score": home_score,
            "actual_score": f"{away_score}-{home_score}",
            "actual_outcome": reverse_outcome,
            "actual_winner": away_team if reverse_outcome == "home_win" else home_team if reverse_outcome == "away_win" else "Draw",
            "actual_source": "soccerbase",
        }
    return actuals


def load_espn_actuals(results_path: Path = DEFAULT_ESPN_RESULTS_PATH) -> dict[tuple[str, str, str], dict[str, Any]]:
    actuals: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not results_path.exists():
        return actuals
    for row in read_csv(results_path):
        if str(row.get("completed", "")).lower() not in {"true", "1"}:
            continue
        date = str(row.get("date", ""))[:10]
        if not date.startswith("2026-"):
            continue
        if not has_score_value(row.get("home_score")) or not has_score_value(row.get("away_score")):
            continue
        home_team = str(row.get("home_team", ""))
        away_team = str(row.get("away_team", ""))
        try:
            home_score = int(float(row.get("home_score", 0)))
            away_score = int(float(row.get("away_score", 0)))
        except (TypeError, ValueError):
            continue
        outcome = outcome_label(home_score, away_score)
        reverse_outcome = outcome_label(away_score, home_score)
        direct = {
            "actual_available": True,
            "actual_home_score": home_score,
            "actual_away_score": away_score,
            "actual_score": f"{home_score}-{away_score}",
            "actual_outcome": outcome,
            "actual_winner": home_team if outcome == "home_win" else away_team if outcome == "away_win" else "Draw",
            "actual_source": "espn",
            "actual_online_verified": True,
        }
        reverse = {
            "actual_available": True,
            "actual_home_score": away_score,
            "actual_away_score": home_score,
            "actual_score": f"{away_score}-{home_score}",
            "actual_outcome": reverse_outcome,
            "actual_winner": away_team if reverse_outcome == "home_win" else home_team if reverse_outcome == "away_win" else "Draw",
            "actual_source": "espn",
            "actual_online_verified": True,
        }
        date_keys = [date]
        try:
            date_keys.append((datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d"))
        except ValueError:
            pass
        for date_key in date_keys:
            actuals[(date_key, normalize_key(home_team), normalize_key(away_team))] = direct
            actuals[(date_key, normalize_key(away_team), normalize_key(home_team))] = reverse
    return actuals


def prediction_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("match_number", "")),
        normalize_key(row.get("home_team", "")),
        normalize_key(row.get("away_team", "")),
    )


def load_locked_scores(path: Path) -> dict[str, dict[str, Any]]:
    locked: dict[str, dict[str, Any]] = {}
    for row in read_csv(path):
        match_number = str(row.get("match_number", "")).strip()
        score = str(row.get("score", "")).strip()
        if not match_number or parse_score(score) is None:
            continue
        locked[match_number] = {
            "score": score,
            "predicted_winner": str(row.get("predicted_winner", "")).strip(),
            "note": row.get("note", ""),
        }
    return locked


def attach_actual_results(
    predictions: list[dict[str, Any]],
    results_path: Path,
    soccerbase_stats_path: Path,
    previous_dashboard: dict[str, Any],
    snapshot_key: int | None,
    locked_scores: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    actuals = load_soccerbase_actuals(soccerbase_stats_path)
    actuals.update(load_worldcup_actuals(results_path))
    actuals.update(load_espn_actuals())
    previous_rows = previous_dashboard.get("predictions", [])
    previous_by_key = {
        prediction_key(row): row
        for row in previous_rows
        if isinstance(row, dict) and str(row.get("match_number", ""))
    }
    previous_by_match = {
        str(row.get("match_number", "")).strip(): row
        for row in previous_rows
        if isinstance(row, dict) and str(row.get("match_number", "")).strip()
    }
    starts = stage_start_keys(predictions)

    enriched: list[dict[str, Any]] = []
    for row in predictions:
        output = dict(row)
        previous_same_slot = previous_by_match.get(str(output.get("match_number", "")).strip(), {})
        if previous_same_slot:
            if is_placeholder_team(output.get("home_team")) and not is_placeholder_team(previous_same_slot.get("home_team")):
                output["home_team"] = previous_same_slot.get("home_team", output.get("home_team"))
            if is_placeholder_team(output.get("away_team")) and not is_placeholder_team(previous_same_slot.get("away_team")):
                output["away_team"] = previous_same_slot.get("away_team", output.get("away_team"))
        output["home_team"] = canonical_team_name(output.get("home_team", ""))
        output["away_team"] = canonical_team_name(output.get("away_team", ""))
        previous = previous_by_key.get(prediction_key(output), {})
        model_score = output.get("score", "")
        model_winner = canonical_winner(output.get("predicted_winner", ""), output["home_team"], output["away_team"])
        output["predicted_winner"] = model_winner
        model_confidence = output.get("confidence", "")
        model_favourite_prob = output.get("model_favourite_prob", "")
        locked = is_stage_locked(output.get("stage", ""), starts, snapshot_key)

        output["model_score"] = model_score
        output["model_predicted_winner"] = model_winner
        output["filled_score"] = model_score
        output["filled_predicted_winner"] = model_winner
        output["new_model_score"] = ""
        output["new_model_predicted_winner"] = ""
        output["round_locked"] = locked
        output["round_status"] = "locked" if locked else "open"
        output["score_source"] = "current_run"

        if locked and previous:
            filled_score = (
                previous.get("filled_score")
                or previous.get("pre_match_score")
                or previous.get("score")
                or model_score
            )
            filled_winner = (
                previous.get("filled_predicted_winner")
                or previous.get("pre_match_predicted_winner")
                or previous.get("predicted_winner")
                or model_winner
            )
            output["filled_score"] = filled_score
            output["filled_predicted_winner"] = filled_winner
            output["score"] = filled_score
            output["predicted_winner"] = filled_winner
            output["score_source"] = previous.get("score_source") or "previous_dashboard"
            if str(model_score) != str(filled_score):
                output["new_model_score"] = model_score
            if str(model_winner) != str(filled_winner):
                output["new_model_predicted_winner"] = model_winner

        date = str(output.get("date", ""))[:10]
        actual = actuals.get(
            (date, normalize_key(output.get("home_team", "")), normalize_key(output.get("away_team", "")))
        )
        if actual:
            output.update(actual)
            output["round_status"] = "played"
            output["new_model_score"] = ""
            output["new_model_predicted_winner"] = ""
        else:
            output.update(
                {
                    "actual_available": False,
                    "actual_home_score": "",
                    "actual_away_score": "",
                    "actual_score": "",
                    "actual_outcome": "",
                    "actual_winner": "",
                    "actual_source": "",
                    "actual_online_verified": False,
                }
            )

        output["pre_match_score"] = output.get("filled_score", output.get("score", ""))
        output["pre_match_predicted_winner"] = output.get(
            "filled_predicted_winner",
            output.get("predicted_winner", ""),
        )
        output["pre_match_confidence"] = model_confidence
        output["pre_match_model_favourite_prob"] = model_favourite_prob
        output["pre_match_source"] = output.get("score_source", "current_run")

        if output["actual_available"] and previous:
            output["pre_match_score"] = previous.get("pre_match_score") or previous.get("score", output["pre_match_score"])
            output["pre_match_predicted_winner"] = (
                previous.get("pre_match_predicted_winner")
                or previous.get("predicted_winner", output["pre_match_predicted_winner"])
            )
            output["pre_match_confidence"] = previous.get("pre_match_confidence") or previous.get(
                "confidence", output["pre_match_confidence"]
            )
            output["pre_match_model_favourite_prob"] = previous.get("pre_match_model_favourite_prob") or previous.get(
                "model_favourite_prob", output["pre_match_model_favourite_prob"]
            )
            output["pre_match_source"] = previous.get("pre_match_source") or "previous_dashboard"

        locked_score = locked_scores.get(str(output.get("match_number", "")).strip())
        if locked_score:
            manual_score = locked_score["score"]
            manual_winner = locked_score.get("predicted_winner") or score_winner(
                manual_score,
                output.get("home_team", ""),
                output.get("away_team", ""),
            )
            output["new_model_score"] = model_score if str(model_score) != str(manual_score) else ""
            output["new_model_predicted_winner"] = model_winner if str(model_winner) != str(manual_winner) else ""
            output["filled_score"] = manual_score
            output["filled_predicted_winner"] = manual_winner
            output["score"] = manual_score
            output["predicted_winner"] = manual_winner
            output["round_locked"] = True
            output["round_status"] = "locked"
            output["pre_match_score"] = manual_score
            output["pre_match_predicted_winner"] = manual_winner
            output["score_source"] = "manual_locked_score"
            output["pre_match_source"] = "manual_locked_score"

        if output.get("actual_available") or output.get("actual_score"):
            output["new_model_score"] = ""
            output["new_model_predicted_winner"] = ""

        output = canonicalize_prediction_row(output)

        if output["actual_available"]:
            predicted_score = output.get("pre_match_score", "")
            predicted_outcome = score_outcome(predicted_score)
            actual_outcome = str(output.get("actual_outcome", ""))
            output["prediction_exact"] = str(predicted_score) == str(output.get("actual_score", ""))
            output["prediction_outcome_correct"] = bool(predicted_outcome) and predicted_outcome == actual_outcome
        else:
            output["prediction_exact"] = ""
            output["prediction_outcome_correct"] = ""
        enriched.append(output)
    return enriched


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


def build_group_standings(predictions: list[dict[str, Any]], champions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    champion_by_team = {normalize_key(row.get("team")): row for row in champions}
    fair_play_points = load_fair_play_points()
    standings: dict[str, dict[str, dict[str, Any]]] = {}
    matches_by_group: dict[str, list[tuple[str, str, int, int]]] = {}
    status_by_group: dict[str, dict[str, int]] = {}
    for row in predictions:
        if str(row.get("stage", "")) != "Group Stage":
            continue
        group = str(row.get("group", "") or "-")
        home = canonical_team_name(row.get("home_team", ""))
        away = canonical_team_name(row.get("away_team", ""))
        if is_placeholder_team(home) or is_placeholder_team(away):
            continue
        is_actual = bool(row.get("actual_available") or row.get("actual_score"))
        group_status = status_by_group.setdefault(group, {"total": 0, "actual": 0, "online": 0})
        group_status["total"] += 1
        if is_actual:
            group_status["actual"] += 1
            if row.get("actual_online_verified") or "espn" in str(row.get("actual_source", "")).lower():
                group_status["online"] += 1
        score = row.get("actual_score") if is_actual else row.get("filled_score") or row.get("score")
        parsed = parse_score(score)
        if parsed is None:
            continue
        home_goals, away_goals = parsed
        matches_by_group.setdefault(group, []).append((home, away, home_goals, away_goals))
        group_rows = standings.setdefault(group, {})
        for team in (home, away):
            group_rows.setdefault(
                team,
                {
                    "group": group,
                    "team": team,
                    "played": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "gf": 0,
                    "ga": 0,
                    "gd": 0,
                    "points": 0,
                },
            )
        home_row = group_rows[home]
        away_row = group_rows[away]
        home_row["played"] += 1
        away_row["played"] += 1
        home_row["gf"] += home_goals
        home_row["ga"] += away_goals
        away_row["gf"] += away_goals
        away_row["ga"] += home_goals
        if home_goals > away_goals:
            home_row["wins"] += 1
            away_row["losses"] += 1
            home_row["points"] += 3
        elif home_goals < away_goals:
            away_row["wins"] += 1
            home_row["losses"] += 1
            away_row["points"] += 3
        else:
            home_row["draws"] += 1
            away_row["draws"] += 1
            home_row["points"] += 1
            away_row["points"] += 1

    output: list[dict[str, Any]] = []
    for group in sorted(standings):
        rows = list(standings[group].values())
        for row in rows:
            row["gd"] = int(row["gf"]) - int(row["ga"])
        rows = rank_group_rows(rows, matches_by_group.get(group, []), fair_play_points)
        group_status = status_by_group.get(group, {"total": 0, "actual": 0, "online": 0})
        group_total = int(group_status.get("total", 0))
        group_actual = int(group_status.get("actual", 0))
        group_online = int(group_status.get("online", 0))
        group_complete = group_total > 0 and group_actual == group_total
        group_online_verified = group_total > 0 and group_online == group_total
        standing_source = "actual_results" if group_complete else "mixed_actual_projection" if group_actual else "projection"
        for rank, row in enumerate(rows, 1):
            team_probs = champion_by_team.get(normalize_key(row["team"]), {})
            output.append(
                {
                    **row,
                    "rank": rank,
                    "qualified_by_pick": rank <= 2,
                    "group_matches_total": group_total,
                    "group_matches_actual": group_actual,
                    "group_matches_online_verified": group_online,
                    "group_complete": group_complete,
                    "group_online_verified": group_online_verified,
                    "standing_source": standing_source,
                    "rank_confirmed": group_complete,
                    "qualified_confirmed": group_complete and rank <= 2,
                    "advance_r16_prob": team_probs.get("advance_r16_prob", ""),
                    "advance_qf_prob": team_probs.get("advance_qf_prob", ""),
                    "champion_prob": team_probs.get("champion_prob", ""),
                }
            )
    return output


def build_round_top_scorers(
    top_scorers: list[dict[str, Any]],
    group_top_scorers: list[dict[str, Any]],
    champions: list[dict[str, Any]],
    limit_per_stage: int = 10,
) -> list[dict[str, Any]]:
    champion_by_team = {normalize_key(row.get("team")): row for row in champions}
    stages = [
        ("Group Stage", "Groepsfase", "expected_group_stage_goals", "expected_group_stage_scorito_points"),
        ("Round of 32", "1/16 finale", "r32_goals", "r32_scorito_points"),
        ("Round of 16", "Achtste finale", "r16_goals", "r16_scorito_points"),
        ("Quarterfinals", "Kwartfinale", "qf_goals", "qf_scorito_points"),
        ("Semifinals", "Halve finale", "sf_goals", "sf_scorito_points"),
        ("Final/Third", "Finale/troost", "final_goals", "final_scorito_points"),
        ("Total", "Totaal", "expected_goals", "expected_scorito_points"),
    ]
    source_rows = top_scorers or group_top_scorers
    output: list[dict[str, Any]] = []

    for row in source_rows:
        team = canonical_team_name(row.get("team", ""))
        team_probs = champion_by_team.get(normalize_key(team), {})
        points_per_goal = float(row.get("scorito_points_per_goal") or 0.0)
        if points_per_goal <= 0:
            points_per_goal = 8.0 if str(row.get("position", "")).upper().startswith("FW") else 16.0

        group_goals = float(row.get("expected_group_stage_goals") or 0.0)
        group_rate = group_goals / 3.0 if group_goals > 0 else float(row.get("expected_goals") or 0.0) / 5.0
        knockout_rate = 0.82 * max(group_rate, 0.0)
        expected_matches = float(row.get("team_expected_matches") or 3.0)
        advance_r16 = float(team_probs.get("advance_r16_prob") or 0.0)
        advance_qf = float(team_probs.get("advance_qf_prob") or 0.0)
        advance_sf = float(team_probs.get("advance_sf_prob") or 0.0)
        advance_final = float(team_probs.get("advance_final_prob") or 0.0)
        advance_r32 = max(0.0, min(1.0, expected_matches - 3.0 - advance_r16 - advance_qf - 2.0 * advance_sf))

        stage_values = {
            "expected_group_stage_goals": group_goals,
            "expected_group_stage_scorito_points": float(row.get("expected_group_stage_scorito_points") or 0.0),
            "r32_goals": knockout_rate * advance_r32,
            "r16_goals": knockout_rate * advance_r16,
            "qf_goals": knockout_rate * advance_qf,
            "sf_goals": knockout_rate * advance_sf,
            "final_goals": knockout_rate * advance_final,
            "expected_goals": float(row.get("expected_goals") or 0.0),
            "expected_scorito_points": float(row.get("expected_scorito_points") or 0.0),
        }
        for goals_key in ("r32_goals", "r16_goals", "qf_goals", "sf_goals", "final_goals"):
            stage_values[goals_key.replace("_goals", "_scorito_points")] = stage_values[goals_key] * points_per_goal

        for stage, label, goals_key, points_key in stages:
            output.append(
                {
                    "stage": stage,
                    "stage_label": label,
                    "team": team,
                    "player": row.get("player", ""),
                    "position": row.get("position", ""),
                    "expected_goals": stage_values.get(goals_key, 0.0),
                    "expected_scorito_points": stage_values.get(points_key, 0.0),
                }
            )

    ranked: list[dict[str, Any]] = []
    for stage, label, _goals_key, _points_key in stages:
        rows = [row for row in output if row["stage"] == stage]
        rows.sort(key=lambda item: (float(item["expected_scorito_points"]), float(item["expected_goals"])), reverse=True)
        for rank, row in enumerate(rows[:limit_per_stage], 1):
            ranked.append({"round_rank": rank, **row})
    return ranked


ROUND_TOPSCORER_ROUTE_STAGES = {
    "Group Stage": {"Group Stage"},
    "Round of 32": {"Round of 32"},
    "Round of 16": {"Round of 16"},
    "Quarterfinals": {"Quarterfinals"},
    "Semifinals": {"Semifinals"},
    "Final/Third": {"Final", "Third Place Playoff"},
}


def route_team_keys_by_topscorer_stage(predictions: list[dict[str, Any]]) -> dict[str, set[str]]:
    teams_by_stage: dict[str, set[str]] = {}
    for topscorer_stage, prediction_stages in ROUND_TOPSCORER_ROUTE_STAGES.items():
        teams: set[str] = set()
        for row in predictions:
            if str(row.get("stage", "")) not in prediction_stages:
                continue
            for field in ("home_team", "away_team"):
                team = canonical_team_name(row.get(field, ""))
                if team and not is_placeholder_team(team):
                    teams.add(normalize_key(team))
        if teams:
            teams_by_stage[topscorer_stage] = teams
    return teams_by_stage


def normalize_stage_top_scorers(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]] | None = None,
    limit_per_stage: int = 10,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    stage_order = {stage: idx for idx, stage in enumerate(STAGE_ORDER + ["Final/Third"])}
    route_teams = route_team_keys_by_topscorer_stage(predictions or [])
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        stage = str(row.get("stage", ""))
        if not stage:
            continue
        grouped_rows.setdefault(stage, []).append(row)

    output: list[dict[str, Any]] = []
    for stage, stage_rows in grouped_rows.items():
        allowed_teams = route_teams.get(stage)
        if allowed_teams:
            route_filtered_rows = [
                row
                for row in stage_rows
                if normalize_key(canonical_team_name(row.get("team", ""))) in allowed_teams
            ]
            if route_filtered_rows:
                stage_rows = route_filtered_rows
        stage_rows.sort(
            key=lambda item: (
                float(item.get("recommended_stage_topscorer_score") or 0.0),
                float(item.get("expected_scorito_points") or 0.0),
                float(item.get("expected_goals") or 0.0),
            ),
            reverse=True,
        )
        for rank, row in enumerate(stage_rows[:limit_per_stage], 1):
            output.append(
                {
                    **row,
                    "round_rank": rank,
                    "stage_rank": rank,
                    "stage_label": row.get("stage_label") or stage,
                    "stage_order": row.get("stage_order", stage_order.get(stage, 999)),
                }
            )
    output.sort(key=lambda row: (int(row.get("stage_order") or 999), int(row.get("round_rank") or 999)))
    return output


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
        "model_score",
        "model_predicted_winner",
        "filled_score",
        "filled_predicted_winner",
        "new_model_score",
        "new_model_predicted_winner",
        "round_locked",
        "round_status",
        "score_source",
        "model_favourite_prob",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "actual_available",
        "actual_home_score",
        "actual_away_score",
        "actual_score",
        "actual_outcome",
        "actual_winner",
        "actual_source",
        "pre_match_score",
        "pre_match_predicted_winner",
        "pre_match_confidence",
        "pre_match_model_favourite_prob",
        "pre_match_source",
        "prediction_exact",
        "prediction_outcome_correct",
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
    generated_at = datetime.now()
    snapshot_key = date_key(generated_at.strftime("%Y-%m-%d"))

    match = re.search(r"(\d{8})$", run_dir.name)
    label = match.group(1) if match else generated_at.strftime("%Y%m%d")

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

    soccerbase_stats_path = model_root / "data" / "extracted" / "soccerbase_match_stats.csv"
    locked_scores = load_locked_scores(model_root / "data" / "extracted" / "scorito_locked_scores.csv")
    predictions = attach_actual_results(
        predictions,
        results_path,
        soccerbase_stats_path,
        previous_dashboard,
        snapshot_key,
        locked_scores,
    )
    played_match_numbers = {
        str(row.get("match_number", "")).strip()
        for row in predictions
        if row.get("actual_available") or row.get("actual_score")
    }
    changes = [
        row
        for row in changes
        if str(row.get("match_number", "")).strip() not in played_match_numbers
    ]

    predictions = repair_knockout_bracket(predictions)
    champions = read_csv(run_dir / "scorito_champion_picks.csv")
    top_scorers = read_csv(run_dir / "scorito_topscorer_picks.csv")
    group_top_scorers = read_csv(run_dir / "scorito_groupstage_topscorer_picks.csv")
    stage_top_scorers = read_csv(run_dir / "scorito_stage_topscorer_picks.csv")
    group_standings = build_group_standings(predictions, champions)
    round_top_scorers = (
        normalize_stage_top_scorers(stage_top_scorers, predictions)
        if stage_top_scorers
        else build_round_top_scorers(top_scorers, group_top_scorers, champions)
    )

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
        "compact_csv": write_csv(
            predictions,
            public_files / "scorito_scores_invullen_compact_latest.csv",
        ),
    }

    sources = [
        file_status("International results", results_path, "martj42 international_results"),
        file_status(
            "ESPN online results",
            DEFAULT_ESPN_RESULTS_PATH,
            "online completed-match check from ESPN scoreboard",
        ),
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
            soccerbase_stats_path,
            "rolling shots/corners/possession/fouls source",
        ),
        file_status(
            "Soccerbase cards",
            model_root / "data" / "extracted" / "soccerbase_cards_events.csv",
            "rolling cards/referee source",
        ),
        file_status(
            "Manual player adjustments",
            model_root / "data" / "extracted" / "manual_player_adjustments.csv",
            "optional injuries, suspensions, penalty takers and player notes",
        ),
    ]

    dashboard = {
        "metadata": {
            "generated_at": generated_at.strftime("%Y-%m-%d %H:%M"),
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
        "champions": champions,
        "group_standings": group_standings,
        "top_scorers": top_scorers,
        "group_top_scorers": group_top_scorers,
        "stage_top_scorers": stage_top_scorers,
        "round_top_scorers": round_top_scorers,
        "sources": sources,
    }

    with (public_data / "dashboard.json").open("w", encoding="utf-8") as handle:
        json.dump(dashboard, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Published dashboard from {run_dir}")
    print(f"Wrote {public_data / 'dashboard.json'}")


if __name__ == "__main__":
    main()
