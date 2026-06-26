#!/usr/bin/env python
"""Fetch ESPN World Cup 2026 match details.

This script enriches completed ESPN World Cup fixtures with lineups, formations,
team stats, player match stats and key events from ESPN's public summary
endpoint.

Important modelling note: these details are mostly available after kickoff or
after full-time. Do not use same-match goals/events/stats as pre-match features.
They are useful as post-match tournament state for later rounds, dashboard
checks, and player availability/topscorer context.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = APP_ROOT.parent
DEFAULT_RESULTS = DEFAULT_MODEL_ROOT / "data" / "extracted" / "espn_worldcup2026_results.csv"
DEFAULT_FIXTURES = DEFAULT_MODEL_ROOT / "data" / "extracted" / "espn_worldcup2026_fixtures.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_ROOT / "outputs_runtime"
DEFAULT_LINEUPS_OUTPUT = DEFAULT_OUTPUT_DIR / "espn_worldcup2026_lineups.csv"
DEFAULT_TEAM_STATS_OUTPUT = DEFAULT_OUTPUT_DIR / "espn_worldcup2026_team_stats.csv"
DEFAULT_PLAYER_STATS_OUTPUT = DEFAULT_OUTPUT_DIR / "espn_worldcup2026_player_match_stats.csv"
DEFAULT_EVENTS_OUTPUT = DEFAULT_OUTPUT_DIR / "espn_worldcup2026_match_events.csv"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary"
USER_AGENT = "voetbal-prediction/1.0"


LINEUP_FIELDS = [
    "source",
    "espn_event_id",
    "date",
    "stage",
    "home_team",
    "away_team",
    "team",
    "opponent",
    "team_id",
    "team_abbreviation",
    "home_away",
    "is_home",
    "formation",
    "player_id",
    "player_name",
    "jersey",
    "position_name",
    "position_abbreviation",
    "soccerbase_position",
    "formation_place",
    "is_starter",
    "is_sub_used",
    "subbed_in",
    "subbed_out",
    "subbed_in_for_id",
    "subbed_in_for_name",
    "subbed_out_for_id",
    "subbed_out_for_name",
    "active",
    "fetched_at",
    "source_url",
]

TEAM_STATS_FIELDS = [
    "source",
    "espn_event_id",
    "date",
    "stage",
    "home_team",
    "away_team",
    "team",
    "opponent",
    "team_id",
    "team_abbreviation",
    "home_away",
    "is_home",
    "stat_name",
    "stat_label",
    "stat_display_value",
    "stat_value",
    "fetched_at",
    "source_url",
]

PLAYER_STATS_FIELDS = [
    "source",
    "espn_event_id",
    "date",
    "stage",
    "home_team",
    "away_team",
    "team",
    "opponent",
    "team_id",
    "team_abbreviation",
    "home_away",
    "is_home",
    "formation",
    "player_id",
    "player_name",
    "position_name",
    "position_abbreviation",
    "is_starter",
    "is_sub_used",
    "stat_name",
    "stat_label",
    "stat_display_value",
    "stat_value",
    "fetched_at",
    "source_url",
]

EVENT_FIELDS = [
    "source",
    "espn_event_id",
    "date",
    "stage",
    "home_team",
    "away_team",
    "event_id",
    "event_type",
    "event_text",
    "event_short_text",
    "period",
    "clock_display",
    "wallclock",
    "scoring_play",
    "shootout",
    "score_value",
    "team",
    "team_id",
    "athlete_id",
    "athlete_name",
    "fetched_at",
    "source_url",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch ESPN World Cup match details.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--lineups-output", type=Path, default=DEFAULT_LINEUPS_OUTPUT)
    parser.add_argument("--team-stats-output", type=Path, default=DEFAULT_TEAM_STATS_OUTPUT)
    parser.add_argument("--player-stats-output", type=Path, default=DEFAULT_PLAYER_STATS_OUTPUT)
    parser.add_argument("--events-output", type=Path, default=DEFAULT_EVENTS_OUTPUT)
    parser.add_argument("--event-id", action="append", default=[], help="Fetch one or more explicit ESPN event ids.")
    parser.add_argument("--include-scheduled", action="store_true", help="Also fetch fixture ids not yet completed.")
    parser.add_argument("--max-events", type=int, default=0, help="Limit number of event ids for smoke tests.")
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "completed"}


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def parse_number(value: Any) -> float | str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return ""
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return ""
    try:
        return float(match.group(0))
    except ValueError:
        return ""


def first_item(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value:
        item = value[0]
        return item if isinstance(item, dict) else {}
    if isinstance(value, dict):
        return value
    return {}


def nested_name(value: Any) -> str:
    if isinstance(value, dict):
        return clean_text(
            value.get("displayName")
            or value.get("fullName")
            or value.get("shortName")
            or value.get("name")
            or value.get("description")
        )
    return clean_text(value)


def event_metadata(results_path: Path, fixtures_path: Path, include_scheduled: bool) -> dict[str, dict[str, str]]:
    rows_by_id: dict[str, dict[str, str]] = {}

    for row in read_csv_rows(fixtures_path):
        event_id = clean_text(row.get("espn_event_id"))
        if not event_id:
            continue
        rows_by_id[event_id] = {
            "espn_event_id": event_id,
            "date": clean_text(row.get("date") or row.get("kickoff_utc"))[:10],
            "stage": clean_text(row.get("stage") or "World Cup"),
            "home_team": clean_text(row.get("home_team") or row.get("espn_home_team")),
            "away_team": clean_text(row.get("away_team") or row.get("espn_away_team")),
            "completed": "False",
        }

    completed_ids: set[str] = set()
    for row in read_csv_rows(results_path):
        event_id = clean_text(row.get("espn_event_id"))
        if not event_id:
            continue
        completed_ids.add(event_id)
        base = rows_by_id.get(event_id, {"espn_event_id": event_id})
        base.update(
            {
                "date": clean_text(row.get("date") or base.get("date"))[:10],
                "stage": clean_text(base.get("stage") or "World Cup"),
                "home_team": clean_text(row.get("home_team") or base.get("home_team")),
                "away_team": clean_text(row.get("away_team") or base.get("away_team")),
                "completed": str(truthy(row.get("completed")) or bool(row.get("home_score"))),
            }
        )
        rows_by_id[event_id] = base

    if include_scheduled:
        return rows_by_id
    return {event_id: rows_by_id[event_id] for event_id in completed_ids if event_id in rows_by_id}


def fetch_summary(event_id: str) -> dict[str, Any]:
    response = requests.get(
        SUMMARY_URL,
        params={"event": event_id},
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def team_info(team: dict[str, Any]) -> tuple[str, str, str]:
    return (
        clean_text(team.get("id")),
        clean_text(team.get("displayName") or team.get("shortDisplayName") or team.get("name")),
        clean_text(team.get("abbreviation")),
    )


def opponent_for(team_name: str, meta: dict[str, str]) -> str:
    if team_name == meta.get("home_team"):
        return clean_text(meta.get("away_team"))
    if team_name == meta.get("away_team"):
        return clean_text(meta.get("home_team"))
    return ""


def event_source_url(event_id: str) -> str:
    return f"{SUMMARY_URL}?event={event_id}"


def parse_rosters(
    event_id: str,
    payload: dict[str, Any],
    meta: dict[str, str],
    fetched_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lineup_rows: list[dict[str, Any]] = []
    player_stat_rows: list[dict[str, Any]] = []
    source_url = event_source_url(event_id)

    for roster in payload.get("rosters") or []:
        if not isinstance(roster, dict):
            continue
        team = roster.get("team") or {}
        team_id, team_name, team_abbr = team_info(team)
        home_away = clean_text(roster.get("homeAway"))
        is_home = home_away == "home" or team_name == meta.get("home_team")
        opponent = opponent_for(team_name, meta)
        formation = clean_text(roster.get("formation"))

        for player in roster.get("roster") or []:
            if not isinstance(player, dict):
                continue
            athlete = player.get("athlete") or {}
            position = player.get("position") or {}
            subbed_in_for = player.get("subbedInFor") or {}
            subbed_out_for = player.get("subbedOutFor") or {}
            starter = bool(player.get("starter"))
            subbed_in = bool(player.get("subbedIn"))
            player_id = clean_text(athlete.get("id"))
            player_name = nested_name(athlete)
            position_name = clean_text(position.get("displayName") or position.get("name"))
            position_abbreviation = clean_text(position.get("abbreviation"))
            base = {
                "source": "espn",
                "espn_event_id": event_id,
                "date": meta.get("date", ""),
                "stage": meta.get("stage", ""),
                "home_team": meta.get("home_team", ""),
                "away_team": meta.get("away_team", ""),
                "team": team_name,
                "opponent": opponent,
                "team_id": team_id,
                "team_abbreviation": team_abbr,
                "home_away": home_away,
                "is_home": bool(is_home),
                "formation": formation,
                "player_id": player_id,
                "player_name": player_name,
                "position_name": position_name,
                "position_abbreviation": position_abbreviation,
                "is_starter": int(starter),
                "is_sub_used": int(subbed_in),
                "fetched_at": fetched_at,
                "source_url": source_url,
            }
            lineup_rows.append(
                {
                    **base,
                    "jersey": clean_text(player.get("jersey")),
                    "soccerbase_position": position_abbreviation or position_name,
                    "formation_place": clean_text(player.get("formationPlace")),
                    "subbed_in": int(subbed_in),
                    "subbed_out": int(bool(player.get("subbedOut"))),
                    "subbed_in_for_id": clean_text(subbed_in_for.get("id")),
                    "subbed_in_for_name": nested_name(subbed_in_for),
                    "subbed_out_for_id": clean_text(subbed_out_for.get("id")),
                    "subbed_out_for_name": nested_name(subbed_out_for),
                    "active": bool(player.get("active", "")),
                }
            )
            for stat in player.get("stats") or []:
                if not isinstance(stat, dict):
                    continue
                player_stat_rows.append(
                    {
                        **base,
                        "stat_name": clean_text(stat.get("name")),
                        "stat_label": clean_text(
                            stat.get("displayName") or stat.get("shortDisplayName") or stat.get("label")
                        ),
                        "stat_display_value": clean_text(stat.get("displayValue") or stat.get("value")),
                        "stat_value": parse_number(stat.get("displayValue") or stat.get("value")),
                    }
                )
    return lineup_rows, player_stat_rows


def parse_team_stats(
    event_id: str,
    payload: dict[str, Any],
    meta: dict[str, str],
    fetched_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_url = event_source_url(event_id)
    for box_team in ((payload.get("boxscore") or {}).get("teams") or []):
        if not isinstance(box_team, dict):
            continue
        team = box_team.get("team") or {}
        team_id, team_name, team_abbr = team_info(team)
        home_away = clean_text(box_team.get("homeAway"))
        is_home = home_away == "home" or team_name == meta.get("home_team")
        opponent = opponent_for(team_name, meta)
        for stat in box_team.get("statistics") or []:
            if not isinstance(stat, dict):
                continue
            display_value = stat.get("displayValue") or stat.get("value")
            rows.append(
                {
                    "source": "espn",
                    "espn_event_id": event_id,
                    "date": meta.get("date", ""),
                    "stage": meta.get("stage", ""),
                    "home_team": meta.get("home_team", ""),
                    "away_team": meta.get("away_team", ""),
                    "team": team_name,
                    "opponent": opponent,
                    "team_id": team_id,
                    "team_abbreviation": team_abbr,
                    "home_away": home_away,
                    "is_home": bool(is_home),
                    "stat_name": clean_text(stat.get("name")),
                    "stat_label": clean_text(stat.get("label") or stat.get("displayName")),
                    "stat_display_value": clean_text(display_value),
                    "stat_value": parse_number(display_value),
                    "fetched_at": fetched_at,
                    "source_url": source_url,
                }
            )
    return rows


def parse_key_events(
    event_id: str,
    payload: dict[str, Any],
    meta: dict[str, str],
    fetched_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_url = event_source_url(event_id)
    for event in payload.get("keyEvents") or []:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type") or {}
        clock = event.get("clock") or {}
        team = event.get("team") or {}
        participant = first_item(event.get("participants") or event.get("athletes"))
        athlete = participant.get("athlete") if isinstance(participant.get("athlete"), dict) else participant
        if not isinstance(athlete, dict):
            athlete = {}
        period = event.get("period") or {}
        rows.append(
            {
                "source": "espn",
                "espn_event_id": event_id,
                "date": meta.get("date", ""),
                "stage": meta.get("stage", ""),
                "home_team": meta.get("home_team", ""),
                "away_team": meta.get("away_team", ""),
                "event_id": clean_text(event.get("id")),
                "event_type": clean_text(event_type.get("type") or event_type.get("text") or event_type.get("id")),
                "event_text": clean_text(event.get("text")),
                "event_short_text": clean_text(event.get("shortText")),
                "period": clean_text(period.get("number") if isinstance(period, dict) else period),
                "clock_display": clean_text(clock.get("displayValue")),
                "wallclock": clean_text(event.get("wallclock")),
                "scoring_play": bool(event.get("scoringPlay")),
                "shootout": bool(event.get("shootout")),
                "score_value": parse_number(event.get("scoreValue")),
                "team": nested_name(team),
                "team_id": clean_text(team.get("id")) if isinstance(team, dict) else "",
                "athlete_id": clean_text(athlete.get("id")),
                "athlete_name": nested_name(athlete),
                "fetched_at": fetched_at,
                "source_url": source_url,
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], output: Path, fieldnames: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metadata = event_metadata(args.results, args.fixtures, args.include_scheduled)
    for event_id in args.event_id:
        event_id = clean_text(event_id)
        if event_id and event_id not in metadata:
            metadata[event_id] = {
                "espn_event_id": event_id,
                "date": "",
                "stage": "",
                "home_team": "",
                "away_team": "",
                "completed": "True",
            }

    event_ids = sorted(metadata.keys(), key=lambda item: (metadata[item].get("date", ""), item))
    if args.max_events and args.max_events > 0:
        event_ids = event_ids[: args.max_events]

    fetched_at = now_utc()
    all_lineups: list[dict[str, Any]] = []
    all_team_stats: list[dict[str, Any]] = []
    all_player_stats: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    errors: list[str] = []

    for index, event_id in enumerate(event_ids, start=1):
        try:
            payload = fetch_summary(event_id)
            lineups, player_stats = parse_rosters(event_id, payload, metadata[event_id], fetched_at)
            team_stats = parse_team_stats(event_id, payload, metadata[event_id], fetched_at)
            events = parse_key_events(event_id, payload, metadata[event_id], fetched_at)
            all_lineups.extend(lineups)
            all_player_stats.extend(player_stats)
            all_team_stats.extend(team_stats)
            all_events.extend(events)
            print(
                f"[{index}/{len(event_ids)}] {event_id}: "
                f"{len(lineups)} lineup rows, {len(team_stats)} team stats, "
                f"{len(player_stats)} player stats, {len(events)} events"
            )
        except Exception as exc:  # noqa: BLE001 - keep scraping robust in automation.
            errors.append(f"{event_id}: {exc}")
            print(f"[{index}/{len(event_ids)}] {event_id}: ERROR {exc}")
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    write_csv(all_lineups, args.lineups_output, LINEUP_FIELDS)
    write_csv(all_team_stats, args.team_stats_output, TEAM_STATS_FIELDS)
    write_csv(all_player_stats, args.player_stats_output, PLAYER_STATS_FIELDS)
    write_csv(all_events, args.events_output, EVENT_FIELDS)

    print(
        "Wrote "
        f"{len(all_lineups):,} lineup rows to {args.lineups_output}; "
        f"{len(all_team_stats):,} team stat rows to {args.team_stats_output}; "
        f"{len(all_player_stats):,} player stat rows to {args.player_stats_output}; "
        f"{len(all_events):,} event rows to {args.events_output}."
    )
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
