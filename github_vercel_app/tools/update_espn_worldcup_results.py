#!/usr/bin/env python
"""Fetch completed World Cup 2026 scores from ESPN's public scoreboard endpoint."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Any

import requests


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = APP_ROOT.parent
DEFAULT_OUTPUT = DEFAULT_MODEL_ROOT / "data" / "extracted" / "espn_worldcup2026_results.csv"
DEFAULT_DATES = "20260611-20260720"
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch completed World Cup scores from ESPN.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dates", default=DEFAULT_DATES)
    parser.add_argument("--limit", type=int, default=200)
    return parser.parse_args()


def team_name(competitor: dict[str, Any]) -> str:
    team = competitor.get("team") or {}
    return str(team.get("displayName") or team.get("shortDisplayName") or team.get("name") or "").strip()


def fetch_scores(dates: str, limit: int) -> list[dict[str, Any]]:
    response = requests.get(
        SCOREBOARD_URL,
        params={"limit": str(limit), "dates": dates},
        headers={"User-Agent": "voetbal-prediction/1.0"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    rows: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competition = competitions[0]
        status = (competition.get("status") or event.get("status") or {}).get("type") or {}
        if not status.get("completed"):
            continue
        competitors = competition.get("competitors") or []
        home = next((item for item in competitors if item.get("homeAway") == "home"), None)
        away = next((item for item in competitors if item.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        try:
            home_score = int(float(home.get("score")))
            away_score = int(float(away.get("score")))
        except (TypeError, ValueError):
            continue
        event_date = str(event.get("date") or competition.get("date") or "")[:10]
        if not event_date.startswith("2026-"):
            continue
        rows.append(
            {
                "source": "espn",
                "espn_event_id": event.get("id", ""),
                "date": event_date,
                "home_team": team_name(home),
                "away_team": team_name(away),
                "home_score": home_score,
                "away_score": away_score,
                "status": status.get("name") or status.get("description") or "",
                "completed": True,
                "fetched_at": date.today().isoformat(),
                "source_url": SCOREBOARD_URL,
            }
        )
    rows.sort(key=lambda row: (row["date"], str(row["espn_event_id"])))
    return rows


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "espn_event_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "status",
        "completed",
        "fetched_at",
        "source_url",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = fetch_scores(args.dates, args.limit)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows):,} ESPN completed World Cup results to {args.output}")


if __name__ == "__main__":
    main()
