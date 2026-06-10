"""
Convert the unofficial FIFA World Cup 2026 Kaggle dataset to model fixtures.

The output can be passed to train_xgboost_worldcup.py with --future-fixtures.
Rows with placeholder teams are kept in the full export and excluded from the
ready export because the model cannot predict "Winner UEFA Playoff D".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path("data/kagglehub/datasets/areezvisram12/fifa-world-cup-2026-match-data-unofficial/versions/3")
DEFAULT_OUTPUT = Path("data/extracted/worldcup2026_future_fixtures.csv")
DEFAULT_READY_OUTPUT = Path("data/extracted/worldcup2026_future_fixtures_known_teams.csv")

TEAM_ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
}
HOST_COUNTRY_ALIASES = {
    "USA": "United States",
}


def clean_team_name(name: object) -> str:
    text = str(name).strip()
    return TEAM_ALIASES.get(text, text)


def clean_host_country(name: object) -> str:
    text = str(name).strip()
    return HOST_COUNTRY_ALIASES.get(text, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare World Cup 2026 future fixtures for the model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ready-output", type=Path, default=DEFAULT_READY_OUTPUT)
    args = parser.parse_args()

    matches = pd.read_csv(args.data_dir / "matches.csv")
    teams = pd.read_csv(args.data_dir / "teams.csv")
    cities = pd.read_csv(args.data_dir / "host_cities.csv")
    stages = pd.read_csv(args.data_dir / "tournament_stages.csv")

    home = teams.add_prefix("home_")
    away = teams.add_prefix("away_")
    city = cities.add_prefix("city_")
    stage = stages.add_prefix("stage_")

    fixtures = (
        matches.merge(home, left_on="home_team_id", right_on="home_id", how="left")
        .merge(away, left_on="away_team_id", right_on="away_id", how="left")
        .merge(city, left_on="city_id", right_on="city_id", how="left")
        .merge(stage, left_on="stage_id", right_on="stage_id", how="left")
    )

    output = pd.DataFrame(
        {
            "date": fixtures["kickoff_at"].astype(str).str.slice(0, 10),
            "kickoff_at": fixtures["kickoff_at"],
            "match_number": fixtures["match_number"],
            "home_team": fixtures["home_team_name"].map(clean_team_name),
            "away_team": fixtures["away_team_name"].map(clean_team_name),
            "tournament": "FIFA World Cup",
            "stage": fixtures.get("stage_stage_name", fixtures.get("match_label", "")),
            "group": fixtures["home_group_letter"].fillna(fixtures["match_label"]),
            "city": fixtures["city_city_name"],
            "country": fixtures["city_country"].map(clean_host_country),
            "venue": fixtures["city_venue_name"],
            "region_cluster": fixtures["city_region_cluster"],
            "airport_code": fixtures["city_airport_code"],
            "neutral": True,
            "home_is_placeholder": fixtures["home_is_placeholder"].astype(bool),
            "away_is_placeholder": fixtures["away_is_placeholder"].astype(bool),
        }
    )
    output["neutral"] = ~(
        output["home_team"].eq(output["country"]) | output["away_team"].eq(output["country"])
    )
    output = output.sort_values(["date", "match_number"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    ready = output[~output["home_is_placeholder"] & ~output["away_is_placeholder"]].copy()
    ready.to_csv(args.ready_output, index=False)

    print(f"Wrote {len(output):,} fixtures to {args.output}")
    print(f"Wrote {len(ready):,} known-team fixtures to {args.ready_output}")


if __name__ == "__main__":
    main()
