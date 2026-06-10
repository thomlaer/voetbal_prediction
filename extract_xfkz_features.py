"""
Extract compact country-level features from xfkzujqjvx97n/football-datasets.

The raw Kaggle dataset is large. This script keeps only dated, pre-match-safe
features that can be joined to historical international matches:

- country market-value snapshots at match dates
- position-specific market-value summaries
- active injury burden by country at match dates
- last completed season player-performance summaries by country

It also writes current national-player/squad files for future manual use, but
those are not used in historical training because all-time caps can leak future
information.
"""

from __future__ import annotations

import argparse
import heapq
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_XFKZ_DIR = Path("data/kagglehub/datasets/xfkzujqjvx97n/football-datasets/versions/2")
DEFAULT_RESULTS = Path("data/results.csv")
DEFAULT_OUTPUT_DIR = Path("data/extracted")


TEAM_ALIASES = {
    "cote d ivoire": "ivory coast",
    "cote divoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "cabo verde": "cape verde",
    "czechia": "czech republic",
    "dr congo": "congo dr",
    "congo dr": "congo dr",
    "ir iran": "iran",
    "korea republic": "south korea",
    "korea dpr": "north korea",
    "north macedonia": "macedonia",
    "people's republic of china": "china",
    "china pr": "china",
    "republic of ireland": "ireland",
    "turkiye": "turkey",
    "united states": "united states",
    "usa": "united states",
    "viet nam": "vietnam",
}


def normalize_name(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    for char in (".", ",", "-", "_", "(", ")", "&"):
        text = text.replace(char, " ")
    text = " ".join(text.replace("'", "").split())
    return TEAM_ALIASES.get(text, text)


def primary_citizenship(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    # Transfermarkt exports multiple citizenships separated by repeated spaces.
    first = [part.strip() for part in text.split("  ") if part.strip()]
    return normalize_name(first[0] if first else text)


def position_group(value: Any) -> str:
    text = str(value).strip().lower()
    if "goalkeeper" in text:
        return "goalkeeper"
    if "defender" in text or "back" in text:
        return "defender"
    if "midfield" in text:
        return "midfield"
    if "attack" in text or "forward" in text or "winger" in text or "striker" in text:
        return "attack"
    return "midfield"


def load_match_dates(results_path: Path, train_from: str) -> pd.DataFrame:
    results = pd.read_csv(results_path, usecols=["date", "home_team", "away_team", "home_score", "away_score"])
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(subset=["date", "home_score", "away_score"])
    results = results[results["date"] >= pd.Timestamp(train_from)].copy()
    results["home_key"] = results["home_team"].map(normalize_name)
    results["away_key"] = results["away_team"].map(normalize_name)
    return results


def top_value_features(
    values_by_player: dict[int, float],
    birth_year_by_player: dict[int, float],
    match_year: int,
) -> dict[str, float]:
    if not values_by_player:
        return {
            "market_player_count": 0.0,
            "market_top11_value": 0.0,
            "market_top18_value": 0.0,
            "market_top23_value": 0.0,
            "market_top50_value": 0.0,
            "market_top23_avg_value": 0.0,
            "market_top23_avg_age": 0.0,
        }
    top = heapq.nlargest(50, values_by_player.items(), key=lambda item: item[1])
    top_values = [float(value) for _, value in top]
    top23 = top[:23]
    ages = [
        float(match_year - birth_year_by_player[player_id])
        for player_id, _ in top23
        if player_id in birth_year_by_player and not pd.isna(birth_year_by_player[player_id])
    ]
    return {
        "market_player_count": float(len(values_by_player)),
        "market_top11_value": float(sum(top_values[:11])),
        "market_top18_value": float(sum(top_values[:18])),
        "market_top23_value": float(sum(top_values[:23])),
        "market_top50_value": float(sum(top_values[:50])),
        "market_top23_avg_value": float(np.mean(top_values[:23])) if top_values[:23] else 0.0,
        "market_top23_avg_age": float(np.mean(ages)) if ages else 0.0,
    }


def season_availability_date(value: Any) -> pd.Timestamp | pd.NaT:
    text = str(value).strip()
    parts = text.split("/")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return pd.NaT
    ending_year = int(parts[1])
    year = 2000 + ending_year if ending_year <= 30 else 1900 + ending_year
    return pd.Timestamp(year=year, month=7, day=1)


def current_season_name(as_of_date: pd.Timestamp) -> str:
    start_year = as_of_date.year if as_of_date.month >= 7 else as_of_date.year - 1
    return f"{start_year % 100:02d}/{(start_year + 1) % 100:02d}"


def top_performance_features(
    performance_by_player: dict[int, dict[str, float]],
    values_by_player: dict[int, float],
) -> dict[str, float]:
    empty = {
        "perf_player_count": 0.0,
        "perf_top11_minutes": 0.0,
        "perf_top23_minutes": 0.0,
        "perf_top23_apps": 0.0,
        "perf_top23_goals": 0.0,
        "perf_top23_assists": 0.0,
        "perf_top23_goal_contrib": 0.0,
        "perf_top23_goal_contrib_per90": 0.0,
        "perf_top23_cards": 0.0,
        "perf_top23_clean_sheets": 0.0,
        "perf_top23_goals_conceded": 0.0,
    }
    if not performance_by_player:
        return empty

    top_players = heapq.nlargest(
        23,
        performance_by_player.items(),
        key=lambda item: (
            values_by_player.get(item[0], 0.0),
            item[1].get("minutes_played", 0.0),
            item[1].get("goal_contrib", 0.0),
        ),
    )
    top11_players = top_players[:11]

    def total(players: list[tuple[int, dict[str, float]]], key: str) -> float:
        return float(sum(float(stats.get(key, 0.0)) for _, stats in players))

    minutes = total(top_players, "minutes_played")
    goal_contrib = total(top_players, "goal_contrib")
    return {
        "perf_player_count": float(len(performance_by_player)),
        "perf_top11_minutes": total(top11_players, "minutes_played"),
        "perf_top23_minutes": minutes,
        "perf_top23_apps": total(top_players, "nb_on_pitch"),
        "perf_top23_goals": total(top_players, "goals"),
        "perf_top23_assists": total(top_players, "assists"),
        "perf_top23_goal_contrib": goal_contrib,
        "perf_top23_goal_contrib_per90": float(goal_contrib * 90.0 / minutes) if minutes > 0 else 0.0,
        "perf_top23_cards": total(top_players, "cards"),
        "perf_top23_clean_sheets": total(top_players, "clean_sheets"),
        "perf_top23_goals_conceded": total(top_players, "goals_conceded"),
    }


def summarize_country_performance(performance: pd.DataFrame) -> pd.DataFrame:
    if performance.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | str]] = []
    for (availability_date, country), group in performance.groupby(["availability_date", "country_key"], sort=True):
        group = group.sort_values(
            ["minutes_played", "goal_contrib", "nb_on_pitch"],
            ascending=False,
        )
        top23 = group.head(23)
        top11 = group.head(11)
        minutes = float(top23["minutes_played"].sum())
        goal_contrib = float(top23["goal_contrib"].sum())
        rows.append(
            {
                "availability_date": availability_date,
                "country_key": country,
                "perf_player_count": float(len(group)),
                "perf_top11_minutes": float(top11["minutes_played"].sum()),
                "perf_top23_minutes": minutes,
                "perf_top23_apps": float(top23["nb_on_pitch"].sum()),
                "perf_top23_goals": float(top23["goals"].sum()),
                "perf_top23_assists": float(top23["assists"].sum()),
                "perf_top23_goal_contrib": goal_contrib,
                "perf_top23_goal_contrib_per90": goal_contrib * 90.0 / minutes if minutes > 0 else 0.0,
                "perf_top23_cards": float(top23["cards"].sum()),
                "perf_top23_clean_sheets": float(top23["clean_sheets"].sum()),
                "perf_top23_goals_conceded": float(top23["goals_conceded"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["availability_date", "country_key"]).reset_index(drop=True)


def attach_performance_snapshots(snapshots: pd.DataFrame, performance_snapshots: pd.DataFrame) -> pd.DataFrame:
    if performance_snapshots.empty:
        return snapshots
    output = snapshots.copy()
    output["row_id"] = range(len(output))
    output["date_ts"] = pd.to_datetime(output["date"], errors="coerce")
    performance = performance_snapshots.copy()
    performance["date_ts"] = pd.to_datetime(performance["availability_date"], errors="coerce")
    feature_cols = [
        col
        for col in performance.columns
        if col not in {"availability_date", "date_ts", "country_key"}
    ]

    pieces = []
    for country, group in output.groupby("country_key", sort=False):
        perf_country = performance[performance["country_key"] == country].sort_values("date_ts")
        group = group.sort_values("date_ts")
        if perf_country.empty:
            for col in feature_cols:
                group[col] = 0.0
            pieces.append(group)
            continue
        merged = pd.merge_asof(
            group,
            perf_country[["date_ts"] + feature_cols],
            on="date_ts",
            direction="backward",
        )
        pieces.append(merged)

    output = pd.concat(pieces, ignore_index=True).sort_values("row_id")
    for col in feature_cols:
        output[col] = pd.to_numeric(output[col], errors="coerce").fillna(0.0)
    return output.drop(columns=["row_id", "date_ts"])


def build_market_and_injury_snapshots(
    xfkz_dir: Path,
    results: pd.DataFrame,
    current_season_as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    profiles = pd.read_csv(
        xfkz_dir / "player_profiles/player_profiles.csv",
        usecols=["player_id", "citizenship", "date_of_birth", "main_position"],
    )
    profiles["country_key"] = profiles["citizenship"].map(primary_citizenship)
    profiles["position_group"] = profiles["main_position"].map(position_group)
    profiles["date_of_birth"] = pd.to_datetime(profiles["date_of_birth"], errors="coerce")
    profiles["birth_year"] = profiles["date_of_birth"].dt.year
    profiles = profiles[profiles["country_key"] != ""].drop_duplicates("player_id")
    player_country = dict(zip(profiles["player_id"], profiles["country_key"]))
    player_position = dict(zip(profiles["player_id"], profiles["position_group"]))
    player_birth_year = dict(zip(profiles["player_id"], profiles["birth_year"]))

    market = pd.read_csv(xfkz_dir / "player_market_value/player_market_value.csv")
    market["date"] = pd.to_datetime(market["date_unix"], errors="coerce")
    market["value"] = pd.to_numeric(market["value"], errors="coerce").fillna(0.0)
    market = market.dropna(subset=["date", "player_id"])
    market = market[market["player_id"].isin(player_country)].sort_values("date").reset_index(drop=True)

    injuries = pd.read_csv(
        xfkz_dir / "player_injuries/player_injuries.csv",
        usecols=["player_id", "from_date", "end_date", "days_missed", "games_missed"],
    )
    injuries["from_date"] = pd.to_datetime(injuries["from_date"], errors="coerce")
    injuries["end_date"] = pd.to_datetime(injuries["end_date"], errors="coerce")
    injuries["days_missed"] = pd.to_numeric(injuries["days_missed"], errors="coerce").fillna(0.0)
    injuries["games_missed"] = pd.to_numeric(injuries["games_missed"], errors="coerce").fillna(0.0)
    injuries = injuries.dropna(subset=["from_date", "player_id"])
    injuries["end_date"] = injuries["end_date"].fillna(injuries["from_date"] + pd.to_timedelta(injuries["days_missed"], unit="D"))
    injuries = injuries[injuries["player_id"].isin(player_country)].copy()
    injuries["country_key"] = injuries["player_id"].map(player_country)
    starts = injuries.sort_values("from_date").reset_index(drop=True)
    ends = injuries.sort_values("end_date").reset_index(drop=True)

    performance = pd.read_csv(
        xfkz_dir / "player_performances/player_performances.csv",
        usecols=[
            "player_id",
            "season_name",
            "nb_on_pitch",
            "goals",
            "assists",
            "yellow_cards",
            "second_yellow_cards",
            "direct_red_cards",
            "minutes_played",
            "goals_conceded",
            "clean_sheets",
        ],
    )
    performance = performance[performance["player_id"].isin(player_country)].copy()
    performance["availability_date"] = performance["season_name"].map(season_availability_date)
    if current_season_as_of is not None:
        season_name = current_season_name(current_season_as_of)
        performance.loc[performance["season_name"].eq(season_name), "availability_date"] = current_season_as_of
    performance = performance.dropna(subset=["availability_date"])
    performance["country_key"] = performance["player_id"].map(player_country)
    numeric_perf_cols = [
        "nb_on_pitch",
        "goals",
        "assists",
        "yellow_cards",
        "second_yellow_cards",
        "direct_red_cards",
        "minutes_played",
        "goals_conceded",
        "clean_sheets",
    ]
    for col in numeric_perf_cols:
        performance[col] = pd.to_numeric(performance[col], errors="coerce").fillna(0.0)
    performance["goal_contrib"] = performance["goals"] + performance["assists"]
    performance["cards"] = (
        performance["yellow_cards"]
        + 2.0 * performance["second_yellow_cards"]
        + 3.0 * performance["direct_red_cards"]
    )
    performance = (
        performance.groupby(["availability_date", "country_key", "player_id"], as_index=False)
        [
            [
                "nb_on_pitch",
                "goals",
                "assists",
                "minutes_played",
                "goals_conceded",
                "clean_sheets",
                "goal_contrib",
                "cards",
            ]
        ]
        .sum()
        .sort_values("availability_date")
        .reset_index(drop=True)
    )
    performance_snapshots = summarize_country_performance(performance)

    country_values: dict[str, dict[int, float]] = defaultdict(dict)
    country_position_values: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    active_injury_count: dict[str, float] = defaultdict(float)
    active_injury_days: dict[str, float] = defaultdict(float)
    active_injury_games: dict[str, float] = defaultdict(float)

    date_teams = (
        pd.concat(
            [
                results[["date", "home_key"]].rename(columns={"home_key": "country_key"}),
                results[["date", "away_key"]].rename(columns={"away_key": "country_key"}),
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values(["date", "country_key"])
    )
    if current_season_as_of is not None:
        current_rows = pd.DataFrame(
            {
                "date": current_season_as_of,
                "country_key": sorted(date_teams["country_key"].dropna().unique()),
            }
        )
        date_teams = (
            pd.concat([date_teams, current_rows], ignore_index=True)
            .drop_duplicates()
            .sort_values(["date", "country_key"])
        )

    market_idx = 0
    start_idx = 0
    end_idx = 0
    snapshots: list[dict[str, float | str]] = []
    for date, group in date_teams.groupby("date", sort=True):
        while market_idx < len(market) and market.at[market_idx, "date"] <= date:
            player_id = int(market.at[market_idx, "player_id"])
            value = float(market.at[market_idx, "value"])
            country = player_country.get(player_id)
            position = player_position.get(player_id, "midfield")
            if country:
                if value > 0:
                    country_values[country][player_id] = value
                    country_position_values[(country, position)][player_id] = value
                else:
                    country_values[country].pop(player_id, None)
                    country_position_values[(country, position)].pop(player_id, None)
            market_idx += 1

        while start_idx < len(starts) and starts.at[start_idx, "from_date"] <= date:
            country = starts.at[start_idx, "country_key"]
            active_injury_count[country] += 1.0
            active_injury_days[country] += float(starts.at[start_idx, "days_missed"])
            active_injury_games[country] += float(starts.at[start_idx, "games_missed"])
            start_idx += 1

        while end_idx < len(ends) and ends.at[end_idx, "end_date"] < date:
            country = ends.at[end_idx, "country_key"]
            active_injury_count[country] = max(0.0, active_injury_count[country] - 1.0)
            active_injury_days[country] = max(0.0, active_injury_days[country] - float(ends.at[end_idx, "days_missed"]))
            active_injury_games[country] = max(0.0, active_injury_games[country] - float(ends.at[end_idx, "games_missed"]))
            end_idx += 1

        match_year = int(pd.Timestamp(date).year)
        for country in group["country_key"].unique():
            row = {"date": pd.Timestamp(date).date().isoformat(), "country_key": country}
            row.update(top_value_features(country_values.get(country, {}), player_birth_year, match_year))
            for position in ("attack", "midfield", "defender", "goalkeeper"):
                pos_features = top_value_features(
                    country_position_values.get((country, position), {}),
                    player_birth_year,
                    match_year,
                )
                row[f"{position}_market_top5_value"] = float(
                    sum(heapq.nlargest(5, country_position_values.get((country, position), {}).values()))
                )
                row[f"{position}_market_player_count"] = pos_features["market_player_count"]
            row["active_injured_players"] = float(active_injury_count[country])
            row["active_injury_days_missed"] = float(active_injury_days[country])
            row["active_injury_games_missed"] = float(active_injury_games[country])
            snapshots.append(row)

    return attach_performance_snapshots(pd.DataFrame(snapshots), performance_snapshots)


def build_current_national_extracts(xfkz_dir: Path, output_dir: Path) -> None:
    profiles = pd.read_csv(
        xfkz_dir / "player_profiles/player_profiles.csv",
        usecols=[
            "player_id",
            "player_name",
            "citizenship",
            "date_of_birth",
            "main_position",
            "current_club_name",
        ],
    )
    profiles["country_key"] = profiles["citizenship"].map(primary_citizenship)
    profiles["position_group"] = profiles["main_position"].map(position_group)
    latest_values = pd.read_csv(xfkz_dir / "player_latest_market_value/player_latest_market_value.csv")
    latest_values["latest_market_value"] = pd.to_numeric(latest_values["value"], errors="coerce").fillna(0.0)
    national = pd.read_csv(xfkz_dir / "player_national_performances/player_national_performances.csv")
    current = national[national["career_state"].isin(["CURRENT_NATIONAL_PLAYER", "RECENT_NATIONAL_PLAYER"])].copy()
    current = current.merge(profiles, on="player_id", how="left").merge(
        latest_values[["player_id", "latest_market_value"]],
        on="player_id",
        how="left",
    )
    current["latest_market_value"] = current["latest_market_value"].fillna(0.0)
    current = current[current["country_key"].notna() & current["country_key"].ne("")]
    current.to_csv(output_dir / "xfkz_current_national_players.csv", index=False)

    squad_rows = []
    for country, group in current.groupby("country_key"):
        group = group.sort_values("latest_market_value", ascending=False)
        squad_rows.append(
            {
                "country_key": country,
                "current_recent_player_count": int(len(group)),
                "current_recent_caps": float(pd.to_numeric(group["matches"], errors="coerce").fillna(0.0).sum()),
                "current_recent_goals": float(pd.to_numeric(group["goals"], errors="coerce").fillna(0.0).sum()),
                "current_top11_market_value": float(group["latest_market_value"].head(11).sum()),
                "current_top18_market_value": float(group["latest_market_value"].head(18).sum()),
                "current_top23_market_value": float(group["latest_market_value"].head(23).sum()),
                "current_top23_avg_market_value": float(group["latest_market_value"].head(23).mean() or 0.0),
            }
        )
    pd.DataFrame(squad_rows).to_csv(output_dir / "xfkz_current_squad_strength.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xfkz-dir", type=Path, default=DEFAULT_XFKZ_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-from", default="1993-01-01")
    parser.add_argument(
        "--current-season-as-of",
        default=None,
        help="Optional date that makes the currently running club season available for future fixtures only.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = load_match_dates(args.results, args.train_from)
    current_as_of = pd.Timestamp(args.current_season_as_of) if args.current_season_as_of else None
    snapshots = build_market_and_injury_snapshots(args.xfkz_dir, results, current_as_of)
    snapshots.to_csv(args.output_dir / "xfkz_country_market_injury_snapshots.csv", index=False)
    build_current_national_extracts(args.xfkz_dir, args.output_dir)
    print(f"Wrote {len(snapshots):,} country-date snapshots to {args.output_dir}")


if __name__ == "__main__":
    main()
