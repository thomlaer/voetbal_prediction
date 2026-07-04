#!/usr/bin/env python
"""Compare fixture-xG and rounded-score topscorer rankings by World Cup round.

The backtest uses tournament squads and information available before each
round. Actual goals from the round are used only for evaluation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import make_scorito_worldcup_picks as scorer


TOURNAMENTS = {
    2018: {
        "predictions": Path("outputs_backtest_2018/test_predictions_xgboost.csv"),
        "cutoff": "2018-06-13",
        "start": "2018-06-14",
        "end": "2018-07-15",
        "tournament_id": "WC-2018",
    },
    2022: {
        "predictions": Path("outputs_backtest_2022/test_predictions_xgboost.csv"),
        "cutoff": "2022-11-19",
        "start": "2022-11-20",
        "end": "2022-12-18",
        "tournament_id": "WC-2022",
    },
}
STAGES = ["Group Stage", "Round of 16", "Quarterfinals", "Semifinals", "Final/Third"]
STAGE_BY_MATCH_INDEX = (
    ["Group Stage"] * 48
    + ["Round of 16"] * 8
    + ["Quarterfinals"] * 4
    + ["Semifinals"] * 2
    + ["Final/Third"] * 2
)
STAGE_ORDER = {stage: index for index, stage in enumerate(STAGES)}
STAGE_POINTS = {
    "Group Stage": {"GK": 0, "DF": 32, "MF": 16, "FW": 8},
    "Round of 16": {"GK": 0, "DF": 96, "MF": 48, "FW": 24},
    "Quarterfinals": {"GK": 0, "DF": 128, "MF": 64, "FW": 32},
    "Semifinals": {"GK": 0, "DF": 160, "MF": 80, "FW": 40},
    "Final/Third": {"GK": 0, "DF": 192, "MF": 96, "FW": 48},
}
POSITION_WEIGHT = {"GK": 0.0, "DF": 0.10, "MF": 0.38, "FW": 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_topscorer_round_backtest"))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--years", type=int, nargs="+", choices=sorted(TOURNAMENTS), default=sorted(TOURNAMENTS))
    return parser.parse_args()


def position_code(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("G"):
        return "GK"
    if text.startswith("D"):
        return "DF"
    if text.startswith("F"):
        return "FW"
    return "MF"


def player_name(given: Any, family: Any) -> str:
    ignored = {"", "nan", "none", "not applicable", "not available"}
    parts = [str(value or "").strip() for value in (given, family)]
    return " ".join(part for part in parts if part.lower() not in ignored)


def stage_name(value: Any) -> str:
    text = str(value or "").lower()
    if "group" in text:
        return "Group Stage"
    if "round of 16" in text:
        return "Round of 16"
    if "quarter" in text:
        return "Quarterfinals"
    if "semi" in text:
        return "Semifinals"
    if "third" in text or "final" in text:
        return "Final/Third"
    return str(value)


def load_matches(data_root: Path, year: int) -> pd.DataFrame:
    config = TOURNAMENTS[year]
    path = data_root / config["predictions"]
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame[
        frame["tournament"].eq("FIFA World Cup")
        & frame["date"].between(config["start"], config["end"])
    ].sort_values("date").reset_index(drop=True)
    if len(frame) != 64:
        raise ValueError(f"Expected 64 World Cup {year} matches in {path}, got {len(frame)}")
    frame["stage"] = STAGE_BY_MATCH_INDEX
    return frame


def load_candidates(data_root: Path, year: int) -> pd.DataFrame:
    config = TOURNAMENTS[year]
    squads = pd.read_csv(data_root / "data/fjelstul_worldcup/data-csv/squads.csv")
    squads = squads[squads["tournament_id"].eq(config["tournament_id"])].copy()
    squads["player"] = [player_name(given, family) for given, family in zip(squads["given_name"], squads["family_name"])]
    squads["team"] = squads["team_name"].astype(str)
    squads["position"] = squads["position_code"].map(position_code)
    squads["team_key"] = squads["team"].map(scorer.normalize_name)
    squads["player_key"] = squads["player"].map(scorer.normalize_person_name)

    wiki = pd.read_csv(data_root / "data/extracted/wikipedia_tournament_squads.csv")
    wiki = wiki[wiki["competition"].eq("FIFA World Cup") & wiki["year"].eq(year)].copy()
    wiki["team_key"] = wiki["team"].map(scorer.normalize_name)
    wiki["player_key"] = wiki["player"].map(scorer.normalize_person_name)
    candidates = squads.merge(
        wiki[["team_key", "player_key", "caps", "goals"]],
        on=["team_key", "player_key"],
        how="left",
    )
    candidates["caps"] = pd.to_numeric(candidates["caps"], errors="coerce").fillna(0.0)
    candidates["goals"] = pd.to_numeric(candidates["goals"], errors="coerce").fillna(0.0)

    sofifa = scorer.latest_sofifa_by_player(
        data_root / "data/extracted/sofifa_yearly_player_ratings.csv",
        cutoff=config["cutoff"],
    )
    indexes = scorer.build_sofifa_indexes(sofifa)
    ratings: list[dict[str, float]] = []
    for _, player in candidates.iterrows():
        matched = scorer.match_sofifa_player(player, indexes)
        overall = scorer.numeric(matched.get("overall") if matched is not None else np.nan, 65.0)
        shooting = scorer.numeric(matched.get("shooting") if matched is not None else np.nan, overall)
        ratings.append(
            {
                "sofifa_overall": overall,
                "sofifa_shooting": shooting,
                "sofifa_finishing": scorer.numeric(
                    matched.get("attacking_finishing") if matched is not None else np.nan,
                    shooting,
                ),
                "sofifa_positioning": scorer.numeric(
                    matched.get("mentality_positioning") if matched is not None else np.nan,
                    shooting,
                ),
                "sofifa_reactions": scorer.numeric(
                    matched.get("movement_reactions") if matched is not None else np.nan,
                    overall,
                ),
            }
        )
    return pd.concat([candidates.reset_index(drop=True), pd.DataFrame(ratings)], axis=1)


def load_actual_events(data_root: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tournament_id = TOURNAMENTS[year]["tournament_id"]
    goals = pd.read_csv(data_root / "data/fjelstul_worldcup/data-csv/goals.csv")
    goals = goals[goals["tournament_id"].eq(tournament_id) & goals["own_goal"].eq(0)].copy()
    goals["stage"] = goals["stage_name"].map(stage_name)
    goals["stage_order"] = goals["stage"].map(STAGE_ORDER)
    goals["player"] = [player_name(given, family) for given, family in zip(goals["given_name"], goals["family_name"])]
    goals["team_key"] = goals["player_team_name"].map(scorer.normalize_name)
    goals["player_key"] = goals["player"].map(scorer.normalize_person_name)

    appearances = pd.read_csv(data_root / "data/fjelstul_worldcup/data-csv/player_appearances.csv")
    appearances = appearances[appearances["tournament_id"].eq(tournament_id)].copy()
    appearances["stage"] = appearances["stage_name"].map(stage_name)
    appearances["stage_order"] = appearances["stage"].map(STAGE_ORDER)
    appearances["player"] = [
        player_name(given, family)
        for given, family in zip(appearances["given_name"], appearances["family_name"])
    ]
    appearances["team_key"] = appearances["team_name"].map(scorer.normalize_name)
    appearances["player_key"] = appearances["player"].map(scorer.normalize_person_name)
    return goals, appearances


def team_stage_goals(matches: pd.DataFrame, source: str) -> pd.DataFrame:
    if source == "fixture_xg":
        home_column, away_column = "expected_home_goals", "expected_away_goals"
    elif source == "rounded_score":
        home_column, away_column = "pool_predicted_home_score", "pool_predicted_away_score"
    else:
        home_column, away_column = "home_score", "away_score"
    rows: list[dict[str, Any]] = []
    for row in matches.itertuples(index=False):
        rows.extend(
            [
                {
                    "stage": row.stage,
                    "team_key": scorer.normalize_name(row.home_team),
                    "team_goals": float(getattr(row, home_column)),
                },
                {
                    "stage": row.stage,
                    "team_key": scorer.normalize_name(row.away_team),
                    "team_goals": float(getattr(row, away_column)),
                },
            ]
        )
    return pd.DataFrame(rows).groupby(["stage", "team_key"], as_index=False)["team_goals"].sum()


def player_context(
    candidates: pd.DataFrame,
    predicted_team_goals: pd.DataFrame,
    goals: pd.DataFrame,
    appearances: pd.DataFrame,
    stage: str,
) -> pd.DataFrame:
    stage_index = STAGE_ORDER[stage]
    frame = candidates.merge(
        predicted_team_goals[predicted_team_goals["stage"].eq(stage)][["team_key", "team_goals"]],
        on="team_key",
        how="inner",
    )
    past_goals = (
        goals[goals["stage_order"].lt(stage_index)]
        .groupby(["team_key", "player_key"])
        .size()
        .rename("past_goals")
        .reset_index()
    )
    past_penalties = (
        goals[goals["stage_order"].lt(stage_index) & goals["penalty"].eq(1)]
        .groupby(["team_key", "player_key"])
        .size()
        .rename("past_penalty_goals")
        .reset_index()
    )
    actual_stage_goals = (
        goals[goals["stage"].eq(stage)]
        .groupby(["team_key", "player_key"])
        .size()
        .rename("actual_stage_goals")
        .reset_index()
    )
    past_appearances = (
        appearances[appearances["stage_order"].lt(stage_index)]
        .groupby(["team_key", "player_key"])
        .agg(past_apps=("match_id", "nunique"), past_starts=("starter", "sum"))
        .reset_index()
    )
    for context in (past_goals, past_penalties, actual_stage_goals, past_appearances):
        frame = frame.merge(context, on=["team_key", "player_key"], how="left")
    for column in ("past_goals", "past_penalty_goals", "actual_stage_goals", "past_apps", "past_starts"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["stage"] = stage
    frame["points_per_goal"] = frame["position"].map(STAGE_POINTS[stage]).fillna(0.0)
    frame["actual_points"] = frame["actual_stage_goals"] * frame["points_per_goal"]
    return frame


def rank_players(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    caps = frame["caps"].astype(float)
    international_goals = frame["goals"].astype(float)
    goal_rate = international_goals / np.maximum(caps, 8.0)
    international_signal = 0.65 * np.minimum(goal_rate / 0.45, 1.25) + 0.35 * np.minimum(
        international_goals / 80.0,
        1.25,
    )
    attack_quality = (
        0.34 * frame["sofifa_finishing"] / 100.0
        + 0.24 * frame["sofifa_shooting"] / 100.0
        + 0.18 * frame["sofifa_positioning"] / 100.0
        + 0.14 * frame["sofifa_overall"] / 100.0
        + 0.10 * frame["sofifa_reactions"] / 100.0
    )
    base_weight = frame["position"].map(POSITION_WEIGHT).fillna(0.0) * attack_quality * (
        0.78 + 0.78 * scorer.TOPSCORER_INTL_MULT * international_signal
    )
    base_weight *= np.where(international_goals >= 20, 1.10, 1.0)
    base_weight *= np.where((caps >= 45) & frame["position"].map(POSITION_WEIGHT).fillna(0.0).ge(0.35), 1.05, 1.0)
    form = (
        1.0
        + scorer.TOPSCORER_PAST_GOAL_BONUS * np.minimum(frame["past_goals"], 4.0)
        + scorer.TOPSCORER_PAST_START_BONUS * np.sqrt(np.minimum(frame["past_starts"], 7.0))
        + scorer.TOPSCORER_PAST_APP_BONUS * np.sqrt(np.minimum(frame["past_apps"], 7.0))
        + scorer.TOPSCORER_PENALTY_GOAL_BONUS * np.minimum(frame["past_penalty_goals"], 2.0)
    )
    frame["adjusted_weight"] = base_weight * form
    frame["team_weight_sum"] = frame.groupby("team_key")["adjusted_weight"].transform("sum")
    frame["goal_share"] = np.where(frame["team_weight_sum"].gt(0), frame["adjusted_weight"] / frame["team_weight_sum"], 0.0)
    frame["predicted_expected_goals"] = frame["team_goals"] * frame["goal_share"]
    max_weight = max(float(base_weight.max()), 0.01)
    star_power = (
        0.48 * (base_weight / max_weight).clip(0.0, 1.15)
        + 0.28 * (frame["sofifa_overall"] / 91.0).clip(0.0, 1.08)
        + 0.24 * (international_goals / 70.0).clip(0.0, 1.15)
    )
    frame["rank_score"] = frame["predicted_expected_goals"] * frame["points_per_goal"] * (
        1.0 - scorer.TOPSCORER_STAR_BLEND + scorer.TOPSCORER_STAR_BLEND * star_power
    )
    return frame.sort_values(["rank_score", "predicted_expected_goals"], ascending=False)


def backtest_year(
    data_root: Path,
    year: int,
    top_n: int,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], pd.DataFrame]:
    matches = load_matches(data_root, year)
    candidates = load_candidates(data_root, year)
    goals, appearances = load_actual_events(data_root, year)
    actual_team_goals = team_stage_goals(matches, "actual")
    summary: list[dict[str, Any]] = []
    all_picks: list[pd.DataFrame] = []

    actual_leaders = (
        goals.groupby(["stage", "team_key", "player_key", "player", "player_team_name"], as_index=False)
        .size()
        .rename(columns={"size": "actual_stage_goals", "player_team_name": "team"})
        .merge(
            candidates[["team_key", "player_key", "position"]],
            on=["team_key", "player_key"],
            how="left",
        )
    )
    actual_leaders["points_per_goal"] = [
        STAGE_POINTS[stage].get(position_code(position), 0)
        for stage, position in zip(actual_leaders["stage"], actual_leaders["position"])
    ]
    actual_leaders["actual_points"] = (
        actual_leaders["actual_stage_goals"] * actual_leaders["points_per_goal"]
    )
    actual_leaders = actual_leaders.sort_values(
        ["stage", "actual_stage_goals", "actual_points", "player"],
        ascending=[True, False, False, True],
    )
    actual_leaders["actual_rank"] = actual_leaders.groupby("stage").cumcount() + 1
    actual_leaders.insert(0, "year", year)
    actual_leaders = actual_leaders[actual_leaders["actual_rank"].le(top_n)].copy()

    for source in ("fixture_xg", "rounded_score"):
        predicted_team_goals = team_stage_goals(matches, source)
        for stage in STAGES:
            ranked = rank_players(
                player_context(candidates, predicted_team_goals, goals, appearances, stage)
            )
            selected = ranked.head(top_n).copy()
            selected.insert(0, "selected_rank", np.arange(1, len(selected) + 1))
            selected.insert(0, "source", source)
            selected.insert(0, "year", year)
            all_picks.append(selected)

            team_comparison = predicted_team_goals[predicted_team_goals["stage"].eq(stage)].merge(
                actual_team_goals[actual_team_goals["stage"].eq(stage)],
                on=["stage", "team_key"],
                suffixes=("_predicted", "_actual"),
            )
            error = team_comparison["team_goals_predicted"] - team_comparison["team_goals_actual"]
            summary.append(
                {
                    "year": year,
                    "source": source,
                    "stage": stage,
                    "top_n": top_n,
                    "top_n_actual_goals": float(selected["actual_stage_goals"].sum()),
                    "top_n_scorer_hits": int(selected["actual_stage_goals"].gt(0).sum()),
                    "top_n_actual_points": float(selected["actual_points"].sum()),
                    "team_goal_mae": float(error.abs().mean()),
                    "team_goal_rmse": float(math.sqrt(float(np.square(error).mean()))),
                    "predicted_team_goals": float(team_comparison["team_goals_predicted"].sum()),
                    "actual_team_goals": float(team_comparison["team_goals_actual"].sum()),
                }
            )
    return summary, all_picks, actual_leaders


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    pick_frames: list[pd.DataFrame] = []
    actual_leader_frames: list[pd.DataFrame] = []
    for year in args.years:
        year_summary, year_picks, actual_leaders = backtest_year(data_root, year, args.top_n)
        summary_rows.extend(year_summary)
        pick_frames.extend(year_picks)
        actual_leader_frames.append(actual_leaders)

    summary = pd.DataFrame(summary_rows)
    picks = pd.concat(pick_frames, ignore_index=True)
    actual_leaders = pd.concat(actual_leader_frames, ignore_index=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    picks[
        [
            "year",
            "source",
            "stage",
            "selected_rank",
            "player",
            "team",
            "position",
            "predicted_expected_goals",
            "rank_score",
            "actual_stage_goals",
            "actual_points",
        ]
    ].to_csv(args.output_dir / "picks.csv", index=False)
    actual_leaders[
        [
            "year",
            "stage",
            "actual_rank",
            "player",
            "team",
            "position",
            "actual_stage_goals",
            "actual_points",
        ]
    ].to_csv(args.output_dir / "actual_leaders.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {args.output_dir / 'summary.csv'}")
    print(f"Wrote {args.output_dir / 'picks.csv'}")
    print(f"Wrote {args.output_dir / 'actual_leaders.csv'}")


if __name__ == "__main__":
    main()
