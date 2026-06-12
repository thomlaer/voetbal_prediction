#!/usr/bin/env python
"""Build pragmatic World Cup 2026 pool picks from the current model outputs.

The default XGBoost model remains untouched. This script adds a betting-pool
decision layer: choose scorelines by expected pool points, keep them consistent
with the model/odds favourite except in genuinely close matches, and create
champion/topscorer advice files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_xgboost_worldcup import (
    normalize_name,
    normalize_person_name,
    person_name_match_score,
    poisson_pmf,
)


DEFAULT_OUTPUT_DIR = Path("outputs_worldcup2026_default")
DEFAULT_GROUP_PREDICTIONS = DEFAULT_OUTPUT_DIR / "worldcup2026_group_match_predictions.csv"
DEFAULT_BRACKET = DEFAULT_OUTPUT_DIR / "worldcup2026_bracket_prediction.csv"
DEFAULT_TEAM_PROBS = DEFAULT_OUTPUT_DIR / "worldcup2026_montecarlo_team_probabilities.csv"
DEFAULT_SCHEDULE = Path("data/extracted/oddsportal_worldcup2026_fixture_odds_schedule.csv")
DEFAULT_SQUADS = Path("data/extracted/wikipedia_worldcup2026_squads_finalish.csv")
DEFAULT_SOFIFA = Path("data/extracted/sofifa_yearly_player_ratings.csv")
DEFAULT_GOALSCORERS = Path("data/goalscorers.csv")


OUTCOME_COLUMNS = ["prob_home_win", "prob_draw", "prob_away_win"]
TOPSCORER_COLUMNS = [
    "rank",
    "team",
    "player",
    "position",
    "caps",
    "international_goals",
    "sofifa_matched",
    "sofifa_overall",
    "sofifa_shooting",
    "sofifa_finishing",
    "team_expected_matches",
    "team_expected_group_goals",
    "team_expected_goals",
    "team_champion_prob",
    "raw_scorer_weight",
    "team_weight_sum",
    "goal_share",
    "expected_group_stage_goals",
    "expected_goals",
    "prob_4plus_goals",
    "topscorer_score",
    "star_scorer_power",
    "recommended_topscorer_score",
    "golden_boot_rank",
    "scorito_points_per_goal",
    "expected_group_stage_scorito_points",
    "expected_scorito_points",
    "recommended_group_stage_topscorer_score",
    "recommended_scorito_topscorer_score",
    "group_stage_rank",
]
WORLD_CUP_2022_TUNED_SCORE_PARAMS = {
    "draw_margin": 0.18,
    "draw_min_prob": 0.26,
    "draw_00_total": 1.90,
    "draw_22_total": 3.20,
    "strong_prob": 0.72,
    "strong_30_fav_xg": 2.15,
    "strong_30_dog_xg": 0.60,
    "strong_21_dog_xg": 0.90,
    "strong_21_total": 2.80,
    "medium_prob": 0.52,
    "medium_21_total": 2.10,
    "medium_21_dog_xg": 0.80,
    "medium_20_fav_xg": 1.70,
    "medium_20_dog_xg": 0.70,
    "small_21_total": 1.90,
    "small_21_dog_xg": 0.65,
    "small_20_fav_xg": 1.60,
    "small_20_dog_xg": 0.70,
}
ROBUST_HYBRID_SCORE_PARAMS = {
    "fav_prob_tuned_below": 0.45,
    "close_draw_margin": 0.08,
    "close_draw_min_prob": 0.25,
    "total_xg_tuned_at_least": 2.40,
    "close_second_winner_margin": 0.08,
    "close_second_winner_draw_min": 0.28,
    "close_second_winner_draw_max": 0.30,
    "close_second_winner_open_total_xg": 2.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Scorito-style World Cup 2026 advice outputs.")
    parser.add_argument("--group-predictions", type=Path, default=DEFAULT_GROUP_PREDICTIONS)
    parser.add_argument("--bracket", type=Path, default=DEFAULT_BRACKET)
    parser.add_argument("--team-probabilities", type=Path, default=DEFAULT_TEAM_PROBS)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--squads", type=Path, default=DEFAULT_SQUADS)
    parser.add_argument("--sofifa", type=Path, default=DEFAULT_SOFIFA)
    parser.add_argument("--goalscorers", type=Path, default=DEFAULT_GOALSCORERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-goals", type=int, default=6)
    parser.add_argument("--close-margin", type=float, default=0.10)
    parser.add_argument("--exact-points", type=float, default=5.0)
    parser.add_argument("--outcome-points", type=float, default=2.0)
    parser.add_argument("--goal-diff-points", type=float, default=1.0)
    parser.add_argument("--team-goal-points", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def outcome_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def clean_player_name(value: Any) -> str:
    name = str(value or "").strip()
    name = re.sub(r"\s*\((?:captain|vice-captain)\)\s*", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip()


def score_points(
    pred_home: int,
    pred_away: int,
    actual_home: int,
    actual_away: int,
    exact_points: float,
    outcome_points: float,
    goal_diff_points: float,
    team_goal_points: float,
) -> float:
    if pred_home == actual_home and pred_away == actual_away:
        return exact_points

    points = 0.0
    if outcome_label(pred_home, pred_away) == outcome_label(actual_home, actual_away):
        points += outcome_points
    if (pred_home - pred_away) == (actual_home - actual_away):
        points += goal_diff_points
    if pred_home == actual_home:
        points += team_goal_points
    if pred_away == actual_away:
        points += team_goal_points
    return points


def calibrated_score_matrix(
    home_xg: float,
    away_xg: float,
    p_home: float,
    p_draw: float,
    p_away: float,
    max_goals: int,
) -> np.ndarray:
    home_probs = poisson_pmf(home_xg, max_goals)
    away_probs = poisson_pmf(away_xg, max_goals)
    matrix = np.outer(home_probs, away_probs)
    home_grid, away_grid = np.indices(matrix.shape)
    masks = {
        "home_win": home_grid > away_grid,
        "draw": home_grid == away_grid,
        "away_win": home_grid < away_grid,
    }
    targets = {"home_win": p_home, "draw": p_draw, "away_win": p_away}

    calibrated = np.zeros_like(matrix, dtype=float)
    for outcome, mask in masks.items():
        mass = float(matrix[mask].sum())
        target = float(max(targets[outcome], 0.0))
        if mass > 0 and target > 0:
            calibrated[mask] = matrix[mask] * (target / mass)
    total = float(calibrated.sum())
    if total <= 0:
        return matrix / matrix.sum()
    return calibrated / total


def candidate_scores(max_goals: int) -> list[tuple[int, int]]:
    scores: list[tuple[int, int]] = []
    for home in range(max_goals + 1):
        for away in range(max_goals + 1):
            # Keep the pool advice realistic. 6-5 exists in the matrix, but it
            # is almost never a sensible manual pool pick.
            if home + away <= 7:
                scores.append((home, away))
    return scores


def optimize_score_for_pool(
    row: pd.Series,
    max_goals: int,
    close_margin: float,
    exact_points: float,
    outcome_points: float,
    goal_diff_points: float,
    team_goal_points: float,
) -> dict[str, Any]:
    probs = {
        "home_win": float(row["prob_home_win"]),
        "draw": float(row["prob_draw"]),
        "away_win": float(row["prob_away_win"]),
    }
    ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    favourite, favourite_prob = ordered[0]
    margin = favourite_prob - ordered[1][1]

    matrix = calibrated_score_matrix(
        float(row["expected_home_goals"]),
        float(row["expected_away_goals"]),
        probs["home_win"],
        probs["draw"],
        probs["away_win"],
        max_goals,
    )
    candidates = candidate_scores(max_goals)
    if margin >= close_margin and favourite != "draw":
        candidates = [(home, away) for home, away in candidates if outcome_label(home, away) == favourite]
    elif margin >= close_margin and favourite == "draw":
        candidates = [(home, away) for home, away in candidates if home == away]

    best: dict[str, Any] | None = None
    all_scores: list[dict[str, Any]] = []
    for pred_home, pred_away in candidates:
        expected_points = 0.0
        for actual_home in range(matrix.shape[0]):
            for actual_away in range(matrix.shape[1]):
                expected_points += float(matrix[actual_home, actual_away]) * score_points(
                    pred_home,
                    pred_away,
                    actual_home,
                    actual_away,
                    exact_points,
                    outcome_points,
                    goal_diff_points,
                    team_goal_points,
                )
        exact_probability = float(matrix[pred_home, pred_away])
        score = {
            "score": f"{pred_home}-{pred_away}",
            "home_score": pred_home,
            "away_score": pred_away,
            "outcome": outcome_label(pred_home, pred_away),
            "expected_pool_points": expected_points,
            "exact_score_probability": exact_probability,
        }
        all_scores.append(score)
        if best is None or (expected_points, exact_probability) > (
            best["expected_pool_points"],
            best["exact_score_probability"],
        ):
            best = score

    assert best is not None
    all_scores.sort(key=lambda item: (item["expected_pool_points"], item["exact_score_probability"]), reverse=True)
    best_exact_flat = int(matrix.argmax())
    exact_home, exact_away = np.unravel_index(best_exact_flat, matrix.shape)
    second = all_scores[1] if len(all_scores) > 1 else best
    return {
        **best,
        "model_favourite_outcome": favourite,
        "model_favourite_prob": favourite_prob,
        "favourite_margin": margin,
        "best_exact_score": f"{int(exact_home)}-{int(exact_away)}",
        "best_exact_probability": float(matrix[exact_home, exact_away]),
        "second_choice_score": second["score"],
        "second_choice_expected_pool_points": second["expected_pool_points"],
        "score_choice_rule": "restricted_to_favourite" if margin >= close_margin else "open_close_match",
    }


def oriented_score(home_goals: int, away_goals: int, outcome: str) -> str:
    if outcome == "home_win":
        return f"{home_goals}-{away_goals}"
    if outcome == "away_win":
        return f"{away_goals}-{home_goals}"
    return f"{home_goals}-{home_goals}"


def score_parts(score: str) -> tuple[int, int]:
    home, away = str(score).split("-", 1)
    return int(home), int(away)


def upside_score(row: pd.Series) -> tuple[str, str]:
    params = WORLD_CUP_2022_TUNED_SCORE_PARAMS
    favourite = str(row["model_favourite_outcome"])
    margin = float(row["favourite_margin"])
    fav_prob = float(row["model_favourite_prob"])
    home_xg = float(row["expected_home_goals"])
    away_xg = float(row["expected_away_goals"])
    total_xg = home_xg + away_xg

    if margin <= params["draw_margin"] and float(row["prob_draw"]) >= params["draw_min_prob"]:
        if total_xg >= params["draw_22_total"]:
            return "2-2", "wk2022_tuned_close_draw_2_2"
        return ("0-0" if total_xg < params["draw_00_total"] else "1-1"), "wk2022_tuned_close_draw"

    if favourite == "draw":
        return ("0-0" if total_xg < params["draw_00_total"] else "1-1"), "wk2022_tuned_model_draw"

    fav_xg = home_xg if favourite == "home_win" else away_xg
    dog_xg = away_xg if favourite == "home_win" else home_xg

    if fav_prob >= params["strong_prob"]:
        if fav_xg >= params["strong_30_fav_xg"] and dog_xg <= params["strong_30_dog_xg"]:
            return oriented_score(3, 0, favourite), "wk2022_tuned_strong_3_0"
        if dog_xg >= params["strong_21_dog_xg"] or total_xg >= params["strong_21_total"]:
            return oriented_score(2, 1, favourite), "wk2022_tuned_strong_2_1"
        return oriented_score(2, 0, favourite), "wk2022_tuned_strong_2_0"

    if fav_prob >= params["medium_prob"]:
        if total_xg >= params["medium_21_total"] and dog_xg >= params["medium_21_dog_xg"]:
            return oriented_score(2, 1, favourite), "wk2022_tuned_medium_2_1"
        if fav_xg >= params["medium_20_fav_xg"] and dog_xg <= params["medium_20_dog_xg"]:
            return oriented_score(2, 0, favourite), "wk2022_tuned_medium_2_0"
        return oriented_score(1, 0, favourite), "wk2022_tuned_medium_1_0"

    if total_xg >= params["small_21_total"] and dog_xg >= params["small_21_dog_xg"]:
        return oriented_score(2, 1, favourite), "wk2022_tuned_small_2_1"
    if fav_xg >= params["small_20_fav_xg"] and dog_xg <= params["small_20_dog_xg"]:
        return oriented_score(2, 0, favourite), "wk2022_tuned_small_2_0"
    return oriented_score(1, 0, favourite), "wk2022_tuned_small_1_0"


def apply_recommended_scores(output: pd.DataFrame) -> pd.DataFrame:
    output = output.copy()
    output = output.rename(
        columns={
            "score": "safe_score",
            "home_score": "safe_home_score",
            "away_score": "safe_away_score",
            "outcome": "safe_outcome",
        }
    )
    upside = output.apply(upside_score, axis=1, result_type="expand")
    output["upside_score"] = upside[0]
    output["upside_rule"] = upside[1]
    output["recommended_score"] = output["safe_score"]
    output["recommended_rule"] = "safe_expected_points"

    for idx, row in output.iterrows():
        up_home, up_away = score_parts(row["upside_score"])
        safe_home, safe_away = int(row["safe_home_score"]), int(row["safe_away_score"])
        total_xg = float(row["expected_home_goals"]) + float(row["expected_away_goals"])
        use_upside = (
            row["upside_score"] != row["safe_score"]
            and (
                float(row["model_favourite_prob"]) < ROBUST_HYBRID_SCORE_PARAMS["fav_prob_tuned_below"]
                or (
                    float(row["favourite_margin"]) < ROBUST_HYBRID_SCORE_PARAMS["close_draw_margin"]
                    and float(row["prob_draw"]) >= ROBUST_HYBRID_SCORE_PARAMS["close_draw_min_prob"]
                )
                or total_xg >= ROBUST_HYBRID_SCORE_PARAMS["total_xg_tuned_at_least"]
            )
        )
        if use_upside:
            output.at[idx, "recommended_score"] = row["upside_score"]
            output.at[idx, "recommended_rule"] = row["upside_rule"]

        probs = {
            "home_win": float(row["prob_home_win"]),
            "draw": float(row["prob_draw"]),
            "away_win": float(row["prob_away_win"]),
        }
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        close_second_winner = (
            ranked[0][1] - ranked[1][1] <= ROBUST_HYBRID_SCORE_PARAMS["close_second_winner_margin"]
            and ROBUST_HYBRID_SCORE_PARAMS["close_second_winner_draw_min"]
            <= probs["draw"]
            < ROBUST_HYBRID_SCORE_PARAMS["close_second_winner_draw_max"]
            and ranked[1][0] in {"home_win", "away_win"}
        )
        if close_second_winner:
            if total_xg >= ROBUST_HYBRID_SCORE_PARAMS["close_second_winner_open_total_xg"]:
                output.at[idx, "recommended_score"] = oriented_score(2, 1, ranked[1][0])
            else:
                output.at[idx, "recommended_score"] = oriented_score(1, 0, ranked[1][0])
            output.at[idx, "recommended_rule"] = "close_second_winner_data_boost"

    rec_parts = output["recommended_score"].map(score_parts)
    output["home_score"] = [part[0] for part in rec_parts]
    output["away_score"] = [part[1] for part in rec_parts]
    output["score"] = output["recommended_score"]
    output["outcome"] = [outcome_label(h, a) for h, a in zip(output["home_score"], output["away_score"])]
    output["predicted_winner"] = np.select(
        [output["outcome"].eq("home_win"), output["outcome"].eq("away_win")],
        [output["home_team"], output["away_team"]],
        default="Draw",
    )
    return output


def load_group_rows(path: Path) -> pd.DataFrame:
    group = pd.read_csv(path)
    group["prob_home_win"] = group["sim_prob_home_win"]
    group["prob_draw"] = group["sim_prob_draw"]
    group["prob_away_win"] = group["sim_prob_away_win"]
    group["stage"] = "Group Stage"
    group["source"] = "model_plus_odds"
    return group


def load_knockout_rows(bracket_path: Path, schedule_path: Path) -> pd.DataFrame:
    bracket = pd.read_csv(bracket_path)
    schedule = pd.read_csv(schedule_path)
    schedule_cols = ["match_number", "date", "stage", "group", "city", "country", "venue"]
    schedule_cols = [col for col in schedule_cols if col in schedule.columns]
    bracket = bracket.merge(schedule[schedule_cols], on="match_number", how="left", suffixes=("", "_schedule"))
    if "stage_schedule" in bracket.columns:
        bracket["stage"] = bracket["stage_schedule"].fillna(bracket["stage"])

    p_winner = pd.to_numeric(bracket["winner_win_prob"], errors="coerce").fillna(0.55).clip(0.50, 0.95)
    home_is_winner = bracket["predicted_winner"].eq(bracket["home_team"])
    bracket["prob_home_win"] = np.where(home_is_winner, p_winner, 1.0 - p_winner)
    bracket["prob_away_win"] = np.where(home_is_winner, 1.0 - p_winner, p_winner)
    bracket["prob_draw"] = 0.0

    winner_xg = np.select(
        [p_winner >= 0.82, p_winner >= 0.68],
        [2.05, 1.70],
        default=1.45,
    )
    loser_xg = np.select(
        [p_winner >= 0.82, p_winner >= 0.68],
        [0.55, 0.75],
        default=1.02,
    )
    late_stage = bracket["stage"].isin(["Quarterfinals", "Semifinals", "Third Place Playoff", "Final"])
    winner_xg = np.where(late_stage, winner_xg * 0.92, winner_xg)
    loser_xg = np.where(late_stage, loser_xg * 0.92, loser_xg)
    bracket["expected_home_goals"] = np.where(home_is_winner, winner_xg, loser_xg)
    bracket["expected_away_goals"] = np.where(home_is_winner, loser_xg, winner_xg)
    bracket["group"] = ""
    bracket["source"] = "predicted_bracket"
    return bracket


def lock_actual_group_results(pool: pd.DataFrame) -> pd.DataFrame:
    if "actual_available" not in pool.columns:
        return pool
    pool = pool.copy()
    actual_mask = pool["actual_available"].fillna(False).astype(bool)
    if not actual_mask.any():
        return pool

    for idx, row in pool[actual_mask].iterrows():
        home_score = int(row["actual_home_score"])
        away_score = int(row["actual_away_score"])
        outcome = outcome_label(home_score, away_score)
        pool.at[idx, "home_score"] = home_score
        pool.at[idx, "away_score"] = away_score
        pool.at[idx, "score"] = f"{home_score}-{away_score}"
        pool.at[idx, "outcome"] = outcome
        pool.at[idx, "predicted_winner"] = (
            row["home_team"] if outcome == "home_win" else row["away_team"] if outcome == "away_win" else "Draw"
        )
        pool.at[idx, "recommended_rule"] = "actual_result_locked"
        pool.at[idx, "confidence"] = "actual"
    return pool


def build_pool_predictions(args: argparse.Namespace) -> pd.DataFrame:
    group = load_group_rows(args.group_predictions)
    knockout = load_knockout_rows(args.bracket, args.schedule)
    base_cols = [
        "match_number",
        "date",
        "stage",
        "group",
        "home_team",
        "away_team",
        "city",
        "country",
        "venue",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "expected_home_goals",
        "expected_away_goals",
        "source",
        "actual_available",
        "actual_home_score",
        "actual_away_score",
        "actual_score",
        "actual_outcome",
        "actual_winner",
    ]
    for col in base_cols:
        if col not in group.columns:
            group[col] = np.nan
        if col not in knockout.columns:
            knockout[col] = np.nan
    combined = pd.concat([group[base_cols], knockout[base_cols]], ignore_index=True)
    combined = combined.sort_values("match_number").reset_index(drop=True)

    picks = [
        optimize_score_for_pool(
            row,
            args.max_goals,
            args.close_margin,
            args.exact_points,
            args.outcome_points,
            args.goal_diff_points,
            args.team_goal_points,
        )
        for _, row in combined.iterrows()
    ]
    pick_df = pd.DataFrame(picks)
    output = pd.concat([combined, pick_df], axis=1)
    output = apply_recommended_scores(output)
    output["confidence"] = np.select(
        [output["model_favourite_prob"] >= 0.70, output["model_favourite_prob"] >= 0.55],
        ["high", "medium"],
        default="low",
    )
    output = lock_actual_group_results(output)
    return output


def group_standings_from_picks(pool: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_matches = pool[pool["stage"].eq("Group Stage")].copy()
    for group, matches in group_matches.groupby("group"):
        table: dict[str, dict[str, Any]] = {}
        teams = sorted(set(matches["home_team"]).union(matches["away_team"]))
        for team in teams:
            table[team] = {
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
            }
        for match in matches.itertuples(index=False):
            home = match.home_team
            away = match.away_team
            hg = int(match.home_score)
            ag = int(match.away_score)
            table[home]["played"] += 1
            table[away]["played"] += 1
            table[home]["gf"] += hg
            table[home]["ga"] += ag
            table[away]["gf"] += ag
            table[away]["ga"] += hg
            if hg > ag:
                table[home]["wins"] += 1
                table[away]["losses"] += 1
                table[home]["points"] += 3
            elif hg < ag:
                table[away]["wins"] += 1
                table[home]["losses"] += 1
                table[away]["points"] += 3
            else:
                table[home]["draws"] += 1
                table[away]["draws"] += 1
                table[home]["points"] += 1
                table[away]["points"] += 1
        for team_data in table.values():
            team_data["gd"] = team_data["gf"] - team_data["ga"]
        ranked = sorted(table.values(), key=lambda r: (r["points"], r["gd"], r["gf"], r["wins"]), reverse=True)
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
            row["qualified_by_pick"] = rank <= 2
            rows.append(row)
    return pd.DataFrame(rows)


def expected_matches_and_goals(team_probs: pd.DataFrame) -> pd.DataFrame:
    team_probs = team_probs.copy()
    for col in [
        "advance_r32_prob",
        "advance_r16_prob",
        "advance_qf_prob",
        "advance_sf_prob",
        "expected_group_gf",
    ]:
        team_probs[col] = pd.to_numeric(team_probs[col], errors="coerce").fillna(0.0)
    team_probs["expected_matches"] = (
        3.0
        + team_probs["advance_r32_prob"]
        + team_probs["advance_r16_prob"]
        + team_probs["advance_qf_prob"]
        + 2.0 * team_probs["advance_sf_prob"]
    )
    group_gf_per_match = team_probs["expected_group_gf"] / 3.0
    knockout_matches = np.maximum(team_probs["expected_matches"] - 3.0, 0.0)
    team_probs["expected_tournament_goals"] = team_probs["expected_group_gf"] + 0.82 * group_gf_per_match * knockout_matches
    return team_probs


def parse_date(value: Any) -> pd.Timestamp | pd.NaT:
    parsed = pd.to_datetime(value, errors="coerce")
    if isinstance(parsed, pd.Timestamp):
        return parsed
    return pd.NaT


def latest_sofifa_by_player(path: Path, cutoff: str = "2026-06-10") -> pd.DataFrame:
    sofifa = pd.read_csv(path)
    sofifa["available_from"] = pd.to_datetime(sofifa["available_from"], errors="coerce")
    sofifa["dob"] = pd.to_datetime(sofifa["dob"], errors="coerce").dt.strftime("%Y-%m-%d")
    sofifa = sofifa[sofifa["available_from"].le(pd.Timestamp(cutoff))]
    sofifa["name_key_short"] = sofifa["short_name"].map(normalize_person_name)
    sofifa["name_key_long"] = sofifa["long_name"].map(normalize_person_name)
    sofifa["nationality_key"] = sofifa["nationality_name"].map(normalize_name)
    sort_cols = ["available_from", "overall", "potential"]
    sofifa = sofifa.sort_values(sort_cols)
    if "sofifa_id" in sofifa.columns:
        sofifa = sofifa.drop_duplicates("sofifa_id", keep="last")
    return sofifa


def build_sofifa_indexes(candidates: pd.DataFrame) -> dict[str, Any]:
    records = candidates.to_dict("records")
    name_index: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    dob_index: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    dob_name_index: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    nat_index: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        keys = {record.get("name_key_short", ""), record.get("name_key_long", "")}
        keys = {str(key) for key in keys if str(key)}
        dob = str(record.get("dob", "") or "")
        nat = str(record.get("nationality_key", "") or "")
        for key in keys:
            name_index[key].append(record)
            if dob:
                dob_name_index[(dob, key)].append(record)
        if dob:
            dob_index[dob].append(record)
        if nat:
            nat_index[nat].append(record)

    for nat, nat_records in list(nat_index.items()):
        nat_index[nat] = sorted(nat_records, key=lambda r: numeric(r.get("overall")), reverse=True)[:500]
    return {
        "name": name_index,
        "dob": dob_index,
        "dob_name": dob_name_index,
        "nat": nat_index,
    }


def best_sofifa_record(records: list[dict[str, Any]], team_key: str = "") -> dict[str, Any] | None:
    if not records:
        return None
    if team_key:
        same_nat = [record for record in records if record.get("nationality_key") == team_key]
        if same_nat:
            records = same_nat
    return max(
        records,
        key=lambda record: (
            numeric(record.get("overall")),
            numeric(record.get("potential")),
            str(record.get("available_from", "")),
        ),
    )


def match_sofifa_player(player: pd.Series, indexes: dict[str, Any]) -> dict[str, Any] | None:
    player_key = normalize_person_name(player["player"])
    dob = parse_date(player.get("date_of_birth"))
    dob_key = dob.strftime("%Y-%m-%d") if not pd.isna(dob) else ""
    team_key = normalize_name(player.get("team"))

    if dob_key:
        exact_dob = indexes["dob_name"].get((dob_key, player_key), [])
        if exact_dob:
            return best_sofifa_record(exact_dob, team_key)

    exact_name = indexes["name"].get(player_key, [])
    if exact_name:
        return best_sofifa_record(exact_name, team_key)

    if dob_key:
        dob_candidates = indexes["dob"].get(dob_key, [])
        if dob_candidates:
            scored = []
            for record in dob_candidates:
                score = max(
                    person_name_match_score(player_key, str(record.get("name_key_long", ""))),
                    person_name_match_score(player_key, str(record.get("name_key_short", ""))),
                )
                if record.get("nationality_key") == team_key:
                    score += 0.08
                scored.append((score, record))
            score, best = max(scored, key=lambda item: (item[0], numeric(item[1].get("overall"))))
            if score >= 0.55:
                return best

    # Limited fallback: only consider same nationality to avoid wild matches.
    same_nat = indexes["nat"].get(team_key, [])
    if same_nat:
        scored = []
        for record in same_nat:
            score = max(
                person_name_match_score(player_key, str(record.get("name_key_long", ""))),
                person_name_match_score(player_key, str(record.get("name_key_short", ""))),
            )
            scored.append((score, record))
        score, best = max(scored, key=lambda item: (item[0], numeric(item[1].get("overall"))))
        if score >= 0.74:
            return best
    return None


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def position_weight(position: Any, player_positions: Any) -> float:
    squad_pos = str(position or "").upper()
    fifa_pos = str(player_positions or "").upper()
    if squad_pos == "GK" or "GK" in fifa_pos.split(","):
        return 0.0
    if squad_pos == "FW":
        return 1.0
    if any(pos in fifa_pos for pos in ["ST", "CF", "LW", "RW"]):
        return 0.95
    if squad_pos == "MF":
        return 0.38
    if any(pos in fifa_pos for pos in ["CAM", "LM", "RM", "CM"]):
        return 0.42
    if squad_pos == "DF":
        return 0.10
    return 0.25


def scorito_points_per_goal(position: Any) -> int:
    squad_pos = str(position or "").upper()
    if squad_pos == "DF":
        return 32
    if squad_pos == "MF":
        return 16
    if squad_pos == "FW":
        return 8
    return 0


def build_topscorer_ranking(args: argparse.Namespace, team_probs: pd.DataFrame) -> pd.DataFrame:
    if not args.squads.exists():
        return pd.DataFrame(columns=TOPSCORER_COLUMNS)
    squads = pd.read_csv(args.squads)
    squads = squads[squads["team"].notna() & squads["player"].notna()].copy()
    if args.sofifa.exists():
        sofifa = latest_sofifa_by_player(args.sofifa)
        sofifa_indexes = build_sofifa_indexes(sofifa)
    else:
        print(f"Skipping SoFIFA topscorer enrichment: {args.sofifa} does not exist.")
        sofifa_indexes = build_sofifa_indexes(pd.DataFrame())
    team_probs = expected_matches_and_goals(team_probs)
    team_goal_map = dict(zip(team_probs["team"], team_probs["expected_tournament_goals"]))
    team_group_goal_map = dict(zip(team_probs["team"], team_probs["expected_group_gf"]))
    team_match_map = dict(zip(team_probs["team"], team_probs["expected_matches"]))
    champ_map = dict(zip(team_probs["team"], team_probs["champion_prob"]))

    rows: list[dict[str, Any]] = []
    for _, player in squads.iterrows():
        player = player.copy()
        player["player"] = clean_player_name(player["player"])
        team = str(player["team"])
        matched = match_sofifa_player(player, sofifa_indexes)
        overall = numeric(matched.get("overall") if matched is not None else np.nan, 65.0)
        shooting = numeric(matched.get("shooting") if matched is not None else np.nan, overall)
        finishing = numeric(matched.get("attacking_finishing") if matched is not None else np.nan, shooting)
        positioning = numeric(matched.get("mentality_positioning") if matched is not None else np.nan, shooting)
        reactions = numeric(matched.get("movement_reactions") if matched is not None else np.nan, overall)
        pos_w = position_weight(player.get("position"), matched.get("player_positions") if matched is not None else "")
        caps = numeric(player.get("caps"), 0.0)
        goals = numeric(player.get("goals"), 0.0)
        goal_rate = goals / max(caps, 8.0)
        intl_signal = 0.65 * min(goal_rate / 0.45, 1.25) + 0.35 * min(goals / 80.0, 1.25)
        attack_quality = (
            0.34 * (finishing / 100.0)
            + 0.24 * (shooting / 100.0)
            + 0.18 * (positioning / 100.0)
            + 0.14 * (overall / 100.0)
            + 0.10 * (reactions / 100.0)
        )
        scorer_weight = pos_w * attack_quality * (0.78 + 0.78 * intl_signal)
        if goals >= 20:
            scorer_weight *= 1.10
        if caps >= 45 and pos_w >= 0.35:
            scorer_weight *= 1.05
        rows.append(
            {
                "team": team,
                "player": player["player"],
                "position": player.get("position"),
                "caps": caps,
                "international_goals": goals,
                "sofifa_matched": matched is not None,
                "sofifa_overall": overall if matched is not None else np.nan,
                "sofifa_shooting": shooting if matched is not None else np.nan,
                "sofifa_finishing": finishing if matched is not None else np.nan,
                "team_expected_matches": team_match_map.get(team, 3.0),
                "team_expected_group_goals": team_group_goal_map.get(team, 2.5),
                "team_expected_goals": team_goal_map.get(team, 2.5),
                "team_champion_prob": champ_map.get(team, 0.0),
                "raw_scorer_weight": scorer_weight,
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return pd.DataFrame(columns=TOPSCORER_COLUMNS)
    ranking["team_weight_sum"] = ranking.groupby("team")["raw_scorer_weight"].transform("sum")
    ranking["goal_share"] = np.where(
        ranking["team_weight_sum"].gt(0),
        ranking["raw_scorer_weight"] / ranking["team_weight_sum"],
        0.0,
    )
    ranking["expected_group_stage_goals"] = ranking["team_expected_group_goals"] * ranking["goal_share"]
    ranking["expected_goals"] = ranking["team_expected_goals"] * ranking["goal_share"]
    lam = ranking["expected_goals"].clip(lower=0.0)
    ranking["prob_4plus_goals"] = 1.0 - np.exp(-lam) * (1.0 + lam + lam**2 / 2.0 + lam**3 / 6.0)
    ranking["topscorer_score"] = ranking["expected_goals"] + 0.85 * ranking["prob_4plus_goals"]
    max_raw = max(float(ranking["raw_scorer_weight"].max()), 0.01)
    star_power = (
        0.48 * (ranking["raw_scorer_weight"] / max_raw).clip(0.0, 1.15)
        + 0.28 * (ranking["sofifa_overall"].fillna(72.0) / 91.0).clip(0.0, 1.08)
        + 0.24 * (ranking["international_goals"] / 70.0).clip(0.0, 1.15)
    )
    ranking["star_scorer_power"] = star_power
    ranking["recommended_topscorer_score"] = ranking["topscorer_score"] * (0.72 + 0.28 * star_power)
    ranking["golden_boot_rank"] = (
        ranking["recommended_topscorer_score"].rank(method="first", ascending=False).astype(int)
    )
    ranking["scorito_points_per_goal"] = ranking["position"].map(scorito_points_per_goal)
    ranking["expected_group_stage_scorito_points"] = (
        ranking["expected_group_stage_goals"] * ranking["scorito_points_per_goal"]
    )
    ranking["expected_scorito_points"] = ranking["expected_goals"] * ranking["scorito_points_per_goal"]
    ranking["recommended_group_stage_topscorer_score"] = (
        ranking["expected_group_stage_scorito_points"] * (0.72 + 0.28 * star_power)
    )
    ranking["recommended_scorito_topscorer_score"] = ranking["expected_scorito_points"] * (0.72 + 0.28 * star_power)
    ranking["group_stage_rank"] = (
        ranking["recommended_group_stage_topscorer_score"].rank(method="first", ascending=False).astype(int)
    )
    ranking = ranking.sort_values("recommended_scorito_topscorer_score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def champion_picks(team_probs: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "team",
        "champion_prob",
        "advance_final_prob",
        "advance_sf_prob",
        "advance_qf_prob",
        "advance_r16_prob",
        "expected_group_points",
        "expected_group_gd",
        "expected_group_gf",
        "expected_group_rank",
    ]
    available = [col for col in cols if col in team_probs.columns]
    output = team_probs[available].sort_values("champion_prob", ascending=False).reset_index(drop=True)
    output.insert(0, "rank", np.arange(1, len(output) + 1))
    output["recommendation"] = ""
    if not output.empty:
        output.loc[0, "recommendation"] = "kampioen pick"
    if len(output) > 1:
        output.loc[1, "recommendation"] = "finale outsider"
    if len(output) > 2:
        output.loc[2, "recommendation"] = "sterke hedge"
    return output


def spain_france_check(team_probs: pd.DataFrame) -> pd.DataFrame:
    subset = team_probs[team_probs["team"].isin(["Spain", "France"])].copy()
    subset = subset.sort_values("champion_prob", ascending=False)
    notes = []
    for row in subset.itertuples(index=False):
        if row.team == "Spain":
            notes.append("Model ziet Spanje als beste combinatie van teamsterkte, groepspad en knockoutpad.")
        else:
            notes.append("Frankrijk is extreem sterk, maar heeft in deze bracket vaker een zwaarder pad.")
    subset["interpretation"] = notes
    return subset


def write_summary(
    output_dir: Path,
    pool: pd.DataFrame,
    champion: pd.DataFrame,
    topscorers: pd.DataFrame,
    scoring_config: dict[str, float],
) -> None:
    draws = int(pool["outcome"].eq("draw").sum())
    group_draws = int(pool[pool["stage"].eq("Group Stage")]["outcome"].eq("draw").sum())
    top_champion = champion.iloc[0].to_dict() if not champion.empty else {}
    top_scorer = topscorers.iloc[0].to_dict() if not topscorers.empty else {}
    summary = {
        "scoring_config": scoring_config,
        "score_strategy": "robust_2018_2022_hybrid_plus_close_second_winner",
        "score_strategy_backtest_2022": {
            "baseline_exact_accuracy": 0.09375,
            "tuned_exact_accuracy": 0.140625,
            "baseline_avg_pool_points": 1.7421875,
            "tuned_avg_pool_points": 1.7890625,
            "baseline_one_nil_picks": 54,
            "tuned_one_nil_picks": 9,
            "baseline_outcome_accuracy": 0.5625,
            "tuned_outcome_accuracy": 0.53125,
        },
        "score_strategy_backtest_2018_2022": {
            "safe_total_points": 234.5,
            "safe_exact_accuracy": 0.1328125,
            "safe_outcome_accuracy": 0.546875,
            "safe_one_nil_picks": 110,
            "full_tuned_total_points": 229.5,
            "full_tuned_exact_accuracy": 0.140625,
            "full_tuned_outcome_accuracy": 0.515625,
            "full_tuned_one_nil_picks": 11,
            "hybrid_total_points": 234.5,
            "hybrid_exact_accuracy": 0.1484375,
            "hybrid_outcome_accuracy": 0.5234375,
            "hybrid_one_nil_picks": 19,
            "hybrid_with_close_second_winner_total_points": 237.0,
            "hybrid_with_close_second_winner_exact_accuracy": 0.1484375,
            "hybrid_with_close_second_winner_outcome_accuracy": 0.53125,
            "hybrid_with_close_second_winner_one_nil_picks": 19,
        },
        "matches": int(len(pool)),
        "draw_picks_total": draws,
        "draw_picks_group_stage": group_draws,
        "recommended_champion": top_champion.get("team"),
        "recommended_champion_probability": top_champion.get("champion_prob"),
        "recommended_topscorer": top_scorer.get("player"),
        "recommended_topscorer_team": top_scorer.get("team"),
        "recommended_topscorer_expected_goals": top_scorer.get("expected_goals"),
        "note": (
            "Safe scores maximize expected pool points under the configured point weights. "
            "Final score picks use a robust 2018+2022 hybrid layer plus a small close-second-winner "
            "mechanism to reduce excessive 1-0/0-1 and improve combined historical pool points. "
            "Scores remain consistent with the selected winner/draw."
        ),
    }
    (output_dir / "scorito_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def clean_entry_sheet(pool: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
        "actual_available",
        "actual_home_score",
        "actual_away_score",
        "actual_score",
        "actual_outcome",
        "actual_winner",
    ]
    return pool[[col for col in columns if col in pool.columns]].copy()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pool = build_pool_predictions(args)
    group_tables = group_standings_from_picks(pool)
    team_probs = pd.read_csv(args.team_probabilities)
    champion = champion_picks(team_probs)
    spain_france = spain_france_check(team_probs)
    topscorers_all = build_topscorer_ranking(args, team_probs)
    topscorers = topscorers_all.head(args.top_n)
    groupstage_topscorers = (
        topscorers_all.sort_values("recommended_group_stage_topscorer_score", ascending=False)
        .reset_index(drop=True)
        .head(args.top_n)
        .copy()
    )
    if not groupstage_topscorers.empty:
        groupstage_topscorers.insert(0, "groupstage_rank", np.arange(1, len(groupstage_topscorers) + 1))

    pool_out = args.output_dir / "scorito_pool_predictions.csv"
    entry_out = args.output_dir / "scorito_invuladvies.csv"
    group_out = args.output_dir / "scorito_group_standings.csv"
    champion_out = args.output_dir / "scorito_champion_picks.csv"
    spain_france_out = args.output_dir / "scorito_spain_france_check.csv"
    topscorer_out = args.output_dir / "scorito_topscorer_picks.csv"
    groupstage_topscorer_out = args.output_dir / "scorito_groupstage_topscorer_picks.csv"

    pool.to_csv(pool_out, index=False)
    clean_entry_sheet(pool).to_csv(entry_out, index=False)
    group_tables.to_csv(group_out, index=False)
    champion.to_csv(champion_out, index=False)
    spain_france.to_csv(spain_france_out, index=False)
    topscorers.to_csv(topscorer_out, index=False)
    groupstage_topscorers.to_csv(groupstage_topscorer_out, index=False)
    write_summary(
        args.output_dir,
        pool,
        champion,
        groupstage_topscorers if not groupstage_topscorers.empty else topscorers,
        {
            "exact_points": args.exact_points,
            "outcome_points": args.outcome_points,
            "goal_diff_points": args.goal_diff_points,
            "team_goal_points": args.team_goal_points,
            "close_margin": args.close_margin,
        },
    )

    print(f"Wrote {pool_out}")
    print(f"Wrote {entry_out}")
    print(f"Wrote {group_out}")
    print(f"Wrote {champion_out}")
    print(f"Wrote {topscorer_out}")
    print(f"Wrote {groupstage_topscorer_out}")
    print("\nChampion picks:")
    print(champion.head(8).to_string(index=False))
    if not topscorers.empty:
        print("\nTop scorer picks (whole tournament, Scorito weighted):")
        print(
            topscorers.head(8)[
                [
                    "rank",
                    "player",
                    "team",
                    "position",
                    "expected_goals",
                    "expected_scorito_points",
                    "golden_boot_rank",
                ]
            ].to_string(index=False)
        )
    if not groupstage_topscorers.empty:
        print("\nTop scorer picks (group stage, Scorito weighted):")
        print(
            groupstage_topscorers.head(8)[
                [
                    "groupstage_rank",
                    "player",
                    "team",
                    "position",
                    "expected_group_stage_goals",
                    "expected_group_stage_scorito_points",
                    "golden_boot_rank",
                ]
            ].to_string(index=False)
        )
    print("\nScore pick outcome counts:")
    print(pool["outcome"].value_counts().to_string())


if __name__ == "__main__":
    main()
