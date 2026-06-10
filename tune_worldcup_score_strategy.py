#!/usr/bin/env python
"""Tune the manual World Cup score-pick layer on historical World Cup rows."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BACKTEST = Path("outputs_backtest_2022/test_predictions_xgboost.csv")
DEFAULT_OUTPUT_DIR = Path("outputs_backtest_2022")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune score-pick heuristics on historical World Cup predictions.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_BACKTEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--exact-points", type=float, default=5.0)
    parser.add_argument("--outcome-points", type=float, default=2.0)
    parser.add_argument("--goal-diff-points", type=float, default=1.0)
    parser.add_argument("--team-goal-points", type=float, default=0.5)
    return parser.parse_args()


def outcome_label(home: int, away: int) -> str:
    if home > away:
        return "home_win"
    if home < away:
        return "away_win"
    return "draw"


def oriented_score(fav_goals: int, dog_goals: int, outcome: str) -> tuple[int, int]:
    if outcome == "home_win":
        return fav_goals, dog_goals
    if outcome == "away_win":
        return dog_goals, fav_goals
    return fav_goals, fav_goals


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
    if pred_home - pred_away == actual_home - actual_away:
        points += goal_diff_points
    if pred_home == actual_home:
        points += team_goal_points
    if pred_away == actual_away:
        points += team_goal_points
    return points


def pick_score(row: pd.Series, params: dict[str, Any]) -> tuple[int, int, str]:
    probs = {
        "home_win": float(row["prob_home_win"]),
        "draw": float(row["prob_draw"]),
        "away_win": float(row["prob_away_win"]),
    }
    ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    fav, fav_prob = ordered[0]
    margin = fav_prob - ordered[1][1]
    home_xg = float(row["expected_home_goals"])
    away_xg = float(row["expected_away_goals"])
    total_xg = home_xg + away_xg

    if margin <= params["draw_margin"] and probs["draw"] >= params["draw_min_prob"]:
        draw_goals = 0 if total_xg < params["draw_00_total"] else 1
        if total_xg >= params["draw_22_total"]:
            draw_goals = 2
        return draw_goals, draw_goals, "close_draw"

    if fav == "draw":
        draw_goals = 0 if total_xg < params["draw_00_total"] else 1
        return draw_goals, draw_goals, "model_draw"

    fav_xg = home_xg if fav == "home_win" else away_xg
    dog_xg = away_xg if fav == "home_win" else home_xg

    if fav_prob >= params["strong_prob"]:
        if fav_xg >= params["strong_30_fav_xg"] and dog_xg <= params["strong_30_dog_xg"]:
            home, away = oriented_score(3, 0, fav)
            return home, away, "strong_3_0"
        if dog_xg >= params["strong_21_dog_xg"] or total_xg >= params["strong_21_total"]:
            home, away = oriented_score(2, 1, fav)
            return home, away, "strong_2_1"
        home, away = oriented_score(2, 0, fav)
        return home, away, "strong_2_0"

    if fav_prob >= params["medium_prob"]:
        if total_xg >= params["medium_21_total"] and dog_xg >= params["medium_21_dog_xg"]:
            home, away = oriented_score(2, 1, fav)
            return home, away, "medium_2_1"
        if fav_xg >= params["medium_20_fav_xg"] and dog_xg <= params["medium_20_dog_xg"]:
            home, away = oriented_score(2, 0, fav)
            return home, away, "medium_2_0"
        home, away = oriented_score(1, 0, fav)
        return home, away, "medium_1_0"

    if total_xg >= params["small_21_total"] and dog_xg >= params["small_21_dog_xg"]:
        home, away = oriented_score(2, 1, fav)
        return home, away, "small_2_1"
    if fav_xg >= params["small_20_fav_xg"] and dog_xg <= params["small_20_dog_xg"]:
        home, away = oriented_score(2, 0, fav)
        return home, away, "small_2_0"
    home, away = oriented_score(1, 0, fav)
    return home, away, "small_1_0"


def score_strategy(frame: pd.DataFrame, params: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    points = []
    for _, row in frame.iterrows():
        pred_home, pred_away, rule = pick_score(row, params)
        actual_home = int(row["home_score"])
        actual_away = int(row["away_score"])
        rows.append(
            {
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "actual_score": f"{actual_home}-{actual_away}",
                "predicted_score": f"{pred_home}-{pred_away}",
                "predicted_home_score": pred_home,
                "predicted_away_score": pred_away,
                "predicted_outcome_from_score": outcome_label(pred_home, pred_away),
                "actual_outcome": row["actual_outcome"],
                "model_predicted_outcome": row["predicted_outcome"],
                "rule": rule,
                "prob_home_win": row["prob_home_win"],
                "prob_draw": row["prob_draw"],
                "prob_away_win": row["prob_away_win"],
                "expected_home_goals": row["expected_home_goals"],
                "expected_away_goals": row["expected_away_goals"],
            }
        )
        points.append(
            score_points(
                pred_home,
                pred_away,
                actual_home,
                actual_away,
                args.exact_points,
                args.outcome_points,
                args.goal_diff_points,
                args.team_goal_points,
            )
        )
    pred = pd.DataFrame(rows)
    pred["pool_points"] = points
    summary = {
        **params,
        "matches": int(len(pred)),
        "avg_pool_points": float(np.mean(points)),
        "total_pool_points": float(np.sum(points)),
        "exact_accuracy": float(pred["predicted_score"].eq(pred["actual_score"]).mean()),
        "outcome_accuracy": float(pred["predicted_outcome_from_score"].eq(pred["actual_outcome"]).mean()),
        "draw_picks": int(pred["predicted_outcome_from_score"].eq("draw").sum()),
        "one_nil_picks": int(pred["predicted_score"].isin(["1-0", "0-1"]).sum()),
        "two_one_picks": int(pred["predicted_score"].isin(["2-1", "1-2"]).sum()),
        "two_nil_picks": int(pred["predicted_score"].isin(["2-0", "0-2"]).sum()),
        "three_nil_picks": int(pred["predicted_score"].isin(["3-0", "0-3"]).sum()),
    }
    return summary, pred


def base_summary(frame: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    points = []
    pred_scores = []
    for row in frame.itertuples(index=False):
        pred_home = int(row.pool_predicted_home_score)
        pred_away = int(row.pool_predicted_away_score)
        actual_home = int(row.home_score)
        actual_away = int(row.away_score)
        pred_scores.append(f"{pred_home}-{pred_away}")
        points.append(
            score_points(
                pred_home,
                pred_away,
                actual_home,
                actual_away,
                args.exact_points,
                args.outcome_points,
                args.goal_diff_points,
                args.team_goal_points,
            )
        )
    pred_outcomes = [outcome_label(*map(int, score.split("-"))) for score in pred_scores]
    return {
        "strategy": "current_model_pool_score",
        "matches": int(len(frame)),
        "avg_pool_points": float(np.mean(points)),
        "total_pool_points": float(np.sum(points)),
        "exact_accuracy": float(np.mean(np.array(pred_scores) == (frame["home_score"].astype(int).astype(str) + "-" + frame["away_score"].astype(int).astype(str)).to_numpy())),
        "outcome_accuracy": float(np.mean(np.array(pred_outcomes) == frame["actual_outcome"].to_numpy())),
        "draw_picks": int(np.sum(np.array(pred_outcomes) == "draw")),
        "one_nil_picks": int(np.sum(pd.Series(pred_scores).isin(["1-0", "0-1"]))),
        "two_one_picks": int(np.sum(pd.Series(pred_scores).isin(["2-1", "1-2"]))),
        "two_nil_picks": int(np.sum(pd.Series(pred_scores).isin(["2-0", "0-2"]))),
        "three_nil_picks": int(np.sum(pd.Series(pred_scores).isin(["3-0", "0-3"]))),
    }


def parameter_grid() -> list[dict[str, Any]]:
    grid = []
    for values in product(
        [0.12, 0.18, 0.22],  # draw_margin
        [0.26, 0.30],  # draw_min_prob
        [1.90],  # draw_00_total
        [3.20],  # draw_22_total
        [0.72, 0.80],  # strong_prob
        [2.15, 2.40],  # strong_30_fav_xg
        [0.60],  # strong_30_dog_xg
        [0.90],  # strong_21_dog_xg
        [2.80],  # strong_21_total
        [0.52, 0.58],  # medium_prob
        [2.10, 2.30],  # medium_21_total
        [0.65, 0.80],  # medium_21_dog_xg
        [1.70, 1.90],  # medium_20_fav_xg
        [0.70],  # medium_20_dog_xg
        [1.90, 2.10, 2.30],  # small_21_total
        [0.65, 0.80],  # small_21_dog_xg
        [1.60, 1.80],  # small_20_fav_xg
        [0.70],  # small_20_dog_xg
    ):
        (
            draw_margin,
            draw_min_prob,
            draw_00_total,
            draw_22_total,
            strong_prob,
            strong_30_fav_xg,
            strong_30_dog_xg,
            strong_21_dog_xg,
            strong_21_total,
            medium_prob,
            medium_21_total,
            medium_21_dog_xg,
            medium_20_fav_xg,
            medium_20_dog_xg,
            small_21_total,
            small_21_dog_xg,
            small_20_fav_xg,
            small_20_dog_xg,
        ) = values
        if medium_prob >= strong_prob:
            continue
        grid.append(
            {
                "draw_margin": draw_margin,
                "draw_min_prob": draw_min_prob,
                "draw_00_total": draw_00_total,
                "draw_22_total": draw_22_total,
                "strong_prob": strong_prob,
                "strong_30_fav_xg": strong_30_fav_xg,
                "strong_30_dog_xg": strong_30_dog_xg,
                "strong_21_dog_xg": strong_21_dog_xg,
                "strong_21_total": strong_21_total,
                "medium_prob": medium_prob,
                "medium_21_total": medium_21_total,
                "medium_21_dog_xg": medium_21_dog_xg,
                "medium_20_fav_xg": medium_20_fav_xg,
                "medium_20_dog_xg": medium_20_dog_xg,
                "small_21_total": small_21_total,
                "small_21_dog_xg": small_21_dog_xg,
                "small_20_fav_xg": small_20_fav_xg,
                "small_20_dog_xg": small_20_dog_xg,
            }
        )
    return grid


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.predictions)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    frame = data[
        data["tournament"].eq("FIFA World Cup")
        & data["date"].dt.year.eq(args.year)
        & data["home_score"].notna()
        & data["away_score"].notna()
    ].copy()
    if frame.empty:
        raise ValueError(f"No FIFA World Cup rows found for {args.year} in {args.predictions}")

    summaries = [base_summary(frame, args)]
    best_summary: dict[str, Any] | None = None
    best_predictions: pd.DataFrame | None = None
    for params in parameter_grid():
        summary, pred = score_strategy(frame, params, args)
        summaries.append(summary)
        key = (
            summary["avg_pool_points"],
            summary["exact_accuracy"],
            summary["outcome_accuracy"],
            -summary["one_nil_picks"],
        )
        if best_summary is None or key > (
            best_summary["avg_pool_points"],
            best_summary["exact_accuracy"],
            best_summary["outcome_accuracy"],
            -best_summary["one_nil_picks"],
        ):
            best_summary = summary
            best_predictions = pred

    assert best_summary is not None and best_predictions is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries).sort_values(
        ["avg_pool_points", "exact_accuracy", "outcome_accuracy"], ascending=False
    )
    summary_df.to_csv(args.output_dir / f"worldcup{args.year}_score_strategy_grid.csv", index=False)
    best_predictions.to_csv(args.output_dir / f"worldcup{args.year}_score_strategy_best_predictions.csv", index=False)
    (args.output_dir / f"worldcup{args.year}_score_strategy_best.json").write_text(
        json.dumps(best_summary, indent=2), encoding="utf-8"
    )
    print("Baseline:")
    print(pd.DataFrame([summaries[0]]).to_string(index=False))
    print("\nBest tuned strategy:")
    print(pd.DataFrame([best_summary]).to_string(index=False))
    print("\nBest score distribution:")
    print(best_predictions["predicted_score"].value_counts().head(20).to_string())
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
