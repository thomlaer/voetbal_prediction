#!/usr/bin/env python
"""Apply a transparent post-model tweak layer to saved football predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import train_xgboost_worldcup as tw


PROB_COLUMNS = ["prob_away_win", "prob_draw", "prob_home_win"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply draw/goal/manual tweaks to future predictions.")
    parser.add_argument("--predictions", type=Path, default=Path("outputs_worldcup2026_default/future_predictions_xgboost.csv"))
    parser.add_argument("--schedule", type=Path, default=Path("data/extracted/oddsportal_worldcup2026_fixture_odds_schedule.csv"))
    parser.add_argument("--manual-tweaks", type=Path, default=Path("data/extracted/manual_prediction_tweaks.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs_worldcup2026_default/future_predictions_xgboost_tweaked.csv"))
    parser.add_argument("--template-output", type=Path, default=Path("data/extracted/manual_prediction_tweaks_template.csv"))
    parser.add_argument("--draw-multiplier", type=float, default=1.10)
    parser.add_argument("--goal-multiplier", type=float, default=1.00)
    parser.add_argument("--score-max-goals", type=int, default=8)
    return parser.parse_args()


def normalize_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    probs = frame[PROB_COLUMNS].to_numpy(dtype=float)
    probs = probs / probs.sum(axis=1, keepdims=True)
    frame[PROB_COLUMNS] = probs
    return frame


def merge_schedule(predictions: pd.DataFrame, schedule_path: Path) -> pd.DataFrame:
    if not schedule_path.exists():
        return predictions
    schedule = pd.read_csv(schedule_path)
    cols = ["date", "home_team", "away_team", "match_number", "stage", "group", "city", "venue"]
    available = [col for col in cols if col in schedule.columns]
    merged = predictions.merge(schedule[available], on=["date", "home_team", "away_team"], how="left")
    return merged


def write_manual_template(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [col for col in ["match_number", "date", "home_team", "away_team", "stage", "group"] if col in frame.columns]
    template = frame[cols].copy()
    template["home_prob_mult"] = ""
    template["draw_prob_mult"] = ""
    template["away_prob_mult"] = ""
    template["home_goals_adjust"] = ""
    template["away_goals_adjust"] = ""
    template["forced_outcome"] = ""
    template["forced_score"] = ""
    template["note"] = ""
    if not path.exists():
        template.to_csv(path, index=False)


def apply_manual_tweaks(frame: pd.DataFrame, tweaks_path: Path) -> pd.DataFrame:
    if not tweaks_path.exists():
        return frame
    tweaks = pd.read_csv(tweaks_path)
    if tweaks.empty:
        return frame
    key_cols = ["date", "home_team", "away_team"]
    if "match_number" in frame.columns and "match_number" in tweaks.columns:
        key_cols = ["match_number"]
    tweak_cols = [
        "home_prob_mult",
        "draw_prob_mult",
        "away_prob_mult",
        "home_goals_adjust",
        "away_goals_adjust",
        "forced_outcome",
        "forced_score",
        "note",
    ]
    available = key_cols + [col for col in tweak_cols if col in tweaks.columns]
    return frame.merge(tweaks[available], on=key_cols, how="left")


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    predictions = merge_schedule(predictions, args.schedule)
    write_manual_template(predictions, args.template_output)
    predictions = apply_manual_tweaks(predictions, args.manual_tweaks)

    probs = predictions[PROB_COLUMNS].to_numpy(dtype=float)
    probs[:, tw.OUTCOME_TO_ID["draw"]] *= args.draw_multiplier

    mult_cols = {
        "prob_away_win": "away_prob_mult",
        "prob_draw": "draw_prob_mult",
        "prob_home_win": "home_prob_mult",
    }
    for index, prob_col in enumerate(PROB_COLUMNS):
        tweak_col = mult_cols[prob_col]
        if tweak_col in predictions.columns:
            values = pd.to_numeric(predictions[tweak_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
            probs[:, index] *= values
    probs = probs / probs.sum(axis=1, keepdims=True)

    output = predictions.copy()
    output["base_predicted_outcome"] = output["predicted_outcome"]
    output["base_prob_away_win"] = output["prob_away_win"]
    output["base_prob_draw"] = output["prob_draw"]
    output["base_prob_home_win"] = output["prob_home_win"]
    output[PROB_COLUMNS] = probs
    output["predicted_outcome"] = [tw.ID_TO_OUTCOME[int(value)] for value in probs.argmax(axis=1)]

    if "forced_outcome" in output.columns:
        forced = output["forced_outcome"].astype("string").str.strip()
        forced = forced.where(forced.isin(["home_win", "draw", "away_win"]), output["predicted_outcome"])
        output["predicted_outcome"] = forced.astype(str)

    home_xg = pd.to_numeric(output["expected_home_goals"], errors="coerce").fillna(1.2) * args.goal_multiplier
    away_xg = pd.to_numeric(output["expected_away_goals"], errors="coerce").fillna(1.0) * args.goal_multiplier
    if "home_goals_adjust" in output.columns:
        home_xg += pd.to_numeric(output["home_goals_adjust"], errors="coerce").fillna(0.0)
    if "away_goals_adjust" in output.columns:
        away_xg += pd.to_numeric(output["away_goals_adjust"], errors="coerce").fillna(0.0)
    home_xg = np.clip(home_xg.to_numpy(dtype=float), 0.03, 12.0)
    away_xg = np.clip(away_xg.to_numpy(dtype=float), 0.03, 12.0)

    meta = output[["date", "home_team", "away_team", "tournament"]].copy()
    scores = tw.score_prediction_frame(meta, home_xg, away_xg, args.score_max_goals, output["predicted_outcome"].tolist())
    score_cols = [
        "expected_home_goals",
        "expected_away_goals",
        "pool_predicted_home_score",
        "pool_predicted_away_score",
        "pool_predicted_score",
        "pool_predicted_score_probability",
        "pool_predicted_score_outcome",
    ]
    for col in score_cols:
        output[col] = scores[col]
    output["predicted_home_score"] = output["pool_predicted_home_score"]
    output["predicted_away_score"] = output["pool_predicted_away_score"]
    output["predicted_score"] = output["pool_predicted_score"]
    output["predicted_score_outcome"] = output["pool_predicted_score_outcome"]

    if "forced_score" in output.columns:
        forced_score = output["forced_score"].astype("string").str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
        forced_mask = forced_score.notna().all(axis=1)
        output.loc[forced_mask, "predicted_home_score"] = forced_score.loc[forced_mask, 0].astype(int)
        output.loc[forced_mask, "predicted_away_score"] = forced_score.loc[forced_mask, 1].astype(int)
        output.loc[forced_mask, "predicted_score"] = (
            output.loc[forced_mask, "predicted_home_score"].astype(str)
            + "-"
            + output.loc[forced_mask, "predicted_away_score"].astype(str)
        )
        output.loc[forced_mask, "predicted_score_outcome"] = [
            tw.outcome_label(int(home), int(away))
            for home, away in zip(
                output.loc[forced_mask, "predicted_home_score"],
                output.loc[forced_mask, "predicted_away_score"],
            )
        ]

    output["tweak_draw_multiplier"] = args.draw_multiplier
    output["tweak_goal_multiplier"] = args.goal_multiplier
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote tweaked predictions to {args.output}")
    print(f"Wrote manual tweak template to {args.template_output}")


if __name__ == "__main__":
    main()
