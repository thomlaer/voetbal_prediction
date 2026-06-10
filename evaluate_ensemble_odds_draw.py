#!/usr/bin/env python
"""Evaluate simple ensembles of the default model, odds-model and bookmaker odds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from train_xgboost_worldcup import normalize_name


DEFAULT_PREDICTIONS = Path("outputs_default/test_predictions_xgboost.csv")
ODDS_MODEL_PREDICTIONS = Path("outputs_feature_v2_soccerbase_stats_oddsportal/test_predictions_xgboost.csv")
ODDS_CSV = Path("data/extracted/oddsportal_international_closing_1x2.csv")
OUTPUT_SUMMARY = Path("outputs_default/ensemble_odds_draw_summary.csv")

OUTCOMES = ["away_win", "draw", "home_win"]
PROB_COLUMNS = ["prob_away_win", "prob_draw", "prob_home_win"]
OUTCOME_TO_ID = {name: index for index, name in enumerate(OUTCOMES)}


def prediction_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["date_key"] = out["date"].dt.date.astype(str)
    out["home_key"] = out["home_team"].map(normalize_name)
    out["away_key"] = out["away_team"].map(normalize_name)
    return out


def probability_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[PROB_COLUMNS].to_numpy(dtype=float)


def predicted_labels(probs: np.ndarray) -> np.ndarray:
    return np.array([OUTCOMES[index] for index in probs.argmax(axis=1)])


def metric_row(
    name: str,
    frame: pd.DataFrame,
    probs: np.ndarray,
    split: str,
    subset: str = "all",
    params: str = "",
) -> dict[str, object]:
    if subset == "has_odds":
        mask = frame["has_odds"].to_numpy(dtype=bool)
    elif subset == "no_odds":
        mask = ~frame["has_odds"].to_numpy(dtype=bool)
    else:
        mask = np.ones(len(frame), dtype=bool)

    if not mask.any():
        return {
            "strategy": name,
            "split": split,
            "subset": subset,
            "params": params,
            "rows": 0,
            "accuracy": np.nan,
            "log_loss": np.nan,
            "pred_draws": 0,
        }

    part = frame.loc[mask]
    part_probs = probs[mask]
    pred = predicted_labels(part_probs)
    y_true = part["actual_outcome"].map(OUTCOME_TO_ID).to_numpy(dtype=int)
    return {
        "strategy": name,
        "split": split,
        "subset": subset,
        "params": params,
        "rows": int(mask.sum()),
        "accuracy": float(np.mean(pred == part["actual_outcome"].to_numpy())),
        "log_loss": float(log_loss(y_true, part_probs, labels=[0, 1, 2])),
        "pred_draws": int((pred == "draw").sum()),
    }


def apply_draw_margin(probs: np.ndarray, margin: float) -> np.ndarray:
    adjusted = probs.copy()
    draw = adjusted[:, OUTCOME_TO_ID["draw"]]
    max_non_draw = np.maximum(adjusted[:, OUTCOME_TO_ID["away_win"]], adjusted[:, OUTCOME_TO_ID["home_win"]])
    draw_mask = draw >= max_non_draw - margin
    adjusted[draw_mask, :] = 0.0
    adjusted[draw_mask, OUTCOME_TO_ID["draw"]] = 1.0
    return adjusted


def load_bookie_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    odds = pd.read_csv(ODDS_CSV, low_memory=False)
    odds["date"] = pd.to_datetime(odds["date"], errors="coerce")
    odds["date_key"] = odds["date"].dt.date.astype(str)
    odds["home_key"] = odds["home_team"].map(normalize_name)
    odds["away_key"] = odds["away_team"].map(normalize_name)
    odds = odds.dropna(subset=["date_key", "home_key", "away_key", "home_odds", "draw_odds", "away_odds"])
    odds = odds.drop_duplicates(subset=["date_key", "home_key", "away_key"], keep="last")
    for col in ["home_odds", "draw_odds", "away_odds"]:
        odds[col] = pd.to_numeric(odds[col], errors="coerce")

    implied = 1.0 / odds[["away_odds", "draw_odds", "home_odds"]]
    implied_sum = implied.sum(axis=1)
    odds["bookie_prob_away_win"] = implied["away_odds"] / implied_sum
    odds["bookie_prob_draw"] = implied["draw_odds"] / implied_sum
    odds["bookie_prob_home_win"] = implied["home_odds"] / implied_sum

    return frame.merge(
        odds[
            [
                "date_key",
                "home_key",
                "away_key",
                "bookie_prob_away_win",
                "bookie_prob_draw",
                "bookie_prob_home_win",
            ]
        ],
        on=["date_key", "home_key", "away_key"],
        how="left",
    )


def blend_with_bookie(default_probs: np.ndarray, frame: pd.DataFrame, weight: float) -> np.ndarray:
    bookie_cols = ["bookie_prob_away_win", "bookie_prob_draw", "bookie_prob_home_win"]
    bookie = frame[bookie_cols].to_numpy(dtype=float)
    has_odds = frame["has_odds"].to_numpy(dtype=bool)
    blended = default_probs.copy()
    blended[has_odds] = (1.0 - weight) * default_probs[has_odds] + weight * bookie[has_odds]
    blended = blended / blended.sum(axis=1, keepdims=True)
    return blended


def main() -> None:
    default = prediction_keys(pd.read_csv(DEFAULT_PREDICTIONS))
    odds_model = prediction_keys(pd.read_csv(ODDS_MODEL_PREDICTIONS))
    key_cols = ["date_key", "home_key", "away_key"]
    if not default[key_cols].equals(odds_model[key_cols]):
        raise ValueError("Default and odds-model predictions do not align row-for-row.")

    frame = load_bookie_probabilities(default)
    frame["has_odds"] = frame[["bookie_prob_away_win", "bookie_prob_draw", "bookie_prob_home_win"]].notna().all(axis=1)
    default_probs = probability_matrix(default)
    odds_model_probs = probability_matrix(odds_model)

    cal_mask = frame["date"] < pd.Timestamp("2025-01-01")
    val_mask = ~cal_mask
    splits = {
        "cal_2023_2024": (frame.loc[cal_mask].reset_index(drop=True), default_probs[cal_mask], odds_model_probs[cal_mask]),
        "val_2025_2026": (frame.loc[val_mask].reset_index(drop=True), default_probs[val_mask], odds_model_probs[val_mask]),
        "all_2023_2026": (frame, default_probs, odds_model_probs),
    }

    rows: list[dict[str, object]] = []
    for split_name, (split_frame, split_default, split_odds_model) in splits.items():
        strategies: list[tuple[str, np.ndarray, str]] = [
            ("default", split_default, ""),
            ("odds_xgb_model", split_odds_model, ""),
            ("bookie_fallback_default", blend_with_bookie(split_default, split_frame, 1.0), "bookie_weight=1.00"),
        ]
        for weight in np.linspace(0.05, 0.95, 19):
            strategies.append(
                (
                    "blend_default_odds_xgb",
                    (1.0 - weight) * split_default + weight * split_odds_model,
                    f"odds_xgb_weight={weight:.2f}",
                )
            )
            strategies.append(
                (
                    "blend_default_bookie",
                    blend_with_bookie(split_default, split_frame, float(weight)),
                    f"bookie_weight={weight:.2f}",
                )
            )

        for name, probs, params in strategies:
            probs = probs / probs.sum(axis=1, keepdims=True)
            for subset in ("all", "has_odds", "no_odds"):
                rows.append(metric_row(name, split_frame, probs, split_name, subset, params))
            for margin in np.linspace(0.00, 0.08, 9):
                adjusted = apply_draw_margin(probs, float(margin))
                rows.append(
                    metric_row(
                        f"{name}+draw_margin",
                        split_frame,
                        adjusted,
                        split_name,
                        "all",
                        f"{params};draw_margin={margin:.2f}".strip(";"),
                    )
                )

    summary = pd.DataFrame(rows).sort_values(["split", "subset", "accuracy", "log_loss"], ascending=[True, True, False, True])
    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    for split_name in ("cal_2023_2024", "val_2025_2026", "all_2023_2026"):
        print(f"\n{split_name}")
        view = summary[(summary["split"] == split_name) & (summary["subset"] == "all")]
        print(view.head(12)[["strategy", "params", "rows", "accuracy", "log_loss", "pred_draws"]].to_string(index=False))
    print(f"\nSaved summary to {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
