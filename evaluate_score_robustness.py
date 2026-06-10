#!/usr/bin/env python
"""Compare safe, tuned and simple hybrid score strategies on World Cups 2018/2022."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import make_scorito_worldcup_picks as scorer


OUTPUT = Path("outputs_worldcup2026_default/score_strategy_2018_2022_hybrid_check.csv")


def outcome(home: int, away: int) -> str:
    if home > away:
        return "home_win"
    if home < away:
        return "away_win"
    return "draw"


def points(pred_home: int, pred_away: int, actual_home: int, actual_away: int) -> float:
    pred_outcome = outcome(pred_home, pred_away)
    actual_outcome = outcome(actual_home, actual_away)
    if pred_home == actual_home and pred_away == actual_away:
        return 5.0
    return (
        (2.0 if pred_outcome == actual_outcome else 0.0)
        + (1.0 if pred_home - pred_away == actual_home - actual_away else 0.0)
        + (0.5 if pred_home == actual_home else 0.0)
        + (0.5 if pred_away == actual_away else 0.0)
    )


def score_tuple(score: str) -> tuple[int, int]:
    home, away = score.split("-", 1)
    return int(home), int(away)


def load_worldcup_predictions(year: int, path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    rows = raw[raw["tournament"].eq("FIFA World Cup") & raw["date"].dt.year.eq(year)].copy().reset_index(drop=True)
    picks = [
        scorer.optimize_score_for_pool(row, 6, 0.10, 5.0, 2.0, 1.0, 0.5)
        for _, row in rows.iterrows()
    ]
    tuned_input = pd.concat(
        [
            rows[
                [
                    "date",
                    "home_team",
                    "away_team",
                    "prob_home_win",
                    "prob_draw",
                    "prob_away_win",
                    "expected_home_goals",
                    "expected_away_goals",
                ]
            ].copy(),
            pd.DataFrame(picks),
        ],
        axis=1,
    )
    tuned = scorer.apply_recommended_scores(tuned_input)

    output_rows = []
    for idx, row in rows.iterrows():
        safe_score = f"{int(row.pool_predicted_home_score)}-{int(row.pool_predicted_away_score)}"
        tuned_score = str(tuned.loc[idx, "score"])
        actual_score = f"{int(row.home_score)}-{int(row.away_score)}"
        safe_home, safe_away = score_tuple(safe_score)
        tuned_home, tuned_away = score_tuple(tuned_score)
        actual_home, actual_away = score_tuple(actual_score)
        probs = {
            "home_win": float(row.prob_home_win),
            "draw": float(row.prob_draw),
            "away_win": float(row.prob_away_win),
        }
        ordered = sorted(probs.values(), reverse=True)
        output_rows.append(
            {
                "year": year,
                "safe": safe_score,
                "tuned": tuned_score,
                "actual": actual_score,
                "fav_prob": ordered[0],
                "margin": ordered[0] - ordered[1],
                "draw_prob": float(row.prob_draw),
                "total_xg": float(row.expected_home_goals) + float(row.expected_away_goals),
                "safe_pts": points(safe_home, safe_away, actual_home, actual_away),
                "tuned_pts": points(tuned_home, tuned_away, actual_home, actual_away),
                "safe_exact": safe_score == actual_score,
                "tuned_exact": tuned_score == actual_score,
                "safe_out": outcome(safe_home, safe_away) == row.actual_outcome,
                "tuned_out": outcome(tuned_home, tuned_away) == row.actual_outcome,
            }
        )
    return pd.DataFrame(output_rows)


def summarize_strategy(name: str, predictions: pd.Series, actuals: pd.Series, points_values: np.ndarray, tuned_used: int) -> dict:
    exact = predictions.eq(actuals).mean()
    outs = []
    for pred, actual in zip(predictions, actuals):
        pred_home, pred_away = score_tuple(pred)
        actual_home, actual_away = score_tuple(actual)
        outs.append(outcome(pred_home, pred_away) == outcome(actual_home, actual_away))
    return {
        "name": name,
        "total": float(points_values.sum()),
        "avg": float(points_values.mean()),
        "exact": float(exact),
        "outcome": float(np.mean(outs)),
        "one_nil": int(predictions.isin(["1-0", "0-1"]).sum()),
        "tuned_used": int(tuned_used),
    }


def main() -> None:
    data = pd.concat(
        [
            load_worldcup_predictions(2018, "outputs_backtest_2018/test_predictions_xgboost.csv"),
            load_worldcup_predictions(2022, "outputs_backtest_2022/test_predictions_xgboost.csv"),
        ],
        ignore_index=True,
    )
    strategies = [
        summarize_strategy("safe", data["safe"], data["actual"], data["safe_pts"].to_numpy(), 0),
        summarize_strategy("tuned", data["tuned"], data["actual"], data["tuned_pts"].to_numpy(), len(data)),
    ]
    for fav_prob in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for margin in [0.08, 0.12, 0.16, 0.20, 0.25]:
            for total_xg in [2.0, 2.2, 2.4, 2.6]:
                use_tuned = (
                    data["fav_prob"].lt(fav_prob)
                    | (data["margin"].lt(margin) & data["draw_prob"].ge(0.25))
                    | data["total_xg"].ge(total_xg)
                )
                predictions = data["safe"].where(~use_tuned, data["tuned"])
                points_values = data["safe_pts"].where(~use_tuned, data["tuned_pts"]).to_numpy()
                summary = summarize_strategy(
                    f"hybrid_fp{fav_prob}_m{margin}_xg{total_xg}",
                    predictions,
                    data["actual"],
                    points_values,
                    int(use_tuned.sum()),
                )
                summary.update({"fav_prob_thr": fav_prob, "margin_thr": margin, "total_xg_thr": total_xg})
                strategies.append(summary)

    result = pd.DataFrame(strategies).sort_values(["total", "exact", "outcome"], ascending=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.head(20).to_string(index=False))
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
