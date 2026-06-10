#!/usr/bin/env python
"""Test explicit close-match score mechanisms on World Cup backtests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import make_scorito_worldcup_picks as scorer


OUTPUT = Path("outputs_worldcup2026_default/close_match_score_mechanism_grid.csv")


def outcome(home: int, away: int) -> str:
    if home > away:
        return "home_win"
    if home < away:
        return "away_win"
    return "draw"


def split_score(score: str) -> tuple[int, int]:
    home, away = score.split("-", 1)
    return int(home), int(away)


def points(prediction: str, actual: str) -> float:
    pred_home, pred_away = split_score(prediction)
    actual_home, actual_away = split_score(actual)
    if pred_home == actual_home and pred_away == actual_away:
        return 5.0
    pred_outcome = outcome(pred_home, pred_away)
    actual_outcome = outcome(actual_home, actual_away)
    return (
        (2.0 if pred_outcome == actual_outcome else 0.0)
        + (1.0 if pred_home - pred_away == actual_home - actual_away else 0.0)
        + (0.5 if pred_home == actual_home else 0.0)
        + (0.5 if pred_away == actual_away else 0.0)
    )


def draw_score(total_xg: float) -> str:
    if total_xg < 1.85:
        return "0-0"
    if total_xg >= 3.15:
        return "2-2"
    return "1-1"


def oriented_score(fav_goals: int, dog_goals: int, side: str) -> str:
    if side == "home_win":
        return f"{fav_goals}-{dog_goals}"
    if side == "away_win":
        return f"{dog_goals}-{fav_goals}"
    return f"{fav_goals}-{fav_goals}"


def build_backtest_rows() -> pd.DataFrame:
    rows = []
    for year, path in [
        (2018, "outputs_backtest_2018/test_predictions_xgboost.csv"),
        (2022, "outputs_backtest_2022/test_predictions_xgboost.csv"),
    ]:
        raw = pd.read_csv(path)
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        wc = raw[raw["tournament"].eq("FIFA World Cup") & raw["date"].dt.year.eq(year)].copy().reset_index(drop=True)
        picks = [
            scorer.optimize_score_for_pool(row, 6, 0.10, 5.0, 2.0, 1.0, 0.5)
            for _, row in wc.iterrows()
        ]
        candidate = pd.concat(
            [
                wc[
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
        current = scorer.apply_recommended_scores(candidate)
        for idx, row in wc.iterrows():
            probabilities = {
                "home_win": float(row.prob_home_win),
                "draw": float(row.prob_draw),
                "away_win": float(row.prob_away_win),
            }
            ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            actual = f"{int(row.home_score)}-{int(row.away_score)}"
            safe = f"{int(row.pool_predicted_home_score)}-{int(row.pool_predicted_away_score)}"
            tuned = str(current.loc[idx, "upside_score"])
            hybrid = str(current.loc[idx, "score"])
            total_xg = float(row.expected_home_goals) + float(row.expected_away_goals)
            rows.append(
                {
                    "year": year,
                    "actual": actual,
                    "safe": safe,
                    "tuned": tuned,
                    "hybrid": hybrid,
                    "top_outcome": ranked[0][0],
                    "top_prob": ranked[0][1],
                    "second_outcome": ranked[1][0],
                    "second_prob": ranked[1][1],
                    "margin": ranked[0][1] - ranked[1][1],
                    "draw_prob": probabilities["draw"],
                    "home_prob": probabilities["home_win"],
                    "away_prob": probabilities["away_win"],
                    "total_xg": total_xg,
                    "safe_points": points(safe, actual),
                    "tuned_points": points(tuned, actual),
                    "hybrid_points": points(hybrid, actual),
                }
            )
    return pd.DataFrame(rows)


def summarize(name: str, frame: pd.DataFrame, predictions: pd.Series, changed: pd.Series) -> dict:
    point_values = np.array([points(pred, actual) for pred, actual in zip(predictions, frame["actual"])])
    exact = predictions.eq(frame["actual"])
    outcomes = [
        outcome(*split_score(pred)) == outcome(*split_score(actual))
        for pred, actual in zip(predictions, frame["actual"])
    ]
    return {
        "name": name,
        "total_points": float(point_values.sum()),
        "avg_points": float(point_values.mean()),
        "exact_accuracy": float(exact.mean()),
        "outcome_accuracy": float(np.mean(outcomes)),
        "changed_matches": int(changed.sum()),
        "one_nil_picks": int(predictions.isin(["1-0", "0-1"]).sum()),
        "draw_picks": int(sum(outcome(*split_score(score)) == "draw" for score in predictions)),
        "points_2018": float(point_values[frame["year"].eq(2018).to_numpy()].sum()),
        "points_2022": float(point_values[frame["year"].eq(2022).to_numpy()].sum()),
    }


def main() -> None:
    frame = build_backtest_rows()
    rows = [
        summarize("safe", frame, frame["safe"], pd.Series(False, index=frame.index)),
        summarize("tuned", frame, frame["tuned"], pd.Series(True, index=frame.index)),
        summarize("current_hybrid", frame, frame["hybrid"], frame["hybrid"].ne(frame["safe"])),
    ]

    for margin in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22]:
        for draw_prob in [0.24, 0.26, 0.28, 0.30, 0.32]:
            close = frame["margin"].le(margin) & frame["draw_prob"].ge(draw_prob)

            # Mechanism 1: in close matches, force a draw score.
            pred_draw = frame["hybrid"].where(~close, frame["total_xg"].map(draw_score))
            summary = summarize(f"close_force_draw_m{margin}_d{draw_prob}", frame, pred_draw, close)
            summary.update({"mechanism": "force_draw", "margin": margin, "draw_prob_min": draw_prob})
            rows.append(summary)

            # Mechanism 2: only force draw if draw is second-best and teams are very close.
            close_draw_second = close & frame["second_outcome"].eq("draw")
            pred_draw_second = frame["hybrid"].where(~close_draw_second, frame["total_xg"].map(draw_score))
            summary = summarize(
                f"close_second_draw_m{margin}_d{draw_prob}",
                frame,
                pred_draw_second,
                close_draw_second,
            )
            summary.update({"mechanism": "second_draw", "margin": margin, "draw_prob_min": draw_prob})
            rows.append(summary)

            # Mechanism 3: use second-best non-draw winner in close matches when draw is not strong.
            second_winner = close & frame["second_outcome"].isin(["home_win", "away_win"]) & frame["draw_prob"].lt(0.30)
            second_scores = [
                oriented_score(2, 1, side) if total_xg >= 2.25 else oriented_score(1, 0, side)
                for side, total_xg in zip(frame["second_outcome"], frame["total_xg"])
            ]
            pred_second = frame["hybrid"].where(~second_winner, pd.Series(second_scores, index=frame.index))
            summary = summarize(
                f"close_second_winner_m{margin}_d{draw_prob}",
                frame,
                pred_second,
                second_winner,
            )
            summary.update({"mechanism": "second_winner", "margin": margin, "draw_prob_min": draw_prob})
            rows.append(summary)

    result = pd.DataFrame(rows).sort_values(
        ["total_points", "exact_accuracy", "outcome_accuracy"],
        ascending=False,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.head(25).to_string(index=False))
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
