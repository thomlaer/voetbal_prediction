"""
Extract compact 1X2 closing odds from the Beat The Bookie Kaggle dataset.

The raw dataset contains mostly club football. This script keeps the columns the
model can consume and reports how many rows match martj42 international results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train_xgboost_worldcup import normalize_name


DEFAULT_BTB_DIR = Path("data/kagglehub/datasets/austro/beat-the-bookie-worldwide-football-dataset/versions/2")
DEFAULT_RESULTS = Path("data/results.csv")
DEFAULT_OUTPUT = Path("data/extracted/beat_the_bookie_closing_1x2.csv")
DEFAULT_MATCH_REPORT = Path("outputs/beat_the_bookie_odds_match_report.csv")


def standardize_closing_odds(source: Path) -> pd.DataFrame:
    odds = pd.read_csv(source, encoding="latin1")
    output = odds.rename(
        columns={
            "match_date": "date",
            "avg_odds_home_win": "home_odds",
            "avg_odds_draw": "draw_odds",
            "avg_odds_away_win": "away_odds",
            "max_odds_home_win": "max_home_odds",
            "max_odds_draw": "max_draw_odds",
            "max_odds_away_win": "max_away_odds",
        }
    )
    keep_cols = [
        "match_id",
        "league",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_odds",
        "draw_odds",
        "away_odds",
        "max_home_odds",
        "max_draw_odds",
        "max_away_odds",
        "top_bookie_home_win",
        "top_bookie_draw",
        "top_bookie_away_win",
        "n_odds_home_win",
        "n_odds_draw",
        "n_odds_away_win",
    ]
    output = output[keep_cols].copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.date.astype(str)
    for col in [
        "home_score",
        "away_score",
        "home_odds",
        "draw_odds",
        "away_odds",
        "max_home_odds",
        "max_draw_odds",
        "max_away_odds",
        "n_odds_home_win",
        "n_odds_draw",
        "n_odds_away_win",
    ]:
        output[col] = pd.to_numeric(output[col], errors="coerce")
    return output.dropna(subset=["date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds"])


def match_report(odds: pd.DataFrame, results_path: Path) -> pd.DataFrame:
    results = pd.read_csv(results_path, usecols=["date", "home_team", "away_team", "tournament", "home_score", "away_score"])
    results = results.dropna(subset=["home_score", "away_score"]).copy()
    results["date_key"] = pd.to_datetime(results["date"], errors="coerce").dt.date.astype(str)
    results["home_key"] = results["home_team"].map(normalize_name)
    results["away_key"] = results["away_team"].map(normalize_name)

    odds_keys = odds.copy()
    odds_keys["date_key"] = pd.to_datetime(odds_keys["date"], errors="coerce").dt.date.astype(str)
    odds_keys["home_key"] = odds_keys["home_team"].map(normalize_name)
    odds_keys["away_key"] = odds_keys["away_team"].map(normalize_name)
    odds_keys = odds_keys.drop_duplicates(["date_key", "home_key", "away_key"])

    merged = results.merge(
        odds_keys[["date_key", "home_key", "away_key", "league", "home_odds", "draw_odds", "away_odds"]],
        on=["date_key", "home_key", "away_key"],
        how="left",
    )
    by_tournament = (
        merged.assign(has_odds=merged["home_odds"].notna())
        .groupby("tournament", dropna=False)
        .agg(matches=("date", "size"), odds_matches=("has_odds", "sum"))
        .reset_index()
    )
    by_tournament["coverage"] = by_tournament["odds_matches"] / by_tournament["matches"]
    return by_tournament.sort_values(["odds_matches", "coverage"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Beat The Bookie closing 1X2 odds.")
    parser.add_argument("--btb-dir", type=Path, default=DEFAULT_BTB_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--match-report", type=Path, default=DEFAULT_MATCH_REPORT)
    args = parser.parse_args()

    odds = standardize_closing_odds(args.btb_dir / "closing_odds.csv.gz")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.match_report.parent.mkdir(parents=True, exist_ok=True)
    odds.to_csv(args.output, index=False)

    report = match_report(odds, args.results)
    report.to_csv(args.match_report, index=False)

    print(f"Wrote {len(odds):,} closing odds rows to {args.output}")
    print(f"Wrote match report to {args.match_report}")
    print(report.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
