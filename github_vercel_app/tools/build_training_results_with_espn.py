#!/usr/bin/env python
"""Build a training results file with completed ESPN World Cup rows merged in.

The canonical martj42 results.csv stays untouched. This script creates a
temporary/generated training CSV so the model can learn from completed World Cup
matches when the upstream dataset lags behind ESPN.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = APP_ROOT.parent
DEFAULT_BASE = DEFAULT_MODEL_ROOT / "data" / "results.csv"
DEFAULT_ESPN = DEFAULT_MODEL_ROOT / "data" / "extracted" / "espn_worldcup2026_results.csv"
DEFAULT_OUTPUT = DEFAULT_MODEL_ROOT / "data" / "extracted" / "results_training_with_espn.csv"

REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]

TEAM_ALIASES = {
    "bosnia herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "czechia": "Czech Republic",
    "czech republic": "Czech Republic",
    "czech rep": "Czech Republic",
    "turkiye": "Turkey",
    "turkey": "Turkey",
    "usa": "United States",
    "u s a": "United States",
    "united states of america": "United States",
    "cote d ivoire": "Ivory Coast",
    "cote divoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "curacao": "Curaçao",
    "curaçao": "Curaçao",
    "dr congo": "DR Congo",
    "d r congo": "DR Congo",
    "congo dr": "DR Congo",
    "democratic republic of congo": "DR Congo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge completed ESPN WC results into a training CSV.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--espn", type=Path, default=DEFAULT_ESPN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--date-window-days",
        type=int,
        default=1,
        help="Allowed date difference when matching ESPN UTC dates to local fixture dates.",
    )
    return parser.parse_args()


def normalise_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_team(value: object) -> str:
    original = "" if value is None else str(value).strip()
    key = normalise_text(original)
    return TEAM_ALIASES.get(key, original)


def team_key(value: object) -> str:
    return normalise_text(canonical_team(value))


def is_score(value: object) -> bool:
    text = "" if value is None else str(value).strip()
    return bool(re.fullmatch(r"\d+", text))


def scores_present(row: pd.Series) -> bool:
    return is_score(row.get("home_score")) and is_score(row.get("away_score"))


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_base(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    base = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in base.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return base


def load_espn(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=["date", "home_team", "away_team", "home_score", "away_score", "completed"]
        )
    espn = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"date", "home_team", "away_team", "home_score", "away_score"}.difference(espn.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "completed" not in espn.columns:
        espn["completed"] = "true"
    return espn


def find_base_match(
    work: pd.DataFrame,
    match_date: pd.Timestamp,
    home_key: str,
    away_key: str,
    date_window_days: int,
) -> tuple[int | None, bool]:
    direct = (work["_home_key"] == home_key) & (work["_away_key"] == away_key)
    reverse = (work["_home_key"] == away_key) & (work["_away_key"] == home_key)
    candidates = work.loc[(direct | reverse) & work["_date_dt"].notna()].copy()
    if candidates.empty:
        return None, False
    candidates["_date_diff"] = (candidates["_date_dt"] - match_date).abs().dt.days
    candidates = candidates.loc[candidates["_date_diff"] <= date_window_days]
    if candidates.empty:
        return None, False
    candidates["_direct"] = (candidates["_home_key"] == home_key) & (candidates["_away_key"] == away_key)
    candidates = candidates.sort_values(["_date_diff", "_direct"], ascending=[True, False])
    idx = int(candidates.index[0])
    return idx, bool(candidates.loc[idx, "_direct"])


def build_fallback_row(base_columns: list[str], row: pd.Series) -> dict[str, str]:
    output = {column: "" for column in base_columns}
    output.update(
        {
            "date": str(row["date"])[:10],
            "home_team": canonical_team(row["home_team"]),
            "away_team": canonical_team(row["away_team"]),
            "home_score": str(int(float(str(row["home_score"])))),
            "away_score": str(int(float(str(row["away_score"])))),
            "tournament": "FIFA World Cup",
            "city": "",
            "country": "",
            "neutral": "TRUE",
        }
    )
    return output


def merge_results(base: pd.DataFrame, espn: pd.DataFrame, date_window_days: int) -> tuple[pd.DataFrame, dict[str, int]]:
    merged = base.copy()
    work = merged.copy()
    work["_date_dt"] = pd.to_datetime(work["date"], errors="coerce")
    work["_home_key"] = work["home_team"].map(team_key)
    work["_away_key"] = work["away_team"].map(team_key)

    stats = {
        "espn_completed": 0,
        "filled_missing_scores": 0,
        "overwritten_conflicts": 0,
        "skipped_existing": 0,
        "appended_unmatched": 0,
        "skipped_invalid": 0,
    }
    appended_rows: list[dict[str, str]] = []

    seen: set[tuple[str, str, str]] = set()
    for _, row in espn.iterrows():
        if not boolish(row.get("completed", "true")):
            continue
        if not (is_score(row.get("home_score")) and is_score(row.get("away_score"))):
            stats["skipped_invalid"] += 1
            continue
        match_date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(match_date):
            stats["skipped_invalid"] += 1
            continue

        home_key = team_key(row["home_team"])
        away_key = team_key(row["away_team"])
        dedupe_key = (match_date.date().isoformat(), home_key, away_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        stats["espn_completed"] += 1

        home_score = str(int(float(str(row["home_score"]))))
        away_score = str(int(float(str(row["away_score"]))))
        idx, is_direct = find_base_match(work, match_date, home_key, away_key, date_window_days)
        if idx is None:
            appended_rows.append(build_fallback_row(list(merged.columns), row))
            stats["appended_unmatched"] += 1
            continue

        target_home_score = home_score if is_direct else away_score
        target_away_score = away_score if is_direct else home_score
        current = merged.loc[idx]
        if scores_present(current):
            if str(current["home_score"]).strip() == target_home_score and str(current["away_score"]).strip() == target_away_score:
                stats["skipped_existing"] += 1
            else:
                merged.at[idx, "home_score"] = target_home_score
                merged.at[idx, "away_score"] = target_away_score
                stats["overwritten_conflicts"] += 1
            continue

        merged.at[idx, "home_score"] = target_home_score
        merged.at[idx, "away_score"] = target_away_score
        stats["filled_missing_scores"] += 1

    if appended_rows:
        merged = pd.concat([merged, pd.DataFrame(appended_rows, columns=merged.columns)], ignore_index=True)
    return merged, stats


def main() -> None:
    args = parse_args()
    base = load_base(args.base)
    espn = load_espn(args.espn)
    merged, stats = merge_results(base, espn, args.date_window_days)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False, encoding="utf-8")
    print(
        "Built ESPN-merged training results: "
        f"base_rows={len(base):,}, output_rows={len(merged):,}, "
        f"espn_completed={stats['espn_completed']:,}, "
        f"filled={stats['filled_missing_scores']:,}, "
        f"overwritten={stats['overwritten_conflicts']:,}, "
        f"existing={stats['skipped_existing']:,}, "
        f"appended={stats['appended_unmatched']:,}, "
        f"invalid={stats['skipped_invalid']:,}, "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
