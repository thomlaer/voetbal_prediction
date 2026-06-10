"""
Audit how Fjelstul World Cup rows align with martj42 international_results.

Outputs are written to outputs/ and are intentionally small CSV reports:
- fjelstul_match_audit_summary.csv
- fjelstul_match_audit_by_year.csv
- fjelstul_match_audit_rows.csv
- fjelstul_match_audit_non_exact.csv
- fjelstul_match_audit_unmatched.csv
- fjelstul_womens_accidental_matches.csv
- fjelstul_manager_attach_by_tournament.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from train_xgboost_worldcup import (
    attach_fjelstul_manager_features,
    build_fjelstul_manager_features,
    normalize_fjelstul_team_name,
    normalize_name,
)


RESULTS_PATH = Path("data/results.csv")
FJELSTUL_DIR = Path("data/fjelstul_worldcup/data-csv")
OUTPUT_DIR = Path("outputs")


def load_results(path: Path) -> pd.DataFrame:
    results = pd.read_csv(path)
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(subset=["date", "home_score", "away_score"]).copy()
    results["home_score"] = results["home_score"].astype(int)
    results["away_score"] = results["away_score"].astype(int)
    results["date_key"] = results["date"].dt.date.astype(str)
    results["home_key"] = results["home_team"].map(normalize_name)
    results["away_key"] = results["away_team"].map(normalize_name)
    results["result_row_id"] = range(len(results))
    return results


def load_fjelstul_matches(path: Path) -> pd.DataFrame:
    matches = pd.read_csv(path)
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
    matches = matches.dropna(subset=["match_date", "home_team_score", "away_team_score"]).copy()
    matches["home_team_score"] = matches["home_team_score"].astype(int)
    matches["away_team_score"] = matches["away_team_score"].astype(int)
    matches["date_key"] = matches["match_date"].dt.date.astype(str)
    matches["home_key"] = [
        normalize_fjelstul_team_name(team, date)
        for team, date in zip(matches["home_team_name"], matches["match_date"], strict=False)
    ]
    matches["away_key"] = [
        normalize_fjelstul_team_name(team, date)
        for team, date in zip(matches["away_team_name"], matches["match_date"], strict=False)
    ]
    matches["is_mens_world_cup"] = matches["tournament_name"].astype(str).str.contains(
        "FIFA Men's World Cup",
        regex=False,
    )
    return matches


def index_results(results: pd.DataFrame) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in results.to_dict("records"):
        key = (row["date_key"], row["home_key"], row["away_key"])
        lookup.setdefault(key, []).append(row)
    return lookup


def index_results_by_pair(results: pd.DataFrame) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in results.to_dict("records"):
        teams = tuple(sorted([row["home_key"], row["away_key"]]))
        key = (row["date_key"], teams[0], teams[1])
        lookup.setdefault(key, []).append(row)
    return lookup


def score_matches(row: pd.Series, candidate: dict[str, Any], reversed_orientation: bool = False) -> bool:
    if reversed_orientation:
        return (
            int(row["home_team_score"]) == int(candidate["away_score"])
            and int(row["away_team_score"]) == int(candidate["home_score"])
        )
    return (
        int(row["home_team_score"]) == int(candidate["home_score"])
        and int(row["away_team_score"]) == int(candidate["away_score"])
    )


def audit_matches(fjelstul: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    exact_lookup = index_results(results)
    pair_lookup = index_results_by_pair(results)
    rows: list[dict[str, Any]] = []

    for _, row in fjelstul.iterrows():
        exact_key = (row["date_key"], row["home_key"], row["away_key"])
        reverse_key = (row["date_key"], row["away_key"], row["home_key"])
        pair_names = tuple(sorted([row["home_key"], row["away_key"]]))
        pair_key = (row["date_key"], pair_names[0], pair_names[1])

        status = "unmatched"
        candidate: dict[str, Any] | None = None

        for possible in exact_lookup.get(exact_key, []):
            candidate = possible
            if score_matches(row, possible):
                status = "matched_exact"
                break
            status = "score_mismatch"

        if status in {"unmatched", "score_mismatch"}:
            for possible in exact_lookup.get(reverse_key, []):
                candidate = possible
                if score_matches(row, possible, reversed_orientation=True):
                    status = "matched_reversed"
                    break
                if status == "unmatched":
                    status = "reverse_score_mismatch"

        if status == "unmatched" and pair_lookup.get(pair_key):
            candidate = pair_lookup[pair_key][0]
            status = "same_date_pair_mismatch"

        rows.append(
            {
                "status": status,
                "fjelstul_tournament": row["tournament_name"],
                "fjelstul_match_id": row["match_id"],
                "date": row["date_key"],
                "fjelstul_home_team": row["home_team_name"],
                "fjelstul_away_team": row["away_team_name"],
                "fjelstul_score": f"{int(row['home_team_score'])}-{int(row['away_team_score'])}",
                "normalized_home": row["home_key"],
                "normalized_away": row["away_key"],
                "results_tournament": None if candidate is None else candidate["tournament"],
                "results_home_team": None if candidate is None else candidate["home_team"],
                "results_away_team": None if candidate is None else candidate["away_team"],
                "results_score": None
                if candidate is None
                else f"{int(candidate['home_score'])}-{int(candidate['away_score'])}",
                "results_row_id": None if candidate is None else candidate["result_row_id"],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    results = load_results(RESULTS_PATH)
    wc_results = results[results["tournament"].eq("FIFA World Cup")].copy()
    fjelstul = load_fjelstul_matches(FJELSTUL_DIR / "matches.csv")
    fjelstul_men = fjelstul[fjelstul["is_mens_world_cup"]].copy()
    fjelstul_women = fjelstul[~fjelstul["is_mens_world_cup"]].copy()

    audit = audit_matches(fjelstul_men, wc_results)
    audit.to_csv(OUTPUT_DIR / "fjelstul_match_audit_rows.csv", index=False)
    audit[audit["status"] != "matched_exact"].to_csv(
        OUTPUT_DIR / "fjelstul_match_audit_non_exact.csv",
        index=False,
    )
    audit[~audit["status"].isin(["matched_exact", "matched_reversed"])].to_csv(
        OUTPUT_DIR / "fjelstul_match_audit_unmatched.csv",
        index=False,
    )

    by_year = (
        audit.assign(year=audit["date"].str.slice(0, 4))
        .groupby(["year", "status"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    by_year.to_csv(OUTPUT_DIR / "fjelstul_match_audit_by_year.csv", index=False)

    women_audit = audit_matches(fjelstul_women, results)
    women_hits = women_audit[women_audit["status"].isin(["matched_exact", "matched_reversed"])].copy()
    women_hits.to_csv(OUTPUT_DIR / "fjelstul_womens_accidental_matches.csv", index=False)

    manager_features = build_fjelstul_manager_features(FJELSTUL_DIR).drop_duplicates(
        subset=["date_key", "team_key", "opponent_key"],
        keep="last",
    )
    results_for_manager = results.copy()
    manager_home = manager_features.rename(columns={"team_key": "home_key", "opponent_key": "away_key"})
    manager_away = manager_features.rename(columns={"team_key": "away_key", "opponent_key": "home_key"})
    home_merge = results_for_manager.merge(
        manager_home[["date_key", "home_key", "away_key", "manager_matches_before"]],
        on=["date_key", "home_key", "away_key"],
        how="left",
    )
    away_merge = results_for_manager.merge(
        manager_away[["date_key", "home_key", "away_key", "manager_matches_before"]],
        on=["date_key", "home_key", "away_key"],
        how="left",
    )
    manager_row_mask = home_merge["manager_matches_before"].notna() | away_merge["manager_matches_before"].notna()
    manager_row_matches = results_for_manager[manager_row_mask].copy()
    _, manager_side_matches = attach_fjelstul_manager_features(results, FJELSTUL_DIR)
    manager_by_tournament = (
        manager_row_matches.groupby("tournament", dropna=False)
        .size()
        .reset_index(name="matched_match_rows")
        .sort_values("matched_match_rows", ascending=False)
    )
    manager_by_tournament.to_csv(OUTPUT_DIR / "fjelstul_manager_attach_by_tournament.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "martj42_complete_mens_world_cup_matches", "value": len(wc_results)},
            {"metric": "fjelstul_mens_world_cup_matches", "value": len(fjelstul_men)},
            {"metric": "fjelstul_womens_world_cup_matches", "value": len(fjelstul_women)},
            {"metric": "matched_exact_mens_rows", "value": int((audit["status"] == "matched_exact").sum())},
            {"metric": "non_exact_mens_rows", "value": int((audit["status"] != "matched_exact").sum())},
            {
                "metric": "unmatched_mens_rows",
                "value": int((~audit["status"].isin(["matched_exact", "matched_reversed"])).sum()),
            },
            {"metric": "accidental_womens_rows_matching_any_mens_result", "value": len(women_hits)},
            {"metric": "manager_feature_side_matches_after_filter", "value": manager_side_matches},
            {"metric": "manager_feature_match_rows_after_filter", "value": len(manager_row_matches)},
        ]
    )
    summary.to_csv(OUTPUT_DIR / "fjelstul_match_audit_summary.csv", index=False)

    print(summary.to_string(index=False))
    print()
    print("Men's match status counts:")
    print(audit["status"].value_counts().to_string())
    print()
    print("Manager matched tournaments:")
    print(manager_by_tournament.to_string(index=False))


if __name__ == "__main__":
    main()
