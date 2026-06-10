#!/usr/bin/env python
"""Build expected group tables from Monte Carlo team probabilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create worldcup2026_expected_group_tables.csv")
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--team-probabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedule = pd.read_csv(args.schedule)
    team_probs = pd.read_csv(args.team_probabilities)

    group_rows = schedule[schedule["stage"].eq("Group Stage")]
    team_groups: dict[str, str] = {}
    for row in group_rows.itertuples(index=False):
        team_groups[str(row.home_team)] = str(row.group)
        team_groups[str(row.away_team)] = str(row.group)

    groups = pd.DataFrame(
        [{"team": team, "group": group} for team, group in sorted(team_groups.items())]
    )
    output = groups.merge(team_probs, on="team", how="left")
    first_cols = ["group", "team"]
    output = output[first_cols + [col for col in output.columns if col not in first_cols]]
    output = output.sort_values(["group", "expected_group_rank", "team"], na_position="last")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output):,} expected group rows to {args.output}")


if __name__ == "__main__":
    main()
