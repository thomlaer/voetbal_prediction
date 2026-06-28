"""World Cup 2026 bracket prediction + combined team summary CSV.

Leest de Monte Carlo outputs en produceert:
  1. worldcup2026_combined_team_summary.csv  – alle teamdata samengevoegd
  2. worldcup2026_bracket_prediction.csv     – volledig speelschema groepsfase t/m finale

Gebruik:
    python predict_worldcup2026_bracket.py [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from predict_worldcup2026_montecarlo import (
    build_team_strength,
    load_latest_fifa_points,
    merge_schedule_predictions,
    sigmoid,
)
from train_xgboost_worldcup import normalize_name


DEFAULT_OUTPUT_DIR = Path("outputs_worldcup2026_default")
SCHEDULE = Path("data/extracted/oddsportal_worldcup2026_fixture_odds_schedule.csv")
MODEL_PREDS = Path("outputs_worldcup2026_default/future_predictions_xgboost.csv")
RANKINGS = Path("fifa_ranking-2026-04-01.csv")
DEFAULT_RESULTS = Path("data/results.csv")
ODDS_WEIGHT = 0.70


# ---------------------------------------------------------------------------
# Combined team summary
# ---------------------------------------------------------------------------

def combine_team_summary(output_dir: Path) -> pd.DataFrame:
    group_tables = pd.read_csv(output_dir / "worldcup2026_expected_group_tables.csv")
    champion_counts = pd.read_csv(output_dir / "worldcup2026_montecarlo_champion_counts.csv")
    merged = group_tables.merge(
        champion_counts[["team", "champion_count"]],
        on="team",
        how="left",
    )
    merged["champion_count"] = merged["champion_count"].fillna(0).astype(int)
    col_order = (
        ["group", "team", "champion_count", "champion_prob"]
        + [c for c in merged.columns if c not in ("group", "team", "champion_count", "champion_prob")]
    )
    return merged[col_order].sort_values(["group", "expected_group_rank"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Deterministic group results
# ---------------------------------------------------------------------------

def determine_group_order(group_tables: pd.DataFrame) -> dict[str, list[str]]:
    group_order: dict[str, list[str]] = {}
    for group, rows in group_tables.groupby("group"):
        ranked = rows.sort_values("expected_group_rank")
        group_order[str(group)] = ranked["team"].tolist()
    return group_order


def determine_third_order(
    group_tables: pd.DataFrame, group_order: dict[str, list[str]]
) -> list[tuple[str, str, dict]]:
    thirds: list[tuple[str, str, dict]] = []
    for group, teams in group_order.items():
        if len(teams) < 3:
            continue
        third = teams[2]
        row = group_tables[(group_tables["group"] == group) & (group_tables["team"] == third)]
        if row.empty:
            continue
        r = row.iloc[0]
        thirds.append((
            group,
            third,
            {
                "points": float(r["expected_group_points"]),
                "gd": float(r["expected_group_gd"]),
                "gf": float(r["expected_group_gf"]),
                "wins": 0,
            },
        ))
    thirds.sort(key=lambda x: (x[2]["points"], x[2]["gd"], x[2]["gf"]), reverse=True)
    return thirds[:8]


# ---------------------------------------------------------------------------
# Bracket resolution
# ---------------------------------------------------------------------------

def resolve_slot(
    label: str,
    group_order: dict[str, list[str]],
    third_order: list[tuple[str, str, dict]],
    used_third_groups: set[str],
    winners: dict[int, str],
    losers: dict[int, str],
) -> str:
    if re.fullmatch(r"[12][A-L]", label):
        pos = int(label[0]) - 1
        group = label[1]
        return group_order[group][pos]
    if re.fullmatch(r"3[A-L]+", label):
        eligible = set(label[1:])
        for group, team, _ in third_order:
            if group in eligible and group not in used_third_groups:
                used_third_groups.add(group)
                return team
        for group, team, _ in third_order:
            if group not in used_third_groups:
                used_third_groups.add(group)
                return team
        raise ValueError(f"Kan derde-plaatssleuf niet oplossen: {label}")
    if label.startswith("RU"):
        match_num = int(label[2:])
        if match_num not in losers:
            raise ValueError(f"Verliezer van wedstrijd {match_num} nog niet bekend")
        return losers[match_num]
    if label.startswith("W"):
        match_num = int(label[1:])
        if match_num not in winners:
            raise ValueError(f"Winnaar van wedstrijd {match_num} nog niet bekend")
        return winners[match_num]
    if label and label.lower() != "nan":
        return label
    raise ValueError(f"Onbekend label: {label}")


def is_placeholder_slot(value: object) -> bool:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return True
    return bool(
        re.fullmatch(r"(?:W|RU|L)\d+", text, flags=re.IGNORECASE)
        or re.fullmatch(r"[123][A-L]+", text, flags=re.IGNORECASE)
        or text.lower().startswith(("winner ", "runner-up ", "group "))
    )


def known_fixture_prediction(
    fixture: pd.Series | None,
    home: str,
    away: str,
    stage: str,
) -> dict[str, float | str] | None:
    if fixture is None:
        return None
    fixture_home = str(fixture.get("home_team", ""))
    fixture_away = str(fixture.get("away_team", ""))
    if is_placeholder_slot(fixture_home) or is_placeholder_slot(fixture_away):
        return None

    direct = (
        normalize_name(fixture_home) == normalize_name(home)
        and normalize_name(fixture_away) == normalize_name(away)
    )
    reverse = (
        normalize_name(fixture_home) == normalize_name(away)
        and normalize_name(fixture_away) == normalize_name(home)
    )
    if not direct and not reverse:
        return None

    try:
        p_home = float(fixture.get("prob_home_win"))
        p_draw = float(fixture.get("prob_draw"))
        p_away = float(fixture.get("prob_away_win"))
        home_xg = float(fixture.get("expected_home_goals"))
        away_xg = float(fixture.get("expected_away_goals"))
    except (TypeError, ValueError):
        return None
    if not np.isfinite([p_home, p_draw, p_away, home_xg, away_xg]).all():
        return None
    if reverse:
        p_home, p_away = p_away, p_home
        home_xg, away_xg = away_xg, home_xg

    if "third place" in str(stage).lower():
        p_draw = 0.0
    total = p_home + p_draw + p_away
    if total <= 0:
        return None
    p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total
    advance_home = p_home + 0.5 * p_draw
    advance_away = p_away + 0.5 * p_draw
    if advance_home >= advance_away:
        advancing_team, eliminated_team, advance_prob = home, away, advance_home
    else:
        advancing_team, eliminated_team, advance_prob = away, home, advance_away
    return {
        "predicted_winner": advancing_team,
        "predicted_loser": eliminated_team,
        "winner_win_prob": round(float(advance_prob), 3),
        "prob_home_win": float(p_home),
        "prob_draw": float(p_draw),
        "prob_away_win": float(p_away),
        "expected_home_goals": float(home_xg),
        "expected_away_goals": float(away_xg),
        "probability_source": "xgboost_known_fixture",
    }


def pick_winner(
    home: str, away: str, country: str, strength: dict[str, float]
) -> tuple[str, str, float]:
    avg = float(np.mean(list(strength.values())))
    h = float(strength.get(home, avg))
    a = float(strength.get(away, avg))
    country_key = normalize_name(country)
    if normalize_name(home) == country_key:
        h += 55.0
    if normalize_name(away) == country_key:
        a += 55.0
    p_home = sigmoid((h - a) / 230.0)
    if p_home >= 0.5:
        return home, away, round(p_home, 3)
    return away, home, round(1.0 - p_home, 3)


def build_bracket(
    group_order: dict[str, list[str]],
    third_order: list[tuple[str, str, dict]],
    knockout: pd.DataFrame,
    strength: dict[str, float],
    fixture_predictions: pd.DataFrame | None = None,
) -> list[dict]:
    used_third_groups: set[str] = set()
    winners: dict[int, str] = {}
    losers: dict[int, str] = {}
    rows = []
    fixtures_by_match = {}
    if fixture_predictions is not None and "match_number" in fixture_predictions.columns:
        fixtures_by_match = {
            int(row["match_number"]): row
            for _, row in fixture_predictions.dropna(subset=["match_number"]).iterrows()
        }

    for row in knockout.sort_values("match_number").itertuples(index=False):
        match_number = int(row.match_number)
        home_label = str(row.home_team)
        away_label = str(row.away_team)

        # Corrigeer zelf-referentie in wedstrijd 100
        if match_number == 100 and away_label == "W100":
            away_label = "W96"

        home = resolve_slot(home_label, group_order, third_order, used_third_groups, winners, losers)
        away = resolve_slot(away_label, group_order, third_order, used_third_groups, winners, losers)
        known_prediction = known_fixture_prediction(
            fixtures_by_match.get(match_number),
            home,
            away,
            str(row.stage),
        )
        if known_prediction:
            winner = str(known_prediction["predicted_winner"])
            loser = str(known_prediction["predicted_loser"])
            p_win = float(known_prediction["winner_win_prob"])
            probability_fields = known_prediction
        else:
            winner, loser, p_win = pick_winner(home, away, str(row.country), strength)
            probability_fields = {
                "prob_home_win": p_win if winner == home else 1.0 - p_win,
                "prob_draw": 0.0,
                "prob_away_win": p_win if winner == away else 1.0 - p_win,
                "expected_home_goals": np.nan,
                "expected_away_goals": np.nan,
                "probability_source": "bracket_strength_fallback",
            }
        winners[match_number] = winner
        losers[match_number] = loser

        rows.append({
            "match_number": match_number,
            "stage": row.stage,
            "home_team": home,
            "away_team": away,
            "predicted_winner": winner,
            "predicted_loser": loser,
            "advancing_team": winner,
            "winner_win_prob": p_win,
            **probability_fields,
            "fixture_confirmed": not is_placeholder_slot(home_label) and not is_placeholder_slot(away_label),
            "city": row.city,
            "country": row.country,
        })

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_group_stage(group_order: dict[str, list[str]], third_order: list[tuple]) -> None:
    print(f"\n{'=' * 65}")
    print("  GROEPSFASE - VERWACHTE EINDSTAND")
    print(f"{'=' * 65}")
    for group in sorted(group_order.keys()):
        teams = group_order[group]
        print(f"\n  Groep {group}:")
        for i, team in enumerate(teams, 1):
            if i <= 2:
                tag = "DOOR"
            elif i == 3:
                tag = "mogelijk door (3e)"
            else:
                tag = "uitgeschakeld"
            print(f"    {i}. {team:<25s}  {tag}")

    qualifying_thirds = [t for _, t, _ in third_order]
    print(f"\n  Beste 8 derde-plaatsteams die doorstromen:")
    for i, (group, team, stats) in enumerate(third_order, 1):
        print(f"    {i}. {team:<25s}  (groep {group} | {stats['points']:.1f} ptn | GV {stats['gd']:+.2f})")


def print_bracket(bracket: list[dict]) -> None:
    stage_order = [
        "Round of 32",
        "Round of 16",
        "Quarterfinals",
        "Semifinals",
        "Third Place Playoff",
        "Final",
    ]
    stage_names = {
        "Round of 32": "RONDE VAN 32",
        "Round of 16": "RONDE VAN 16",
        "Quarterfinals": "KWARTFINALES",
        "Semifinals": "HALVE FINALES",
        "Third Place Playoff": "DERDE PLAATSWEDSTRIJD",
        "Final": "FINALE",
    }
    for stage in stage_order:
        stage_rows = [r for r in bracket if r["stage"] == stage]
        if not stage_rows:
            continue
        print(f"\n{'=' * 65}")
        print(f"  {stage_names.get(stage, stage)}")
        print(f"{'=' * 65}")
        for r in stage_rows:
            pct = int(r["winner_win_prob"] * 100)
            if r["predicted_winner"] == r["home_team"]:
                home_mark, away_mark = ">>", "  "
            else:
                home_mark, away_mark = "  ", ">>"
            print(
                f"  #{r['match_number']:3d}  "
                f"{home_mark} {r['home_team']:<22s} vs "
                f"{away_mark} {r['away_team']:<22s}  "
                f"Winnaar: {r['predicted_winner']} ({pct}%)"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="WC 2026 bracket voorspelling + gecombineerde CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--schedule", type=Path, default=SCHEDULE)
    parser.add_argument("--model-predictions", type=Path, default=MODEL_PREDS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--odds-weight", type=float, default=ODDS_WEIGHT)
    args = parser.parse_args()

    print("Data laden...")
    fixtures = merge_schedule_predictions(args.schedule, args.model_predictions, args.odds_weight, args.results)
    group_fixtures = fixtures[fixtures["stage"].eq("Group Stage")].copy()
    fifa_points = load_latest_fifa_points(RANKINGS)
    strength = build_team_strength(group_fixtures, fifa_points)

    group_tables = pd.read_csv(args.output_dir / "worldcup2026_expected_group_tables.csv")
    schedule_df = pd.read_csv(args.schedule)
    knockout = schedule_df[schedule_df["stage"] != "Group Stage"].copy()

    # 1. Combined team summary CSV
    print("\nGecombineerde team-samenvatting aanmaken...")
    team_summary = combine_team_summary(args.output_dir)
    out_summary = args.output_dir / "worldcup2026_combined_team_summary.csv"
    team_summary.to_csv(out_summary, index=False)
    print(f"  Opgeslagen: {out_summary}")

    # 2. Bracket prediction
    print("Bracket voorspelling aanmaken...")
    group_order = determine_group_order(group_tables)
    third_order = determine_third_order(group_tables, group_order)
    bracket = build_bracket(
        group_order,
        third_order,
        knockout,
        strength,
        fixtures[fixtures["stage"].ne("Group Stage")].copy(),
    )

    bracket_df = pd.DataFrame(bracket)
    out_bracket = args.output_dir / "worldcup2026_bracket_prediction.csv"
    bracket_df.to_csv(out_bracket, index=False)
    print(f"  Opgeslagen: {out_bracket}")

    # Print resultaten
    print_group_stage(group_order, third_order)
    print_bracket(bracket)

    # Finale samenvatting
    final_row = next((r for r in bracket if r["stage"] == "Final"), None)
    sf_rows = [r for r in bracket if r["stage"] == "Semifinals"]
    if final_row:
        print(f"\n{'=' * 65}")
        print("  VOORSPELD EINDRESULTAAT")
        print(f"{'=' * 65}")
        if sf_rows:
            print(f"  Halve finales:")
            for r in sf_rows:
                print(f"    {r['home_team']} vs {r['away_team']}  =>  {r['predicted_winner']}")
        print(f"\n  FINALE:")
        print(f"    {final_row['home_team']} vs {final_row['away_team']}")
        print(f"  WERELDKAMPIOEN: *** {final_row['predicted_winner']} ***")
        print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
