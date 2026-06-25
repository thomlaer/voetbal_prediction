"""Predict and simulate the 2026 World Cup from the prepared fixture schedule.

This script assumes ``train_xgboost_worldcup.py`` has already produced
``outputs_worldcup2026_default/future_predictions_xgboost.csv``. It keeps the
trained model untouched and adds a Monte Carlo layer for the tournament.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_xgboost_worldcup import normalize_name


DEFAULT_SCHEDULE = Path("data/extracted/oddsportal_worldcup2026_fixture_odds_schedule.csv")
DEFAULT_MODEL_PREDICTIONS = Path("outputs_worldcup2026_default/future_predictions_xgboost.csv")
DEFAULT_RANKINGS = Path("fifa_ranking-2026-04-01.csv")
DEFAULT_RESULTS = Path("data/results.csv")
DEFAULT_CARDS = Path("data/extracted/soccerbase_cards_events.csv")
DEFAULT_ESPN_RESULTS = Path("data/extracted/espn_worldcup2026_results.csv")
DEFAULT_OUTPUT_DIR = Path("outputs_worldcup2026_default")
OUTCOMES = np.array(["away_win", "draw", "home_win"], dtype=object)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def implied_probs(frame: pd.DataFrame) -> pd.DataFrame:
    implied = 1.0 / frame[["home_odds", "draw_odds", "away_odds"]].astype(float)
    total = implied.sum(axis=1)
    return pd.DataFrame(
        {
            "bookie_prob_home_win": implied["home_odds"] / total,
            "bookie_prob_draw": implied["draw_odds"] / total,
            "bookie_prob_away_win": implied["away_odds"] / total,
        },
        index=frame.index,
    )


def load_latest_fifa_points(path: Path) -> dict[str, float]:
    rankings = pd.read_csv(path)
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"], errors="coerce")
    rankings["team_key"] = rankings["country_full"].map(normalize_name)
    rankings["total_points"] = pd.to_numeric(rankings["total_points"], errors="coerce")
    rankings = rankings.dropna(subset=["rank_date", "team_key", "total_points"]).sort_values("rank_date")
    latest = rankings.drop_duplicates("team_key", keep="last")
    return dict(zip(latest["team_key"], latest["total_points"]))


def outcome_from_score(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def load_actual_results(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    results = pd.read_csv(path)
    required = {"date", "home_team", "away_team", "home_score", "away_score", "tournament"}
    if not required.issubset(results.columns):
        return {}
    results = results[results["tournament"].eq("FIFA World Cup")].copy()
    results["date"] = pd.to_datetime(results["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce")
    results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce")
    results = results.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])

    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in results.itertuples(index=False):
        home_score = int(row.home_score)
        away_score = int(row.away_score)
        outcome = outcome_from_score(home_score, away_score)
        direct = {
            "actual_available": True,
            "actual_home_score": home_score,
            "actual_away_score": away_score,
            "actual_score": f"{home_score}-{away_score}",
            "actual_outcome": outcome,
            "actual_winner": row.home_team if outcome == "home_win" else row.away_team if outcome == "away_win" else "Draw",
        }
        reverse_outcome = outcome_from_score(away_score, home_score)
        reverse = {
            "actual_available": True,
            "actual_home_score": away_score,
            "actual_away_score": home_score,
            "actual_score": f"{away_score}-{home_score}",
            "actual_outcome": reverse_outcome,
            "actual_winner": row.away_team if reverse_outcome == "home_win" else row.home_team if reverse_outcome == "away_win" else "Draw",
        }
        date_key = str(row.date)
        lookup[(date_key, normalize_name(row.home_team), normalize_name(row.away_team))] = direct
        lookup[(date_key, normalize_name(row.away_team), normalize_name(row.home_team))] = reverse
    return lookup


def load_espn_actual_results(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    results = pd.read_csv(path)
    required = {"date", "home_team", "away_team", "home_score", "away_score", "completed"}
    if not required.issubset(results.columns):
        return {}
    results = results[results["completed"].astype(str).str.lower().isin(["true", "1"])].copy()
    results["date"] = pd.to_datetime(results["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce")
    results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce")
    results = results.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])

    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in results.itertuples(index=False):
        home_score = int(row.home_score)
        away_score = int(row.away_score)
        outcome = outcome_from_score(home_score, away_score)
        direct = {
            "actual_available": True,
            "actual_home_score": home_score,
            "actual_away_score": away_score,
            "actual_score": f"{home_score}-{away_score}",
            "actual_outcome": outcome,
            "actual_winner": row.home_team if outcome == "home_win" else row.away_team if outcome == "away_win" else "Draw",
        }
        reverse_outcome = outcome_from_score(away_score, home_score)
        reverse = {
            "actual_available": True,
            "actual_home_score": away_score,
            "actual_away_score": home_score,
            "actual_score": f"{away_score}-{home_score}",
            "actual_outcome": reverse_outcome,
            "actual_winner": row.away_team if reverse_outcome == "home_win" else row.home_team if reverse_outcome == "away_win" else "Draw",
        }
        date_keys = [str(row.date)]
        parsed_date = pd.to_datetime(row.date, errors="coerce")
        if pd.notna(parsed_date):
            date_keys.append((parsed_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        for date_key in date_keys:
            lookup[(date_key, normalize_name(row.home_team), normalize_name(row.away_team))] = direct
            lookup[(date_key, normalize_name(row.away_team), normalize_name(row.home_team))] = reverse
    return lookup


def apply_actual_results(fixtures: pd.DataFrame, results_path: Path, espn_results_path: Path) -> pd.DataFrame:
    fixtures = fixtures.copy()
    fixtures["actual_available"] = False
    fixtures["actual_home_score"] = np.nan
    fixtures["actual_away_score"] = np.nan
    for column in ["actual_score", "actual_outcome", "actual_winner"]:
        fixtures[column] = pd.Series([""] * len(fixtures), index=fixtures.index, dtype="object")

    lookup = load_actual_results(results_path)
    lookup.update(load_espn_actual_results(espn_results_path))
    if not lookup:
        return fixtures

    for idx, row in fixtures.iterrows():
        date_key = pd.to_datetime(row["date"], errors="coerce")
        if pd.isna(date_key):
            continue
        key = (
            date_key.strftime("%Y-%m-%d"),
            normalize_name(row["home_team"]),
            normalize_name(row["away_team"]),
        )
        actual = lookup.get(key)
        if not actual:
            continue
        for column, value in actual.items():
            fixtures.at[idx, column] = value
        fixtures.at[idx, "expected_home_goals"] = actual["actual_home_score"]
        fixtures.at[idx, "expected_away_goals"] = actual["actual_away_score"]
        fixtures.at[idx, "sim_prob_home_win"] = 1.0 if actual["actual_outcome"] == "home_win" else 0.0
        fixtures.at[idx, "sim_prob_draw"] = 1.0 if actual["actual_outcome"] == "draw" else 0.0
        fixtures.at[idx, "sim_prob_away_win"] = 1.0 if actual["actual_outcome"] == "away_win" else 0.0
        fixtures.at[idx, "sim_predicted_outcome"] = actual["actual_outcome"]
    return fixtures


def card_conduct_points(card_type: Any) -> int:
    text = str(card_type or "").strip().lower()
    if "red" in text:
        return -4
    if "yellow" in text:
        return -1
    return 0


def load_fair_play_points(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    cards = pd.read_csv(path)
    required = {"date", "competition", "stage", "team", "card_type"}
    if not required.issubset(cards.columns):
        return {}
    cards = cards[
        cards["date"].astype(str).str.startswith("2026-")
        & cards["competition"].astype(str).str.contains("world cup", case=False, na=False)
        & cards["stage"].astype(str).str.contains("group", case=False, na=False)
    ].copy()
    if cards.empty:
        return {}
    cards["team_key"] = cards["team"].map(normalize_name)
    cards["conduct_points"] = cards["card_type"].map(card_conduct_points)
    return cards.groupby("team_key")["conduct_points"].sum().astype(int).to_dict()


def merge_schedule_predictions(
    schedule_path: Path,
    predictions_path: Path,
    odds_weight: float,
    results_path: Path,
    espn_results_path: Path = DEFAULT_ESPN_RESULTS,
) -> pd.DataFrame:
    schedule = pd.read_csv(schedule_path)
    predictions = pd.read_csv(predictions_path)
    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    merge_keys = ["date", "home_team", "away_team", "tournament"]
    if predictions.duplicated(merge_keys).any():
        predictions = predictions.drop_duplicates(merge_keys, keep="last")

    merged = schedule.merge(
        predictions,
        on=merge_keys,
        how="left",
        suffixes=("", "_model"),
    )
    missing = merged["prob_home_win"].isna().sum()
    if missing:
        raise ValueError(f"Could not match model predictions for {missing} fixture rows.")

    bookie = implied_probs(merged)
    merged = pd.concat([merged, bookie], axis=1)
    has_odds = merged["odds_available"].fillna(0).astype(int).eq(1)

    model_home = merged["prob_home_win"].astype(float)
    model_draw = merged["prob_draw"].astype(float)
    model_away = merged["prob_away_win"].astype(float)
    merged["sim_prob_home_win"] = model_home
    merged["sim_prob_draw"] = model_draw
    merged["sim_prob_away_win"] = model_away
    for outcome in ("home_win", "draw", "away_win"):
        merged.loc[has_odds, f"sim_prob_{outcome}"] = (
            (1.0 - odds_weight) * merged.loc[has_odds, f"prob_{outcome}"]
            + odds_weight * merged.loc[has_odds, f"bookie_prob_{outcome}"]
        )
    sim_sum = merged[["sim_prob_away_win", "sim_prob_draw", "sim_prob_home_win"]].sum(axis=1)
    for outcome in ("home_win", "draw", "away_win"):
        merged[f"sim_prob_{outcome}"] = merged[f"sim_prob_{outcome}"] / sim_sum

    sim_probs = merged[["sim_prob_away_win", "sim_prob_draw", "sim_prob_home_win"]].to_numpy()
    merged["sim_predicted_outcome"] = OUTCOMES[sim_probs.argmax(axis=1)]
    merged = apply_actual_results(merged, results_path, espn_results_path)
    return merged


def score_from_outcome(
    rng: np.random.Generator,
    expected_home: float,
    expected_away: float,
    outcome_id: int,
) -> tuple[int, int]:
    home_lambda = max(float(expected_home), 0.05)
    away_lambda = max(float(expected_away), 0.05)
    for _ in range(80):
        home = int(min(rng.poisson(home_lambda), 8))
        away = int(min(rng.poisson(away_lambda), 8))
        if outcome_id == 2 and home > away:
            return home, away
        if outcome_id == 1 and home == away:
            return home, away
        if outcome_id == 0 and home < away:
            return home, away

    if outcome_id == 2:
        return 2, 1 if away_lambda > 0.75 else 0
    if outcome_id == 0:
        return 1 if home_lambda > 0.75 else 0, 2
    draw_goal = int(np.clip(round((home_lambda + away_lambda) / 2.0), 0, 3))
    return draw_goal, draw_goal


def deterministic_score_for_outcome(row: pd.Series) -> tuple[str, str]:
    if bool(row.get("actual_available", False)):
        return str(row.get("actual_score", "")), "actual_result"

    pool_score = str(row.get("pool_predicted_score", ""))
    match = re.fullmatch(r"(\d+)-(\d+)", pool_score)
    if match:
        home_goals = int(match.group(1))
        away_goals = int(match.group(2))
        if row["sim_predicted_outcome"] == "home_win" and home_goals > away_goals:
            return pool_score, "model_pool_score"
        if row["sim_predicted_outcome"] == "away_win" and home_goals < away_goals:
            return pool_score, "model_pool_score"
        if row["sim_predicted_outcome"] == "draw" and home_goals == away_goals:
            return pool_score, "model_pool_score"

    total_goals = float(row["expected_home_goals"]) + float(row["expected_away_goals"])
    favorite_gap = abs(float(row["sim_prob_home_win"]) - float(row["sim_prob_away_win"]))
    if row["sim_predicted_outcome"] == "draw":
        return ("0-0" if total_goals < 1.75 else "1-1"), "adjusted_to_sim_outcome"
    if row["sim_predicted_outcome"] == "home_win":
        if favorite_gap < 0.18:
            return "2-1", "adjusted_to_sim_outcome"
        return ("3-0" if float(row["sim_prob_home_win"]) >= 0.78 else "2-0"), "adjusted_to_sim_outcome"
    if favorite_gap < 0.18:
        return "1-2", "adjusted_to_sim_outcome"
    return ("0-3" if float(row["sim_prob_away_win"]) >= 0.78 else "0-2"), "adjusted_to_sim_outcome"


def init_table(teams: set[str]) -> dict[str, dict[str, float]]:
    return {
        team: {"played": 0, "points": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "gd": 0}
        for team in teams
    }


def add_match(table: dict[str, dict[str, float]], home: str, away: str, home_goals: int, away_goals: int) -> None:
    table[home]["played"] += 1
    table[away]["played"] += 1
    table[home]["gf"] += home_goals
    table[home]["ga"] += away_goals
    table[away]["gf"] += away_goals
    table[away]["ga"] += home_goals
    table[home]["gd"] = table[home]["gf"] - table[home]["ga"]
    table[away]["gd"] = table[away]["gf"] - table[away]["ga"]
    if home_goals > away_goals:
        table[home]["points"] += 3
        table[home]["wins"] += 1
        table[away]["losses"] += 1
    elif home_goals < away_goals:
        table[away]["points"] += 3
        table[away]["wins"] += 1
        table[home]["losses"] += 1
    else:
        table[home]["points"] += 1
        table[away]["points"] += 1
        table[home]["draws"] += 1
        table[away]["draws"] += 1


def head_to_head_stats(
    teams: set[str],
    matches: list[tuple[str, str, int, int]],
) -> dict[str, dict[str, int]]:
    stats = {
        team: {"h2h_points": 0, "h2h_gd": 0, "h2h_gf": 0}
        for team in teams
    }
    for home, away, home_goals, away_goals in matches:
        if home not in teams or away not in teams:
            continue
        stats[home]["h2h_gf"] += home_goals
        stats[away]["h2h_gf"] += away_goals
        stats[home]["h2h_gd"] += home_goals - away_goals
        stats[away]["h2h_gd"] += away_goals - home_goals
        if home_goals > away_goals:
            stats[home]["h2h_points"] += 3
        elif away_goals > home_goals:
            stats[away]["h2h_points"] += 3
        else:
            stats[home]["h2h_points"] += 1
            stats[away]["h2h_points"] += 1
    return stats


def rank_group(
    table: dict[str, dict[str, float]],
    rng: np.random.Generator,
    matches: list[tuple[str, str, int, int]] | None = None,
    fair_play_points: dict[str, int] | None = None,
) -> list[str]:
    ranked: list[str] = []
    fair_play_points = fair_play_points or {}
    point_buckets: dict[int, list[str]] = {}
    for team, stats in table.items():
        point_buckets.setdefault(int(stats["points"]), []).append(team)

    for points in sorted(point_buckets, reverse=True):
        tied = point_buckets[points]
        if len(tied) == 1:
            ranked.extend(tied)
            continue

        h2h = head_to_head_stats(set(tied), matches or [])
        tied.sort(
            key=lambda team: (
                h2h[team]["h2h_points"],
                h2h[team]["h2h_gd"],
                h2h[team]["h2h_gf"],
                table[team]["gd"],
                table[team]["gf"],
                fair_play_points.get(normalize_name(team), 0),
                rng.random(),
            ),
            reverse=True,
        )
        ranked.extend(tied)
    return ranked


def build_team_strength(
    group_fixtures: pd.DataFrame,
    fifa_points: dict[str, float],
) -> dict[str, float]:
    teams = sorted(set(group_fixtures["home_team"]).union(group_fixtures["away_team"]))
    expected = {team: {"points": 0.0, "gd": 0.0, "gf": 0.0, "matches": 0.0} for team in teams}
    for row in group_fixtures.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        p_home = float(row.sim_prob_home_win)
        p_draw = float(row.sim_prob_draw)
        p_away = float(row.sim_prob_away_win)
        expected[home]["points"] += 3.0 * p_home + p_draw
        expected[away]["points"] += 3.0 * p_away + p_draw
        expected[home]["gd"] += float(row.expected_home_goals) - float(row.expected_away_goals)
        expected[away]["gd"] += float(row.expected_away_goals) - float(row.expected_home_goals)
        expected[home]["gf"] += float(row.expected_home_goals)
        expected[away]["gf"] += float(row.expected_away_goals)
        expected[home]["matches"] += 1.0
        expected[away]["matches"] += 1.0

    known_points = [value for value in fifa_points.values() if pd.notna(value)]
    median_points = float(np.median(known_points)) if known_points else 1400.0
    strength = {}
    for team in teams:
        key = normalize_name(team)
        points = float(fifa_points.get(key, median_points))
        played = max(expected[team]["matches"], 1.0)
        # Keep FIFA points as the base scale, then let the model/odds group outlook
        # nudge teams by expected points and goal difference.
        strength[team] = (
            points
            + 55.0 * (expected[team]["points"] / played - 1.5)
            + 35.0 * (expected[team]["gd"] / played)
        )
    return strength


def knockout_advance_probability(home: str, away: str, row: pd.Series, strength: dict[str, float]) -> float:
    home_rating = float(strength.get(home, np.mean(list(strength.values()))))
    away_rating = float(strength.get(away, np.mean(list(strength.values()))))
    country_key = normalize_name(row["country"])
    if normalize_name(home) == country_key:
        home_rating += 55.0
    if normalize_name(away) == country_key:
        away_rating += 55.0
    return sigmoid((home_rating - away_rating) / 230.0)


def resolve_slot(
    label: str,
    group_order: dict[str, list[str]],
    third_order: list[tuple[str, str, dict[str, float]]],
    used_third_groups: set[str],
    winners: dict[int, str],
    losers: dict[int, str],
) -> str:
    if re.fullmatch(r"[12][A-L]", label):
        position = int(label[0]) - 1
        group = label[1]
        return group_order[group][position]
    if re.fullmatch(r"3[A-L]+", label):
        eligible = set(label[1:])
        for group, team, _stats in third_order:
            if group in eligible and group not in used_third_groups:
                used_third_groups.add(group)
                return team
        for group, team, _stats in third_order:
            if group not in used_third_groups:
                used_third_groups.add(group)
                return team
        raise ValueError(f"Could not resolve third-place slot {label}")
    if label.startswith("W"):
        return winners[int(label[1:])]
    if label.startswith("RU"):
        return losers[int(label[2:])]
    if label and label.lower() != "nan":
        return label
    raise ValueError(f"Unsupported bracket label: {label}")


def simulate_tournament(
    fixtures: pd.DataFrame,
    strength: dict[str, float],
    simulations: int,
    seed: int,
    fair_play_points: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    group_fixtures = fixtures[fixtures["stage"].eq("Group Stage")].sort_values("match_number")
    knockout = fixtures[~fixtures["stage"].eq("Group Stage")].sort_values("match_number").copy()
    # The local skeleton has an obvious self-reference for match 100. Treat it as W95 vs W96.
    self_ref = knockout["match_number"].eq(100) & knockout["away_team"].eq("W100")
    knockout.loc[self_ref, "away_team"] = "W96"
    knockout.loc[self_ref, "placeholder_away_label"] = "W96"

    teams = sorted(set(group_fixtures["home_team"]).union(group_fixtures["away_team"]))
    counts = {
        team: {
            "group_winner": 0,
            "group_runner_up": 0,
            "third_place": 0,
            "advance_r32": 0,
            "advance_r16": 0,
            "advance_qf": 0,
            "advance_sf": 0,
            "advance_final": 0,
            "champion": 0,
        }
        for team in teams
    }
    expected_table = {
        team: {"points": 0.0, "gd": 0.0, "gf": 0.0, "ga": 0.0, "rank_sum": 0.0}
        for team in teams
    }
    champion_samples: list[str] = []
    final_pair_counts: defaultdict[tuple[str, str], int] = defaultdict(int)

    for _ in range(simulations):
        grouped_teams = {
            group: set(group_rows["home_team"]).union(group_rows["away_team"])
            for group, group_rows in group_fixtures.groupby("group")
        }
        tables = {group: init_table(team_set) for group, team_set in grouped_teams.items()}
        group_results = {group: [] for group in grouped_teams}
        for row in group_fixtures.itertuples(index=False):
            if bool(getattr(row, "actual_available", False)):
                home_goals = int(getattr(row, "actual_home_score"))
                away_goals = int(getattr(row, "actual_away_score"))
            else:
                probs = np.array([row.sim_prob_away_win, row.sim_prob_draw, row.sim_prob_home_win], dtype=float)
                probs = probs / probs.sum()
                outcome_id = int(rng.choice([0, 1, 2], p=probs))
                home_goals, away_goals = score_from_outcome(
                    rng,
                    row.expected_home_goals,
                    row.expected_away_goals,
                    outcome_id,
                )
            add_match(tables[row.group], row.home_team, row.away_team, home_goals, away_goals)
            group_results[row.group].append((row.home_team, row.away_team, home_goals, away_goals))

        group_order: dict[str, list[str]] = {}
        third_order: list[tuple[str, str, dict[str, float]]] = []
        for group, table in tables.items():
            order = rank_group(table, rng, group_results.get(group, []), fair_play_points)
            group_order[group] = order
            counts[order[0]]["group_winner"] += 1
            counts[order[1]]["group_runner_up"] += 1
            counts[order[2]]["third_place"] += 1
            third_order.append((group, order[2], table[order[2]]))
            for rank, team in enumerate(order, start=1):
                expected_table[team]["rank_sum"] += rank
                for key in ("points", "gd", "gf", "ga"):
                    expected_table[team][key] += table[team][key]

        third_order.sort(
            key=lambda item: (
                item[2]["points"],
                item[2]["gd"],
                item[2]["gf"],
                item[2]["wins"],
                rng.random(),
            ),
            reverse=True,
        )
        qualifying_thirds = third_order[:8]
        third_order = qualifying_thirds
        used_third_groups: set[str] = set()
        winners: dict[int, str] = {}
        losers: dict[int, str] = {}
        semifinal_losers: dict[int, str] = {}

        for row in knockout.itertuples(index=False):
            match_number = int(row.match_number)
            row_series = pd.Series(row._asdict())
            home_label = str(row.home_team)
            away_label = str(row.away_team)
            home = resolve_slot(home_label, group_order, third_order, used_third_groups, winners, losers)
            away = resolve_slot(away_label, group_order, third_order, used_third_groups, winners, losers)
            p_home = knockout_advance_probability(home, away, row_series, strength)
            if rng.random() < p_home:
                winner, loser = home, away
            else:
                winner, loser = away, home
            winners[match_number] = winner
            losers[match_number] = loser
            if row.stage == "Semifinals":
                semifinal_losers[match_number] = loser

            if row.stage == "Round of 32":
                counts[winner]["advance_r16"] += 1
            elif row.stage == "Round of 16":
                counts[winner]["advance_qf"] += 1
            elif row.stage == "Quarterfinals":
                counts[winner]["advance_sf"] += 1
            elif row.stage == "Semifinals":
                counts[winner]["advance_final"] += 1
            elif row.stage == "Final":
                final_pair = tuple(sorted((home, away)))
                final_pair_counts[final_pair] += 1
                counts[winner]["champion"] += 1
                champion_samples.append(winner)

        for team in group_order.values():
            for qualified in team[:2]:
                counts[qualified]["advance_r32"] += 1
        for _group, qualified, _stats in qualifying_thirds:
            counts[qualified]["advance_r32"] += 1

    rows = []
    for team in teams:
        row = {"team": team}
        row.update({f"{key}_prob": value / simulations for key, value in counts[team].items()})
        row["expected_group_points"] = expected_table[team]["points"] / simulations
        row["expected_group_gd"] = expected_table[team]["gd"] / simulations
        row["expected_group_gf"] = expected_table[team]["gf"] / simulations
        row["expected_group_ga"] = expected_table[team]["ga"] / simulations
        row["expected_group_rank"] = expected_table[team]["rank_sum"] / simulations
        rows.append(row)

    probabilities = pd.DataFrame(rows).sort_values("champion_prob", ascending=False).reset_index(drop=True)
    champion_sample = pd.Series(champion_samples, name="champion")
    champion_counts = champion_sample.value_counts().rename_axis("team").reset_index(name="champion_count")
    champion_counts["champion_prob"] = champion_counts["champion_count"] / simulations
    final_pairs = pd.DataFrame(
        [
            {
                "team_1": pair[0],
                "team_2": pair[1],
                "final_count": count,
                "final_pair_prob": count / simulations,
            }
            for pair, count in final_pair_counts.items()
        ]
    )
    if not final_pairs.empty:
        final_pairs = final_pairs.sort_values("final_pair_prob", ascending=False).reset_index(drop=True)
    return probabilities, champion_counts, final_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a World Cup 2026 Monte Carlo simulation.")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--model-predictions", type=Path, default=DEFAULT_MODEL_PREDICTIONS)
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    parser.add_argument("--espn-results", type=Path, default=DEFAULT_ESPN_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--simulations", type=int, default=50000)
    parser.add_argument("--odds-weight", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    fixtures = merge_schedule_predictions(
        args.schedule,
        args.model_predictions,
        args.odds_weight,
        args.results,
        args.espn_results,
    )
    group_fixtures = fixtures[fixtures["stage"].eq("Group Stage")].copy()
    fifa_points = load_latest_fifa_points(args.rankings)
    fair_play_points = load_fair_play_points(args.cards)
    strength = build_team_strength(group_fixtures, fifa_points)
    probabilities, champion_counts, final_pairs = simulate_tournament(
        fixtures,
        strength,
        args.simulations,
        args.seed,
        fair_play_points,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_output = group_fixtures[
        [
            "date",
            "match_number",
            "group",
            "home_team",
            "away_team",
            "city",
            "country",
            "home_odds",
            "draw_odds",
            "away_odds",
            "prob_home_win",
            "prob_draw",
            "prob_away_win",
            "bookie_prob_home_win",
            "bookie_prob_draw",
            "bookie_prob_away_win",
            "sim_prob_home_win",
            "sim_prob_draw",
            "sim_prob_away_win",
            "sim_predicted_outcome",
            "pool_predicted_score",
            "expected_home_goals",
            "expected_away_goals",
            "actual_available",
            "actual_home_score",
            "actual_away_score",
            "actual_score",
            "actual_outcome",
            "actual_winner",
        ]
    ].copy()
    sim_scores = group_output.apply(deterministic_score_for_outcome, axis=1, result_type="expand")
    group_output["sim_predicted_score"] = sim_scores[0]
    group_output["sim_score_source"] = sim_scores[1]
    group_output.to_csv(args.output_dir / "worldcup2026_group_match_predictions.csv", index=False)
    probabilities.to_csv(args.output_dir / "worldcup2026_montecarlo_team_probabilities.csv", index=False)
    champion_counts.to_csv(args.output_dir / "worldcup2026_montecarlo_champion_counts.csv", index=False)
    final_pairs.to_csv(args.output_dir / "worldcup2026_montecarlo_final_pairs.csv", index=False)
    metadata = {
        "schedule": str(args.schedule),
        "model_predictions": str(args.model_predictions),
        "rankings": str(args.rankings),
        "simulations": int(args.simulations),
        "odds_weight_for_group_stage": float(args.odds_weight),
        "group_stage_rows": int(len(group_fixtures)),
        "actual_group_results_locked": int(group_fixtures["actual_available"].fillna(False).sum()),
        "teams": int(probabilities["team"].nunique()),
        "knockout_note": (
            "Group stage uses model probabilities blended with current OddsPortal odds where available. "
            "Knockout matches have no market odds in the schedule, so advancement uses a FIFA-points "
            "strength model nudged by model/odds group outlook. Match 100 self-reference W100 was treated as W96."
        ),
    }
    (args.output_dir / "worldcup2026_montecarlo_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Top champion probabilities:")
    print(probabilities[["team", "champion_prob", "advance_final_prob", "advance_sf_prob"]].head(20).to_string(index=False))
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
