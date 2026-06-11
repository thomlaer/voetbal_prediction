"""
Train an XGBoost 1X2 model for international football.

The script downloads the martj42 international_results CSV files if they are
missing, builds pre-match features without future leakage, evaluates a
time-based holdout set, and saves the trained model plus prediction outputs.

Optional odds support expects a CSV with match/date/team columns and decimal
1X2 odds. Keep API keys in environment variables or a local .env file; do not
hard-code them in this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "joblib": "joblib",
}


def require_dependencies() -> None:
    missing = []
    for import_name, package_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        packages = " ".join(sorted(set(missing)))
        raise SystemExit(
            "Missing Python packages: "
            f"{packages}\n\nInstall them in your environment with:\n"
            f"  python -m pip install {packages}\n"
        )


require_dependencies()

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor


GITHUB_RAW_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
DATA_FILES = ("results.csv", "shootouts.csv", "goalscorers.csv", "former_names.csv")
OUTCOME_TO_ID = {"away_win": 0, "draw": 1, "home_win": 2}
ID_TO_OUTCOME = {value: key for key, value in OUTCOME_TO_ID.items()}


TEAM_ALIASES = {
    "cote d ivoire": "ivory coast",
    "cote divoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "cabo verde": "cape verde",
    "czechia": "czech republic",
    "cape verde islands": "cape verde",
    "dr congo": "congo dr",
    "congo dr": "congo dr",
    "dutch east indies": "indonesia",
    "east germany": "german dr",
    "german democratic republic": "german dr",
    "germany dr": "german dr",
    "ir iran": "iran",
    "korea republic": "south korea",
    "korea dpr": "north korea",
    "north macedonia": "macedonia",
    "people's republic of china": "china",
    "china pr": "china",
    "republic of ireland": "ireland",
    "russia": "russia",
    "serbia and montenegro": "serbia",
    "soviet union": "russia",
    "ussr": "russia",
    "turkiye": "turkey",
    "united states": "united states",
    "usa": "united states",
    "viet nam": "vietnam",
    "west germany": "germany",
    "zaire": "congo dr",
}


DEFAULT_TRANSFERMARKT_DIR = Path("data/kagglehub/datasets/davidcariboo/player-scores/versions/655")
DEFAULT_FJELSTUL_DIR = Path("data/fjelstul_worldcup/data-csv")
DEFAULT_XFKZ_SNAPSHOTS = Path("data/extracted/xfkz_country_market_injury_snapshots.csv")
DEFAULT_GEO_CITY_LOCATIONS = Path("data/extracted/worldcities_city_locations.csv")
DEFAULT_GEO_COUNTRY_REPS = Path("data/extracted/worldcities_country_representatives.csv")
DEFAULT_GOALSCORERS = Path("data/goalscorers.csv")
DEFAULT_TOURNAMENT_SQUADS = Path("data/extracted/wikipedia_tournament_squads.csv")
DEFAULT_SOCCERBASE_LINEUPS = Path("data/extracted/soccerbase_lineups_used.csv")
DEFAULT_SOCCERBASE_MATCH_STATS = Path("data/extracted/soccerbase_match_stats.csv")
DEFAULT_SOCCERBASE_CARDS = Path("data/extracted/soccerbase_cards_events.csv")
DEFAULT_SOFIFA_YEARLY_RATINGS = Path("data/extracted/sofifa_yearly_player_ratings.csv")
DEFAULT_EXTERNAL_ELO = Path(
    "data/kagglehub/datasets/saifalnimri/international-football-elo-ratings/versions/1/eloratings.csv"
)
POSITION_GROUPS = ("attack", "midfield", "defender", "goalkeeper")


@dataclass
class TeamState:
    elo: float = 1500.0
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    home_matches: int = 0
    home_wins: int = 0
    home_goals_for: int = 0
    home_goals_against: int = 0
    away_matches: int = 0
    away_wins: int = 0
    away_goals_for: int = 0
    away_goals_against: int = 0
    last_date: pd.Timestamp | None = None
    recent: deque[dict[str, float]] = field(default_factory=deque)


@dataclass
class TournamentTeamState:
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    last_date: pd.Timestamp | None = None


def normalize_name(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    for char in (".", ",", "-", "_", "(", ")", "&"):
        text = text.replace(char, " ")
    text = " ".join(text.replace("'", "").split())
    return TEAM_ALIASES.get(text, text)


def normalize_person_name(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    for char in (".", ",", "-", "_", "(", ")", "&"):
        text = text.replace(char, " ")
    return " ".join(text.replace("'", "").split())


def person_name_match_score(query_key: str, candidate_key: str) -> float:
    if not query_key or not candidate_key:
        return 0.0
    if query_key == candidate_key:
        return 1.0
    query_tokens = [token for token in query_key.split() if token]
    candidate_tokens = [token for token in candidate_key.split() if token]
    if not query_tokens or not candidate_tokens:
        return 0.0
    query_set = set(query_tokens)
    candidate_set = set(candidate_tokens)
    if query_set.issubset(candidate_set) or candidate_set.issubset(query_set):
        return 0.95
    overlap = query_set & candidate_set
    if not overlap:
        return 0.0
    score = len(overlap) / max(len(query_set), len(candidate_set))
    if query_tokens[-1] in candidate_set or candidate_tokens[-1] in query_set:
        score += 0.20
    if query_tokens[0][0] == candidate_tokens[0][0]:
        score += 0.10
    return min(score, 0.94)


def normalize_fjelstul_team_name(value: Any, date_value: Any) -> str:
    key = normalize_name(value)
    if key == "yugoslavia":
        date = pd.to_datetime(date_value, errors="coerce")
        if pd.notna(date) and date >= pd.Timestamp("1992-04-27"):
            return "serbia"
    return key


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_rate(num: float, den: float, default: float = 0.0) -> float:
    return default if den == 0 else num / den


def cap_rest_days(days: float) -> float:
    if pd.isna(days):
        return 365.0
    return float(min(max(days, 0), 365))


def haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> float:
    values = [lat1, lon1, lat2, lon2]
    if any(pd.isna(value) for value in values):
        return np.nan
    radius_km = 6371.0088
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return radius_km * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def longitude_to_utc_offset(lng: Any) -> float:
    if pd.isna(lng):
        return np.nan
    return float(np.clip(round(float(lng) / 15.0), -12, 14))


def elo_expected(home_elo: float, away_elo: float, neutral: bool, home_advantage: float = 65.0) -> float:
    adjusted_home = home_elo + (0.0 if neutral else home_advantage)
    return 1.0 / (1.0 + 10.0 ** ((away_elo - adjusted_home) / 400.0))


def tournament_weight(tournament: str) -> float:
    name = str(tournament).lower()
    if "world cup" in name:
        return 36.0
    if any(token in name for token in ("euro", "copa america", "african cup", "asian cup", "gold cup")):
        return 30.0
    if "nations league" in name or "qualif" in name:
        return 24.0
    if "friendly" in name:
        return 12.0
    return 18.0


def goal_multiplier(goal_diff: int) -> float:
    diff = abs(goal_diff)
    if diff <= 1:
        return 1.0
    return min(1.0 + math.log(diff), 2.75)


def recent_average(state: TeamState, key: str) -> float:
    if not state.recent:
        return 0.0
    return float(sum(item[key] for item in state.recent) / len(state.recent))


def recent_average_last(state: TeamState, key: str, count: int) -> float:
    if not state.recent:
        return 0.0
    recent_items = list(state.recent)[-count:]
    return float(sum(item[key] for item in recent_items) / len(recent_items))


def recent_count(state: TeamState, count: int) -> float:
    return float(min(len(state.recent), count))


def state_features(prefix: str, state: TeamState, match_date: pd.Timestamp, side: str) -> dict[str, float]:
    rest_days = 365.0 if state.last_date is None else (match_date - state.last_date).days
    side_matches = state.home_matches if side == "home" else state.away_matches
    side_wins = state.home_wins if side == "home" else state.away_wins
    side_gf = state.home_goals_for if side == "home" else state.away_goals_for
    side_ga = state.home_goals_against if side == "home" else state.away_goals_against

    return {
        f"{prefix}_elo": state.elo,
        f"{prefix}_matches": float(state.matches),
        f"{prefix}_log_matches": float(np.log1p(state.matches)),
        f"{prefix}_win_rate": safe_rate(state.wins, state.matches),
        f"{prefix}_draw_rate": safe_rate(state.draws, state.matches),
        f"{prefix}_points_per_match": safe_rate(state.points, state.matches),
        f"{prefix}_goals_for_per_match": safe_rate(state.goals_for, state.matches),
        f"{prefix}_goals_against_per_match": safe_rate(state.goals_against, state.matches),
        f"{prefix}_goal_diff_per_match": safe_rate(state.goals_for - state.goals_against, state.matches),
        f"{prefix}_{side}_matches": float(side_matches),
        f"{prefix}_{side}_win_rate": safe_rate(side_wins, side_matches),
        f"{prefix}_{side}_goals_for_per_match": safe_rate(side_gf, side_matches),
        f"{prefix}_{side}_goals_against_per_match": safe_rate(side_ga, side_matches),
        f"{prefix}_recent_points_per_match": recent_average(state, "points"),
        f"{prefix}_recent_goal_diff": recent_average(state, "goal_diff"),
        f"{prefix}_recent_goals_for": recent_average(state, "goals_for"),
        f"{prefix}_recent_goals_against": recent_average(state, "goals_against"),
        f"{prefix}_recent_win": recent_average(state, "win"),
        f"{prefix}_recent_draw": recent_average(state, "draw"),
        f"{prefix}_recent_loss": recent_average(state, "loss"),
        f"{prefix}_recent5_matches": recent_count(state, 5),
        f"{prefix}_recent5_points_per_match": recent_average_last(state, "points", 5),
        f"{prefix}_recent5_goal_diff": recent_average_last(state, "goal_diff", 5),
        f"{prefix}_recent5_goals_for": recent_average_last(state, "goals_for", 5),
        f"{prefix}_recent5_goals_against": recent_average_last(state, "goals_against", 5),
        f"{prefix}_recent5_win": recent_average_last(state, "win", 5),
        f"{prefix}_recent5_draw": recent_average_last(state, "draw", 5),
        f"{prefix}_recent5_loss": recent_average_last(state, "loss", 5),
        f"{prefix}_rest_days": cap_rest_days(rest_days),
    }


def tournament_key(row: pd.Series) -> str:
    # Keep tournament-state local to a season/year so old editions do not leak into a new tournament.
    return f"{row['tournament']}|{int(row['date'].year)}"


def tournament_state_features(prefix: str, state: TournamentTeamState, match_date: pd.Timestamp) -> dict[str, float]:
    rest_days = 365.0 if state.last_date is None else (match_date - state.last_date).days
    goal_diff = state.goals_for - state.goals_against
    max_group_points_after_three = 9.0
    points_needed_for_typical_qualification = max(0.0, 4.0 - float(state.points))
    return {
        f"{prefix}_tournament_matches": float(state.matches),
        f"{prefix}_tournament_points": float(state.points),
        f"{prefix}_tournament_wins": float(state.wins),
        f"{prefix}_tournament_draws": float(state.draws),
        f"{prefix}_tournament_losses": float(state.losses),
        f"{prefix}_tournament_goals_for": float(state.goals_for),
        f"{prefix}_tournament_goals_against": float(state.goals_against),
        f"{prefix}_tournament_goal_diff": float(goal_diff),
        f"{prefix}_tournament_points_per_match": safe_rate(state.points, state.matches),
        f"{prefix}_tournament_goal_diff_per_match": safe_rate(goal_diff, state.matches),
        f"{prefix}_tournament_goals_for_per_match": safe_rate(state.goals_for, state.matches),
        f"{prefix}_tournament_goals_against_per_match": safe_rate(state.goals_against, state.matches),
        f"{prefix}_tournament_rest_days": cap_rest_days(rest_days),
        f"{prefix}_tournament_group_match_number": float(min(state.matches + 1, 4)),
        f"{prefix}_tournament_points_progress": safe_rate(state.points, max_group_points_after_three),
        f"{prefix}_tournament_points_needed_for_4": points_needed_for_typical_qualification,
        f"{prefix}_tournament_is_opening_match": int(state.matches == 0),
        f"{prefix}_tournament_is_second_match": int(state.matches == 1),
        f"{prefix}_tournament_is_third_match": int(state.matches == 2),
        f"{prefix}_tournament_likely_must_not_lose": int(state.matches >= 2 and state.points <= 1),
        f"{prefix}_tournament_likely_already_safe": int(state.matches >= 2 and state.points >= 6),
    }


def match_feature_row(
    row: pd.Series,
    states: dict[str, TeamState],
    h2h: dict[tuple[str, str], dict[str, float]],
    tournament_states: dict[tuple[str, str], TournamentTeamState] | None = None,
) -> dict[str, Any]:
    match_date = row["date"]
    home_team = row["home_team"]
    away_team = row["away_team"]
    home_key = normalize_name(home_team)
    away_key = normalize_name(away_team)
    home_state = states[home_key]
    away_state = states[away_key]
    neutral = parse_bool(row["neutral"])
    h2h_key = tuple(sorted((home_key, away_key)))
    h2h_state = h2h[h2h_key]
    current_tournament_key = tournament_key(row)
    home_tournament_state = (
        tournament_states[(current_tournament_key, home_key)] if tournament_states is not None else TournamentTeamState()
    )
    away_tournament_state = (
        tournament_states[(current_tournament_key, away_key)] if tournament_states is not None else TournamentTeamState()
    )
    host_key = normalize_name(row["country"])

    features: dict[str, Any] = {
        "date": match_date,
        "home_team": home_team,
        "away_team": away_team,
        "tournament": row["tournament"],
        "city": row["city"],
        "country": row["country"],
        "neutral": int(neutral),
        "year": int(match_date.year),
        "month": int(match_date.month),
        "is_friendly": int(str(row["tournament"]).lower() == "friendly"),
        "is_world_cup": int("world cup" in str(row["tournament"]).lower()),
        "home_team_is_host_country": int(home_key == host_key and not neutral),
        "away_team_is_host_country": int(away_key == host_key and not neutral),
        "either_team_is_host_country": int((home_key == host_key or away_key == host_key) and not neutral),
        "same_country_match": int(home_key == away_key),
    }

    features.update(state_features("home", home_state, match_date, "home"))
    features.update(state_features("away", away_state, match_date, "away"))
    features.update(tournament_state_features("home", home_tournament_state, match_date))
    features.update(tournament_state_features("away", away_tournament_state, match_date))

    features.update(
        {
            "elo_diff": home_state.elo - away_state.elo,
            "elo_expected_home": elo_expected(home_state.elo, away_state.elo, neutral),
            "matches_diff": home_state.matches - away_state.matches,
            "win_rate_diff": safe_rate(home_state.wins, home_state.matches)
            - safe_rate(away_state.wins, away_state.matches),
            "points_per_match_diff": safe_rate(home_state.points, home_state.matches)
            - safe_rate(away_state.points, away_state.matches),
            "goal_diff_per_match_diff": safe_rate(
                home_state.goals_for - home_state.goals_against, home_state.matches
            )
            - safe_rate(away_state.goals_for - away_state.goals_against, away_state.matches),
            "recent_points_diff": recent_average(home_state, "points") - recent_average(away_state, "points"),
            "recent_goal_diff_diff": recent_average(home_state, "goal_diff")
            - recent_average(away_state, "goal_diff"),
            "recent_goals_for_diff": recent_average(home_state, "goals_for")
            - recent_average(away_state, "goals_for"),
            "recent_goals_against_diff": recent_average(home_state, "goals_against")
            - recent_average(away_state, "goals_against"),
            "recent_win_diff": recent_average(home_state, "win") - recent_average(away_state, "win"),
            "recent_draw_diff": recent_average(home_state, "draw") - recent_average(away_state, "draw"),
            "recent_loss_diff": recent_average(home_state, "loss") - recent_average(away_state, "loss"),
            "recent5_points_diff": recent_average_last(home_state, "points", 5)
            - recent_average_last(away_state, "points", 5),
            "recent5_goal_diff_diff": recent_average_last(home_state, "goal_diff", 5)
            - recent_average_last(away_state, "goal_diff", 5),
            "recent5_goals_for_diff": recent_average_last(home_state, "goals_for", 5)
            - recent_average_last(away_state, "goals_for", 5),
            "recent5_goals_against_diff": recent_average_last(home_state, "goals_against", 5)
            - recent_average_last(away_state, "goals_against", 5),
            "recent5_win_diff": recent_average_last(home_state, "win", 5)
            - recent_average_last(away_state, "win", 5),
            "recent5_draw_diff": recent_average_last(home_state, "draw", 5)
            - recent_average_last(away_state, "draw", 5),
            "recent5_loss_diff": recent_average_last(home_state, "loss", 5)
            - recent_average_last(away_state, "loss", 5),
            "rest_days_diff": features["home_rest_days"] - features["away_rest_days"],
            "h2h_matches": h2h_state["matches"],
            "h2h_home_team_points_per_match": safe_rate(h2h_state.get(home_key, 0.0), h2h_state["matches"]),
            "h2h_away_team_points_per_match": safe_rate(h2h_state.get(away_key, 0.0), h2h_state["matches"]),
            "tournament_points_diff": home_tournament_state.points - away_tournament_state.points,
            "tournament_goal_diff_diff": (
                home_tournament_state.goals_for
                - home_tournament_state.goals_against
                - away_tournament_state.goals_for
                + away_tournament_state.goals_against
            ),
            "tournament_points_per_match_diff": safe_rate(
                home_tournament_state.points, home_tournament_state.matches
            )
            - safe_rate(away_tournament_state.points, away_tournament_state.matches),
            "tournament_matches_diff": home_tournament_state.matches - away_tournament_state.matches,
            "tournament_pressure_diff": features["home_tournament_points_needed_for_4"]
            - features["away_tournament_points_needed_for_4"],
        }
    )
    return features


def update_after_match(
    row: pd.Series,
    states: dict[str, TeamState],
    h2h: dict[tuple[str, str], dict[str, float]],
    recent_window: int,
    tournament_states: dict[tuple[str, str], TournamentTeamState] | None = None,
) -> None:
    home_key = normalize_name(row["home_team"])
    away_key = normalize_name(row["away_team"])
    home = states[home_key]
    away = states[away_key]
    match_date = row["date"]
    neutral = parse_bool(row["neutral"])
    home_goals = int(row["home_score"])
    away_goals = int(row["away_score"])
    goal_diff = home_goals - away_goals

    if goal_diff > 0:
        home_points, away_points = 3, 0
        home_win, draw, away_win = 1, 0, 0
        actual_home = 1.0
    elif goal_diff == 0:
        home_points, away_points = 1, 1
        home_win, draw, away_win = 0, 1, 0
        actual_home = 0.5
    else:
        home_points, away_points = 0, 3
        home_win, draw, away_win = 0, 0, 1
        actual_home = 0.0

    expected_home = elo_expected(home.elo, away.elo, neutral)
    k_value = tournament_weight(row["tournament"]) * goal_multiplier(goal_diff)
    home_delta = k_value * (actual_home - expected_home)
    home.elo += home_delta
    away.elo -= home_delta

    home.matches += 1
    home.wins += home_win
    home.draws += draw
    home.losses += away_win
    home.points += home_points
    home.goals_for += home_goals
    home.goals_against += away_goals
    home.home_matches += 1
    home.home_wins += home_win
    home.home_goals_for += home_goals
    home.home_goals_against += away_goals
    home.last_date = match_date

    away.matches += 1
    away.wins += away_win
    away.draws += draw
    away.losses += home_win
    away.points += away_points
    away.goals_for += away_goals
    away.goals_against += home_goals
    away.away_matches += 1
    away.away_wins += away_win
    away.away_goals_for += away_goals
    away.away_goals_against += home_goals
    away.last_date = match_date

    if home.recent.maxlen != recent_window:
        home.recent = deque(home.recent, maxlen=recent_window)
    if away.recent.maxlen != recent_window:
        away.recent = deque(away.recent, maxlen=recent_window)

    home.recent.append(
        {
            "points": float(home_points),
            "goal_diff": float(goal_diff),
            "goals_for": float(home_goals),
            "goals_against": float(away_goals),
            "win": float(home_win),
            "draw": float(draw),
            "loss": float(away_win),
        }
    )
    away.recent.append(
        {
            "points": float(away_points),
            "goal_diff": float(-goal_diff),
            "goals_for": float(away_goals),
            "goals_against": float(home_goals),
            "win": float(away_win),
            "draw": float(draw),
            "loss": float(home_win),
        }
    )

    h2h_key = tuple(sorted((home_key, away_key)))
    h2h_state = h2h[h2h_key]
    h2h_state["matches"] += 1.0
    h2h_state[home_key] = h2h_state.get(home_key, 0.0) + float(home_points)
    h2h_state[away_key] = h2h_state.get(away_key, 0.0) + float(away_points)

    if tournament_states is not None:
        current_tournament_key = tournament_key(row)
        home_tournament = tournament_states[(current_tournament_key, home_key)]
        away_tournament = tournament_states[(current_tournament_key, away_key)]

        home_tournament.matches += 1
        home_tournament.wins += home_win
        home_tournament.draws += draw
        home_tournament.losses += away_win
        home_tournament.points += home_points
        home_tournament.goals_for += home_goals
        home_tournament.goals_against += away_goals
        home_tournament.last_date = match_date

        away_tournament.matches += 1
        away_tournament.wins += away_win
        away_tournament.draws += draw
        away_tournament.losses += home_win
        away_tournament.points += away_points
        away_tournament.goals_for += away_goals
        away_tournament.goals_against += home_goals
        away_tournament.last_date = match_date


def outcome_label(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"


def load_results(path: Path) -> pd.DataFrame:
    results = pd.read_csv(path)
    expected = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    }
    missing = expected.difference(results.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    results["home_score"] = results["home_score"].astype(int)
    results["away_score"] = results["away_score"].astype(int)
    results = results.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return results


def build_historical_features(
    results: pd.DataFrame,
    recent_window: int,
) -> tuple[pd.DataFrame, dict[str, TeamState], dict[tuple[str, str], dict[str, float]]]:
    states: dict[str, TeamState] = defaultdict(lambda: TeamState(recent=deque(maxlen=recent_window)))
    h2h: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"matches": 0.0})
    tournament_states: dict[tuple[str, str], TournamentTeamState] = defaultdict(TournamentTeamState)
    rows: list[dict[str, Any]] = []

    for _, row in results.iterrows():
        features = match_feature_row(row, states, h2h, tournament_states)
        label = outcome_label(int(row["home_score"]), int(row["away_score"]))
        features["actual_outcome"] = label
        features["target"] = OUTCOME_TO_ID[label]
        features["home_score"] = int(row["home_score"])
        features["away_score"] = int(row["away_score"])
        rows.append(features)
        update_after_match(row, states, h2h, recent_window, tournament_states)

    return pd.DataFrame(rows), states, h2h


def attach_rankings(matches: pd.DataFrame, ranking_path: Path | None) -> pd.DataFrame:
    if ranking_path is None or not ranking_path.exists():
        matches["home_fifa_rank"] = np.nan
        matches["away_fifa_rank"] = np.nan
        matches["home_fifa_points"] = np.nan
        matches["away_fifa_points"] = np.nan
        matches["fifa_rank_diff"] = np.nan
        matches["fifa_points_diff"] = np.nan
        return matches

    rankings = pd.read_csv(ranking_path)
    required = {"country_full", "rank", "total_points", "rank_date"}
    missing = required.difference(rankings.columns)
    if missing:
        print(f"Skipping FIFA rankings: missing columns {sorted(missing)}", file=sys.stderr)
        return attach_rankings(matches, None)

    rankings = rankings[["country_full", "rank", "total_points", "rank_date"]].copy()
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"], errors="coerce")
    rankings["team_key"] = rankings["country_full"].map(normalize_name)
    rankings = rankings.dropna(subset=["rank_date", "team_key"]).sort_values("rank_date")

    output = matches.copy()
    for side in ("home", "away"):
        pieces = []
        tmp = output[["date", f"{side}_team"]].copy()
        tmp["row_id"] = output.index
        tmp["team_key"] = tmp[f"{side}_team"].map(normalize_name)
        for team_key, team_matches in tmp.groupby("team_key", dropna=False):
            team_rankings = rankings[rankings["team_key"] == team_key]
            team_matches = team_matches.sort_values("date")
            if team_rankings.empty:
                team_matches[f"{side}_fifa_rank"] = np.nan
                team_matches[f"{side}_fifa_points"] = np.nan
            else:
                merged = pd.merge_asof(
                    team_matches,
                    team_rankings.sort_values("rank_date"),
                    left_on="date",
                    right_on="rank_date",
                    direction="backward",
                )
                team_matches[f"{side}_fifa_rank"] = merged["rank"].to_numpy()
                team_matches[f"{side}_fifa_points"] = merged["total_points"].to_numpy()
            pieces.append(team_matches[["row_id", f"{side}_fifa_rank", f"{side}_fifa_points"]])

        side_rankings = pd.concat(pieces, ignore_index=True).set_index("row_id")
        output.loc[side_rankings.index, f"{side}_fifa_rank"] = side_rankings[f"{side}_fifa_rank"]
        output.loc[side_rankings.index, f"{side}_fifa_points"] = side_rankings[f"{side}_fifa_points"]

    output["fifa_rank_diff"] = output["away_fifa_rank"] - output["home_fifa_rank"]
    output["fifa_points_diff"] = output["home_fifa_points"] - output["away_fifa_points"]
    output["home_fifa_rank_missing"] = output["home_fifa_rank"].isna().astype(int)
    output["away_fifa_rank_missing"] = output["away_fifa_rank"].isna().astype(int)
    return output


def attach_external_elo_features(matches: pd.DataFrame, elo_path: Path | None) -> tuple[pd.DataFrame, int]:
    output = matches.copy()
    feature_columns = [
        "home_external_elo",
        "away_external_elo",
        "external_elo_diff",
        "external_elo_expected_home",
        "home_external_elo_missing",
        "away_external_elo_missing",
    ]
    for col in feature_columns:
        output[col] = np.nan
    output["home_external_elo_missing"] = 1
    output["away_external_elo_missing"] = 1

    if elo_path is None:
        return output, 0
    elo_path = Path(elo_path)
    if not elo_path.exists():
        return output, 0

    try:
        elo = pd.read_csv(elo_path, usecols=["date", "team", "rating"])
    except ValueError:
        print(f"Skipping external Elo: missing required columns in {elo_path}", file=sys.stderr)
        return output, 0

    elo["date"] = pd.to_datetime(elo["date"], errors="coerce", format="mixed")
    elo["team_key"] = elo["team"].map(normalize_name)
    elo["rating"] = pd.to_numeric(elo["rating"], errors="coerce")
    elo = elo.dropna(subset=["date", "team_key", "rating"]).sort_values("date")
    elo["date"] = elo["date"].astype("datetime64[ns]")
    if elo.empty:
        return output, 0

    for side in ("home", "away"):
        pieces = []
        tmp = output[["date", f"{side}_team"]].copy()
        tmp["row_id"] = output.index
        tmp["team_key"] = tmp[f"{side}_team"].map(normalize_name)
        for team_key, team_matches in tmp.groupby("team_key", dropna=False):
            team_elo = elo[elo["team_key"] == team_key].sort_values("date")
            team_matches = team_matches.sort_values("date")
            if team_elo.empty:
                team_matches[f"{side}_external_elo"] = np.nan
            else:
                # Strictly previous Elo row. Same-day Elo rows can include the match result.
                lookup_matches = team_matches.copy()
                lookup_matches["lookup_date"] = lookup_matches["date"] - pd.Timedelta(nanoseconds=1)
                merged = pd.merge_asof(
                    lookup_matches,
                    team_elo[["date", "rating"]],
                    left_on="lookup_date",
                    right_on="date",
                    direction="backward",
                )
                team_matches[f"{side}_external_elo"] = merged["rating"].to_numpy()
            pieces.append(team_matches[["row_id", f"{side}_external_elo"]])

        side_elo = pd.concat(pieces, ignore_index=True).set_index("row_id")
        output.loc[side_elo.index, f"{side}_external_elo"] = side_elo[f"{side}_external_elo"]
        output[f"{side}_external_elo_missing"] = output[f"{side}_external_elo"].isna().astype(int)

    output["external_elo_diff"] = output["home_external_elo"] - output["away_external_elo"]
    output["external_elo_expected_home"] = [
        elo_expected(
            float(home) if pd.notna(home) else 1500.0,
            float(away) if pd.notna(away) else 1500.0,
            bool(neutral),
        )
        for home, away, neutral in zip(
            output["home_external_elo"],
            output["away_external_elo"],
            output["neutral"],
        )
    ]
    matched = int(
        (
            output["home_external_elo"].notna()
            & output["away_external_elo"].notna()
        ).sum()
    )
    return output, matched


def attach_context_features(matches: pd.DataFrame, ranking_path: Path | None) -> pd.DataFrame:
    output = matches.copy()
    output["home_team_key"] = output["home_team"].map(normalize_name)
    output["away_team_key"] = output["away_team"].map(normalize_name)
    output["host_country_key"] = output["country"].map(normalize_name) if "country" in output.columns else ""

    confed_by_team: dict[str, str] = {}
    if ranking_path is not None and ranking_path.exists():
        try:
            rankings = pd.read_csv(ranking_path, usecols=["country_full", "rank_date", "confederation"])
            rankings["rank_date"] = pd.to_datetime(rankings["rank_date"], errors="coerce")
            rankings["team_key"] = rankings["country_full"].map(normalize_name)
            rankings = rankings.dropna(subset=["rank_date"]).sort_values("rank_date")
            latest = rankings.drop_duplicates("team_key", keep="last")
            confed_by_team = dict(zip(latest["team_key"], latest["confederation"].fillna("Unknown")))
        except ValueError:
            confed_by_team = {}

    output["home_confederation"] = output["home_team_key"].map(confed_by_team).fillna("Unknown")
    output["away_confederation"] = output["away_team_key"].map(confed_by_team).fillna("Unknown")
    output["host_confederation"] = output["host_country_key"].map(confed_by_team).fillna("Unknown")
    output["same_confederation"] = (output["home_confederation"] == output["away_confederation"]).astype(int)
    output["home_same_confederation_as_host"] = (
        (output["home_confederation"] == output["host_confederation"])
        & output["host_confederation"].ne("Unknown")
    ).astype(int)
    output["away_same_confederation_as_host"] = (
        (output["away_confederation"] == output["host_confederation"])
        & output["host_confederation"].ne("Unknown")
    ).astype(int)
    output["confederation_host_advantage_diff"] = (
        output["home_same_confederation_as_host"] - output["away_same_confederation_as_host"]
    )
    return output.drop(columns=["home_team_key", "away_team_key", "host_country_key"])


def attach_geo_features(
    matches: pd.DataFrame,
    city_locations_path: Path | None,
    country_reps_path: Path | None,
) -> tuple[pd.DataFrame, int]:
    if (
        city_locations_path is None
        or country_reps_path is None
        or not city_locations_path.exists()
        or not country_reps_path.exists()
    ):
        return matches, 0

    city_locations = pd.read_csv(city_locations_path)
    country_reps = pd.read_csv(country_reps_path)
    required_city = {"city_key", "country_key", "lat", "lng", "population"}
    required_country = {"country_key", "lat", "lng", "population"}
    if not required_city.issubset(city_locations.columns) or not required_country.issubset(country_reps.columns):
        print("Skipping geo features: extracted city/country geo files have unexpected columns.", file=sys.stderr)
        return matches, 0

    city_locations["population"] = pd.to_numeric(city_locations["population"], errors="coerce").fillna(0.0)
    exact_city_lookup = {
        (str(row.city_key), str(row.country_key)): row
        for row in city_locations.itertuples(index=False)
    }
    city_fallback_lookup = {
        str(row.city_key): row
        for row in city_locations.sort_values("population").drop_duplicates("city_key", keep="last").itertuples(index=False)
    }
    country_lookup = {str(row.country_key): row for row in country_reps.itertuples(index=False)}

    geo_rows: list[dict[str, float]] = []
    matched = 0
    for row in matches.itertuples(index=False):
        city_key = normalize_name(getattr(row, "city", ""))
        venue_country_key = normalize_name(getattr(row, "country", ""))
        home_key = normalize_name(getattr(row, "home_team", ""))
        away_key = normalize_name(getattr(row, "away_team", ""))

        venue = exact_city_lookup.get((city_key, venue_country_key))
        exact_venue_match = int(venue is not None)
        if venue is None:
            venue = city_fallback_lookup.get(city_key)
        home_base = country_lookup.get(home_key)
        away_base = country_lookup.get(away_key)

        venue_lat = np.nan if venue is None else float(venue.lat)
        venue_lng = np.nan if venue is None else float(venue.lng)
        venue_population = 0.0 if venue is None else float(venue.population)
        home_lat = np.nan if home_base is None else float(home_base.lat)
        home_lng = np.nan if home_base is None else float(home_base.lng)
        away_lat = np.nan if away_base is None else float(away_base.lat)
        away_lng = np.nan if away_base is None else float(away_base.lng)

        home_travel = haversine_km(home_lat, home_lng, venue_lat, venue_lng)
        away_travel = haversine_km(away_lat, away_lng, venue_lat, venue_lng)
        venue_offset = longitude_to_utc_offset(venue_lng)
        home_offset = longitude_to_utc_offset(home_lng)
        away_offset = longitude_to_utc_offset(away_lng)
        home_tz_change = abs(home_offset - venue_offset) if pd.notna(home_offset) and pd.notna(venue_offset) else np.nan
        away_tz_change = abs(away_offset - venue_offset) if pd.notna(away_offset) and pd.notna(venue_offset) else np.nan
        venue_available = int(venue is not None)
        matched += venue_available

        geo_rows.append(
            {
                "venue_geo_available": venue_available,
                "venue_geo_exact_city_country": exact_venue_match,
                "venue_lat": venue_lat,
                "venue_lng": venue_lng,
                "venue_abs_lat": abs(venue_lat) if pd.notna(venue_lat) else np.nan,
                "venue_population_log": math.log1p(venue_population),
                "home_base_geo_available": int(home_base is not None),
                "away_base_geo_available": int(away_base is not None),
                "home_travel_km": home_travel,
                "away_travel_km": away_travel,
                "travel_km_diff": home_travel - away_travel if pd.notna(home_travel) and pd.notna(away_travel) else np.nan,
                "abs_travel_km_diff": abs(home_travel - away_travel)
                if pd.notna(home_travel) and pd.notna(away_travel)
                else np.nan,
                "total_travel_km": home_travel + away_travel if pd.notna(home_travel) and pd.notna(away_travel) else np.nan,
                "venue_utc_offset_approx": venue_offset,
                "home_utc_offset_approx": home_offset,
                "away_utc_offset_approx": away_offset,
                "home_tz_change_approx": home_tz_change,
                "away_tz_change_approx": away_tz_change,
                "tz_change_diff": home_tz_change - away_tz_change
                if pd.notna(home_tz_change) and pd.notna(away_tz_change)
                else np.nan,
                "abs_tz_change_diff": abs(home_tz_change - away_tz_change)
                if pd.notna(home_tz_change) and pd.notna(away_tz_change)
                else np.nan,
            }
        )

    geo_features = pd.DataFrame(geo_rows, index=matches.index)
    return pd.concat([matches.reset_index(drop=True), geo_features.reset_index(drop=True)], axis=1), matched


def build_fjelstul_manager_features(fjelstul_dir: Path) -> pd.DataFrame:
    manager_path = fjelstul_dir / "manager_appearances.csv"
    team_path = fjelstul_dir / "team_appearances.csv"
    if not manager_path.exists() or not team_path.exists():
        raise FileNotFoundError("Fjelstul manager/team appearance files are missing.")

    managers = pd.read_csv(
        manager_path,
        usecols=[
            "match_date",
            "match_id",
            "tournament_name",
            "team_name",
            "manager_id",
            "country_name",
        ],
    )
    teams = pd.read_csv(
        team_path,
        usecols=[
            "match_id",
            "tournament_name",
            "team_name",
            "opponent_name",
            "goals_for",
            "goals_against",
            "win",
            "draw",
            "lose",
        ],
    )
    managers = managers[managers["tournament_name"].astype(str).str.contains("FIFA Men's World Cup", regex=False)]
    teams = teams[teams["tournament_name"].astype(str).str.contains("FIFA Men's World Cup", regex=False)]
    rows = managers.merge(
        teams,
        on=["match_id", "tournament_name", "team_name"],
        how="left",
    )
    rows["match_date"] = pd.to_datetime(rows["match_date"], errors="coerce")
    rows["team_key"] = [
        normalize_fjelstul_team_name(team, date)
        for team, date in zip(rows["team_name"], rows["match_date"], strict=False)
    ]
    rows["opponent_key"] = [
        normalize_fjelstul_team_name(team, date)
        for team, date in zip(rows["opponent_name"], rows["match_date"], strict=False)
    ]
    rows["country_key"] = [
        normalize_fjelstul_team_name(country, date)
        for country, date in zip(rows["country_name"], rows["match_date"], strict=False)
    ]
    rows["manager_same_country"] = (
        rows["team_key"] == rows["country_key"]
    ).astype(int)
    rows = rows.dropna(subset=["match_date", "team_key", "opponent_key", "manager_id"])
    rows = rows.sort_values(["match_date", "match_id", "team_key"]).reset_index(drop=True)

    manager_state: dict[str, dict[str, float]] = defaultdict(
        lambda: {"matches": 0.0, "points": 0.0, "goals_for": 0.0, "goals_against": 0.0}
    )
    last_team_manager: dict[str, str] = {}
    team_manager_streak: dict[str, int] = defaultdict(int)
    feature_rows: list[dict[str, Any]] = []

    for row in rows.itertuples(index=False):
        manager = str(row.manager_id)
        team_key = row.team_key
        manager_stats = manager_state[manager]
        changed = int(team_key in last_team_manager and last_team_manager[team_key] != manager)
        current_streak = team_manager_streak[team_key] if last_team_manager.get(team_key) == manager else 0
        points = 3.0 if int(row.win) == 1 else (1.0 if int(row.draw) == 1 else 0.0)
        feature_rows.append(
            {
                "date_key": pd.Timestamp(row.match_date).date().isoformat(),
                "team_key": team_key,
                "opponent_key": row.opponent_key,
                "manager_matches_before": manager_stats["matches"],
                "manager_points_per_match_before": safe_rate(manager_stats["points"], manager_stats["matches"]),
                "manager_goal_diff_per_match_before": safe_rate(
                    manager_stats["goals_for"] - manager_stats["goals_against"],
                    manager_stats["matches"],
                ),
                "manager_same_country": int(row.manager_same_country),
                "team_manager_changed": changed,
                "team_manager_streak_before": float(current_streak),
            }
        )

        manager_stats["matches"] += 1.0
        manager_stats["points"] += points
        manager_stats["goals_for"] += float(row.goals_for)
        manager_stats["goals_against"] += float(row.goals_against)
        if last_team_manager.get(team_key) == manager:
            team_manager_streak[team_key] += 1
        else:
            team_manager_streak[team_key] = 1
        last_team_manager[team_key] = manager

    return pd.DataFrame(feature_rows)


def attach_fjelstul_manager_features(matches: pd.DataFrame, fjelstul_dir: Path | None) -> tuple[pd.DataFrame, int]:
    if fjelstul_dir is None or not fjelstul_dir.exists():
        return matches, 0
    try:
        manager_features = build_fjelstul_manager_features(fjelstul_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Skipping Fjelstul manager features: {exc}", file=sys.stderr)
        return matches, 0
    manager_features = manager_features.drop_duplicates(
        subset=["date_key", "team_key", "opponent_key"],
        keep="last",
    )

    output = matches.copy()
    output["date_key"] = output["date"].dt.date.astype(str)
    output["home_key"] = output["home_team"].map(normalize_name)
    output["away_key"] = output["away_team"].map(normalize_name)

    side_features = [
        "manager_matches_before",
        "manager_points_per_match_before",
        "manager_goal_diff_per_match_before",
        "manager_same_country",
        "team_manager_changed",
        "team_manager_streak_before",
    ]
    home_manager = manager_features.rename(
        columns={
            "team_key": "home_key",
            "opponent_key": "away_key",
            **{col: f"home_{col}" for col in side_features},
        }
    )
    away_manager = manager_features.rename(
        columns={
            "team_key": "away_key",
            "opponent_key": "home_key",
            **{col: f"away_{col}" for col in side_features},
        }
    )
    output = output.merge(
        home_manager[["date_key", "home_key", "away_key"] + [f"home_{col}" for col in side_features]],
        on=["date_key", "home_key", "away_key"],
        how="left",
    )
    output = output.merge(
        away_manager[["date_key", "home_key", "away_key"] + [f"away_{col}" for col in side_features]],
        on=["date_key", "home_key", "away_key"],
        how="left",
    )
    matched = int(
        output["home_manager_matches_before"].notna().sum()
        + output["away_manager_matches_before"].notna().sum()
    )
    for col in side_features:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        output[home_col] = output[home_col].fillna(0.0)
        output[away_col] = output[away_col].fillna(0.0)
        output[f"{col}_diff"] = output[home_col] - output[away_col]

    return output.drop(columns=["date_key", "home_key", "away_key"]), matched


def attach_xfkz_snapshot_features(matches: pd.DataFrame, snapshot_path: Path | None) -> tuple[pd.DataFrame, int]:
    if snapshot_path is None or not snapshot_path.exists():
        return matches, 0

    snapshots = pd.read_csv(snapshot_path)
    snapshots["date"] = pd.to_datetime(snapshots["date"], errors="coerce")
    snapshots = snapshots.dropna(subset=["date", "country_key"]).sort_values(["country_key", "date"])
    feature_cols = [col for col in snapshots.columns if col not in {"date", "country_key"}]
    output = matches.copy()

    for side in ("home", "away"):
        side_frame = output[["date", f"{side}_team"]].copy()
        side_frame["row_id"] = output.index
        side_frame["country_key"] = side_frame[f"{side}_team"].map(normalize_name)
        pieces = []
        for country_key, team_matches in side_frame.groupby("country_key", dropna=False):
            team_matches = team_matches.sort_values("date")
            team_snapshots = snapshots[snapshots["country_key"] == country_key].sort_values("date")
            if team_snapshots.empty:
                empty = team_matches[["row_id"]].copy()
                for col in feature_cols:
                    empty[f"{side}_xfkz_{col}"] = 0.0
                pieces.append(empty)
                continue
            merged = pd.merge_asof(
                team_matches,
                team_snapshots[["date"] + feature_cols],
                on="date",
                direction="backward",
            )
            merged = merged[["row_id"] + feature_cols].rename(
                columns={col: f"{side}_xfkz_{col}" for col in feature_cols}
            )
            pieces.append(merged)

        side_features = pd.concat(pieces, ignore_index=True).set_index("row_id")
        for col in side_features.columns:
            output.loc[side_features.index, col] = pd.to_numeric(side_features[col], errors="coerce").fillna(0.0)

    for col in feature_cols:
        home_col = f"home_xfkz_{col}"
        away_col = f"away_xfkz_{col}"
        if home_col in output.columns and away_col in output.columns:
            output[f"xfkz_{col}_diff"] = output[home_col].fillna(0.0) - output[away_col].fillna(0.0)

    matched = int(
        (
            output.get("home_xfkz_market_player_count", pd.Series(0, index=output.index)).fillna(0).gt(0)
            | output.get("away_xfkz_market_player_count", pd.Series(0, index=output.index)).fillna(0).gt(0)
        ).sum()
    )
    return output, matched


def find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {col.lower().strip(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def squad_competition_key(value: Any) -> str:
    text = normalize_person_name(value)
    if not text or "qualification" in text or "qualifier" in text:
        return ""
    if "fifa world cup" in text or text == "world cup":
        return "fifa world cup"
    if "uefa euro" in text or "european championship" in text:
        return "uefa euro"
    if "copa america" in text:
        return "copa america"
    if "africa cup" in text or "african cup" in text:
        return "africa cup of nations"
    if "afc asian cup" in text or text == "asian cup":
        return "afc asian cup"
    if "concacaf gold cup" in text or text == "gold cup" or "concacaf championship" in text:
        return "gold cup"
    return ""


def squad_position_key(value: Any) -> str:
    text = normalize_person_name(value)
    if text in {"gk", "goalkeeper"}:
        return "gk"
    if text in {"df", "defender"} or "defender" in text or "back" in text:
        return "df"
    if text in {"mf", "midfielder"} or "midfield" in text:
        return "mf"
    if text in {"fw", "forward"} or "forward" in text or "striker" in text or "winger" in text:
        return "fw"
    return "other"


def team_pair_key(home_key: str, away_key: str) -> str:
    return "|".join(sorted([str(home_key), str(away_key)]))


def lineup_position_group(value: Any) -> str:
    text = str(value).upper()
    if "GK" in text:
        return "gk"
    if any(token in text for token in ("CB", "LB", "RB", "LWB", "RWB", "DEF")):
        return "df"
    if any(token in text for token in ("CDM", "CM", "CAM", "LM", "RM", "MID")):
        return "mf"
    if any(token in text for token in ("ST", "CF", "LW", "RW", "FW", "ATT")):
        return "fw"
    return "other"


def attach_odds(matches: pd.DataFrame, odds_path: Path | None) -> tuple[pd.DataFrame, int]:
    if odds_path is None:
        return matches, 0
    odds_path = Path(odds_path)
    if not odds_path.exists():
        raise FileNotFoundError(f"Odds CSV not found: {odds_path}")

    odds = pd.read_csv(odds_path)
    columns = list(odds.columns)
    date_col = find_column(columns, ("date", "date_gmt", "match_date"))
    home_col = find_column(columns, ("home_team", "home_team_name", "home"))
    away_col = find_column(columns, ("away_team", "away_team_name", "away"))
    home_odds_col = find_column(columns, ("home_odds", "odds_ft_home_team_win", "odds_home"))
    draw_odds_col = find_column(columns, ("draw_odds", "odds_ft_draw", "odds_draw"))
    away_odds_col = find_column(columns, ("away_odds", "odds_ft_away_team_win", "odds_away"))
    optional_columns = {
        "odds_max_home_win": find_column(columns, ("max_home_odds", "max_odds_home_win")),
        "odds_max_draw": find_column(columns, ("max_draw_odds", "max_odds_draw")),
        "odds_max_away_win": find_column(columns, ("max_away_odds", "max_odds_away_win")),
        "odds_n_home_win": find_column(columns, ("n_odds_home_win", "n_home_odds")),
        "odds_n_draw": find_column(columns, ("n_odds_draw", "n_draw_odds")),
        "odds_n_away_win": find_column(columns, ("n_odds_away_win", "n_away_odds")),
    }
    required = [date_col, home_col, away_col, home_odds_col, draw_odds_col, away_odds_col]
    if any(col is None for col in required):
        raise ValueError(
            "Odds CSV needs date/home/away columns and decimal 1X2 odds. "
            "Supported odds column names include home_odds, draw_odds, away_odds "
            "or FootyStats odds_ft_home_team_win, odds_ft_draw, odds_ft_away_team_win."
        )

    odds = odds.rename(
        columns={
            date_col: "odds_date",
            home_col: "odds_home_team",
            away_col: "odds_away_team",
            home_odds_col: "odds_home_win",
            draw_odds_col: "odds_draw",
            away_odds_col: "odds_away_win",
            **{source: target for target, source in optional_columns.items() if source is not None},
        }
    )
    odds["date_key"] = pd.to_datetime(odds["odds_date"], errors="coerce").dt.date.astype(str)
    odds["home_key"] = odds["odds_home_team"].map(normalize_name)
    odds["away_key"] = odds["odds_away_team"].map(normalize_name)
    odds_feature_cols = ["odds_home_win", "odds_draw", "odds_away_win"]
    for col in optional_columns:
        if col in odds.columns:
            odds_feature_cols.append(col)
    for col in odds_feature_cols:
        odds[col] = pd.to_numeric(odds[col], errors="coerce")
    odds = odds.dropna(subset=["date_key", "home_key", "away_key", "odds_home_win", "odds_draw", "odds_away_win"])
    odds = odds.drop_duplicates(subset=["date_key", "home_key", "away_key"], keep="last")

    output = matches.copy()
    output["date_key"] = output["date"].dt.date.astype(str)
    output["home_key"] = output["home_team"].map(normalize_name)
    output["away_key"] = output["away_team"].map(normalize_name)
    merged = output.merge(
        odds[["date_key", "home_key", "away_key"] + odds_feature_cols],
        on=["date_key", "home_key", "away_key"],
        how="left",
    )

    implied = 1.0 / merged[["odds_home_win", "odds_draw", "odds_away_win"]]
    implied_sum = implied.sum(axis=1)
    merged["odds_has_1x2"] = merged[["odds_home_win", "odds_draw", "odds_away_win"]].notna().all(axis=1).astype(float)
    merged["odds_overround"] = implied_sum
    merged["odds_prob_home_win"] = implied["odds_home_win"] / implied_sum
    merged["odds_prob_draw"] = implied["odds_draw"] / implied_sum
    merged["odds_prob_away_win"] = implied["odds_away_win"] / implied_sum
    merged["odds_home_draw_gap"] = merged["odds_prob_home_win"] - merged["odds_prob_draw"]
    merged["odds_home_away_gap"] = merged["odds_prob_home_win"] - merged["odds_prob_away_win"]
    merged["odds_favorite_prob"] = merged[["odds_prob_home_win", "odds_prob_draw", "odds_prob_away_win"]].max(axis=1)
    merged["odds_favorite_is_home"] = (
        (merged["odds_prob_home_win"] >= merged["odds_prob_draw"])
        & (merged["odds_prob_home_win"] >= merged["odds_prob_away_win"])
    ).astype(float)
    merged["odds_favorite_is_draw"] = (
        (merged["odds_prob_draw"] > merged["odds_prob_home_win"])
        & (merged["odds_prob_draw"] >= merged["odds_prob_away_win"])
    ).astype(float)
    merged["odds_favorite_is_away"] = (
        (merged["odds_prob_away_win"] > merged["odds_prob_home_win"])
        & (merged["odds_prob_away_win"] > merged["odds_prob_draw"])
    ).astype(float)
    if {"odds_max_home_win", "odds_max_draw", "odds_max_away_win"}.issubset(merged.columns):
        max_implied = 1.0 / merged[["odds_max_home_win", "odds_max_draw", "odds_max_away_win"]]
        max_implied_sum = max_implied.sum(axis=1)
        merged["odds_max_overround"] = max_implied_sum
        merged["odds_max_prob_home_win"] = max_implied["odds_max_home_win"] / max_implied_sum
        merged["odds_max_prob_draw"] = max_implied["odds_max_draw"] / max_implied_sum
        merged["odds_max_prob_away_win"] = max_implied["odds_max_away_win"] / max_implied_sum
        merged["odds_home_win_max_gap"] = merged["odds_max_home_win"] - merged["odds_home_win"]
        merged["odds_draw_max_gap"] = merged["odds_max_draw"] - merged["odds_draw"]
        merged["odds_away_win_max_gap"] = merged["odds_max_away_win"] - merged["odds_away_win"]
    matched = int(merged["odds_home_win"].notna().sum())
    return merged.drop(columns=["date_key", "home_key", "away_key"]), matched


def attach_goalscorer_form_features(
    matches: pd.DataFrame,
    goalscorers_path: Path | None,
    window_days: int,
    half_life_days: int,
) -> tuple[pd.DataFrame, int]:
    if goalscorers_path is None:
        return matches, 0
    goalscorers_path = Path(goalscorers_path)
    if not goalscorers_path.exists():
        raise FileNotFoundError(f"Goalscorers CSV not found: {goalscorers_path}")

    goals = pd.read_csv(goalscorers_path)
    columns = list(goals.columns)
    date_col = find_column(columns, ("date", "match_date"))
    team_col = find_column(columns, ("team", "scoring_team"))
    scorer_col = find_column(columns, ("scorer", "player", "player_name"))
    own_goal_col = find_column(columns, ("own_goal", "own goal"))
    penalty_col = find_column(columns, ("penalty",))
    required = [date_col, team_col, scorer_col]
    if any(col is None for col in required):
        raise ValueError("Goalscorers CSV needs date, team, and scorer columns.")

    goals = goals.rename(
        columns={
            date_col: "goal_date",
            team_col: "team",
            scorer_col: "scorer",
            **({own_goal_col: "own_goal"} if own_goal_col is not None else {}),
            **({penalty_col: "penalty"} if penalty_col is not None else {}),
        }
    )
    goals["goal_date"] = pd.to_datetime(goals["goal_date"], errors="coerce")
    goals["team_key"] = goals["team"].map(normalize_name)
    goals["scorer_key"] = goals["scorer"].map(normalize_person_name)
    if "own_goal" in goals.columns:
        goals["own_goal"] = goals["own_goal"].map(parse_bool)
    else:
        goals["own_goal"] = False
    if "penalty" in goals.columns:
        goals["penalty"] = goals["penalty"].map(parse_bool)
    else:
        goals["penalty"] = False
    goals = goals.dropna(subset=["goal_date"])
    goals = goals[(goals["team_key"] != "") & (goals["scorer_key"] != "") & ~goals["own_goal"]].copy()
    goals = goals.sort_values(["team_key", "goal_date"]).reset_index(drop=True)

    team_goal_events: dict[str, list[dict[str, Any]]] = {}
    for team_key, group in goals.groupby("team_key", sort=False):
        team_goal_events[team_key] = [
            {
                "date": pd.Timestamp(row.goal_date),
                "scorer": str(row.scorer_key),
                "penalty": bool(row.penalty),
            }
            for row in group.itertuples(index=False)
        ]

    output = matches.copy()
    side_rows = []
    for side in ("home", "away"):
        side_rows.append(
            pd.DataFrame(
                {
                    "row_id": np.arange(len(output), dtype=np.int64),
                    "side": side,
                    "date": pd.to_datetime(output["date"], errors="coerce"),
                    "team_key": output[f"{side}_team"].map(normalize_name),
                }
            )
        )
    side_frame = pd.concat(side_rows, ignore_index=True)
    feature_rows: list[dict[str, Any]] = []
    window = pd.Timedelta(days=max(int(window_days), 1))
    half_life = float(max(int(half_life_days), 1))
    decay = math.log(2.0) / half_life

    for team_key, team_matches in side_frame.sort_values(["team_key", "date", "row_id"]).groupby("team_key"):
        events = team_goal_events.get(team_key, [])
        event_index = 0
        active_events: deque[dict[str, Any]] = deque()
        scorer_counts: defaultdict[str, int] = defaultdict(int)
        total_goals = 0
        penalty_goals = 0

        for match in team_matches.itertuples(index=False):
            match_date = pd.Timestamp(match.date)
            if pd.isna(match_date):
                continue

            while event_index < len(events) and events[event_index]["date"] < match_date:
                event = events[event_index]
                active_events.append(event)
                scorer_counts[event["scorer"]] += 1
                total_goals += 1
                penalty_goals += int(event["penalty"])
                event_index += 1

            cutoff = match_date - window
            while active_events and active_events[0]["date"] < cutoff:
                old_event = active_events.popleft()
                scorer = old_event["scorer"]
                scorer_counts[scorer] -= 1
                if scorer_counts[scorer] <= 0:
                    del scorer_counts[scorer]
                total_goals -= 1
                penalty_goals -= int(old_event["penalty"])

            sorted_counts = sorted(scorer_counts.values(), reverse=True)
            top1 = float(sorted_counts[0]) if sorted_counts else 0.0
            top3 = float(sum(sorted_counts[:3]))
            top5 = float(sum(sorted_counts[:5]))
            recency_goals = float(
                sum(math.exp(-decay * max((match_date - event["date"]).days, 0)) for event in active_events)
            )
            total_goals_float = float(total_goals)
            feature_rows.append(
                {
                    "row_id": int(match.row_id),
                    "side": match.side,
                    "scorer_goals_2y": total_goals_float,
                    "scorer_recency_goals_2y": recency_goals,
                    "unique_scorers_2y": float(len(scorer_counts)),
                    "top_scorer_goals_2y": top1,
                    "top3_scorer_goals_2y": top3,
                    "top5_scorer_goals_2y": top5,
                    "penalty_goals_2y": float(penalty_goals),
                    "penalty_goal_share_2y": safe_rate(float(penalty_goals), total_goals_float),
                    "scorer_top1_share_2y": safe_rate(top1, total_goals_float),
                    "scorer_top3_share_2y": safe_rate(top3, total_goals_float),
                }
            )

    feature_cols = [
        "scorer_goals_2y",
        "scorer_recency_goals_2y",
        "unique_scorers_2y",
        "top_scorer_goals_2y",
        "top3_scorer_goals_2y",
        "top5_scorer_goals_2y",
        "penalty_goals_2y",
        "penalty_goal_share_2y",
        "scorer_top1_share_2y",
        "scorer_top3_share_2y",
    ]
    side_features = pd.DataFrame(feature_rows)
    for side in ("home", "away"):
        for col in feature_cols:
            output[f"{side}_{col}"] = 0.0
        if side_features.empty:
            continue
        side_feature_frame = side_features[side_features["side"] == side].set_index("row_id")
        for col in feature_cols:
            output.loc[side_feature_frame.index, f"{side}_{col}"] = pd.to_numeric(
                side_feature_frame[col],
                errors="coerce",
            ).fillna(0.0)

    diff_features = {
        f"{col}_diff": output[f"home_{col}"].fillna(0.0) - output[f"away_{col}"].fillna(0.0)
        for col in feature_cols
    }
    output = pd.concat([output, pd.DataFrame(diff_features, index=output.index)], axis=1)

    matched = int((output["home_scorer_goals_2y"].gt(0) | output["away_scorer_goals_2y"].gt(0)).sum())
    return output, matched


def attach_tournament_squad_features(
    matches: pd.DataFrame,
    squads_path: Path | None,
    transfermarkt_dir: Path | None = None,
    use_market_values: bool = True,
    sofifa_ratings_path: Path | None = None,
    use_sofifa_ratings: bool = False,
) -> tuple[pd.DataFrame, int]:
    if squads_path is None:
        return matches, 0
    squads_path = Path(squads_path)
    if not squads_path.exists():
        return matches, 0

    squads = pd.read_csv(squads_path)
    columns = list(squads.columns)
    competition_col = find_column(columns, ("competition", "tournament"))
    year_col = find_column(columns, ("year", "season"))
    team_col = find_column(columns, ("team", "country", "national_team"))
    position_col = find_column(columns, ("position", "pos"))
    age_col = find_column(columns, ("age",))
    caps_col = find_column(columns, ("caps", "appearances"))
    goals_col = find_column(columns, ("goals", "international_goals"))
    required = [competition_col, year_col, team_col]
    if any(col is None for col in required):
        raise ValueError("Squads CSV needs competition, year, and team columns.")

    squads = squads.rename(
        columns={
            competition_col: "competition",
            year_col: "squad_year",
            team_col: "team",
            **({position_col: "position"} if position_col is not None else {}),
            **({age_col: "age"} if age_col is not None else {}),
            **({caps_col: "caps"} if caps_col is not None else {}),
            **({goals_col: "goals"} if goals_col is not None else {}),
        }
    )
    squads["competition_key"] = squads["competition"].map(squad_competition_key)
    squads["team_key"] = squads["team"].map(normalize_name)
    squads["squad_year"] = pd.to_numeric(squads["squad_year"], errors="coerce")
    squads["position_key"] = squads["position"].map(squad_position_key) if "position" in squads.columns else "other"
    squads["age"] = pd.to_numeric(squads["age"], errors="coerce") if "age" in squads.columns else np.nan
    squads["caps"] = pd.to_numeric(squads["caps"], errors="coerce").fillna(0.0) if "caps" in squads.columns else 0.0
    squads["goals"] = pd.to_numeric(squads["goals"], errors="coerce").fillna(0.0) if "goals" in squads.columns else 0.0
    squads = squads[
        (squads["competition_key"] != "")
        & (squads["team_key"] != "")
        & squads["squad_year"].notna()
    ].copy()
    squads["squad_year"] = squads["squad_year"].astype(int)
    squads["player_key"] = squads["player"].map(normalize_person_name) if "player" in squads.columns else ""
    squads["dob_key"] = (
        pd.to_datetime(squads["date_of_birth"], errors="coerce").dt.date
        if "date_of_birth" in squads.columns
        else pd.NaT
    )

    market_enabled = False
    valuation_lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if use_market_values and transfermarkt_dir is not None:
        transfermarkt_dir = Path(transfermarkt_dir)
        players_path = transfermarkt_dir / "players.csv"
        valuations_path = transfermarkt_dir / "player_valuations.csv"
        if players_path.exists() and valuations_path.exists():
            players = pd.read_csv(
                players_path,
                usecols=[
                    "player_id",
                    "name",
                    "country_of_citizenship",
                    "date_of_birth",
                    "market_value_in_eur",
                    "highest_market_value_in_eur",
                ],
            )
            players["player_key"] = players["name"].map(normalize_person_name)
            players["dob_key"] = pd.to_datetime(players["date_of_birth"], errors="coerce").dt.date
            players["citizenship_key"] = players["country_of_citizenship"].map(normalize_name)
            players["market_value_in_eur"] = pd.to_numeric(players["market_value_in_eur"], errors="coerce")
            players["highest_market_value_in_eur"] = pd.to_numeric(
                players["highest_market_value_in_eur"],
                errors="coerce",
            )
            players = players.dropna(subset=["player_id", "dob_key"])
            players["player_id"] = players["player_id"].astype(int)
            player_candidates: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
            for row in players.itertuples(index=False):
                player_candidates[(row.player_key, row.dob_key)].append(
                    {
                        "player_id": int(row.player_id),
                        "citizenship_key": row.citizenship_key,
                        "current_market_value": float(row.market_value_in_eur)
                        if pd.notna(row.market_value_in_eur)
                        else np.nan,
                        "highest_market_value": float(row.highest_market_value_in_eur)
                        if pd.notna(row.highest_market_value_in_eur)
                        else np.nan,
                    }
                )

            def match_player(row: pd.Series) -> pd.Series:
                candidates = player_candidates.get((row["player_key"], row["dob_key"]), [])
                if not candidates:
                    return pd.Series(
                        {
                            "tm_player_id": np.nan,
                            "tm_current_market_value": np.nan,
                            "tm_highest_market_value": np.nan,
                        }
                    )
                preferred = [candidate for candidate in candidates if candidate["citizenship_key"] == row["team_key"]]
                candidate = preferred[0] if preferred else candidates[0]
                return pd.Series(
                    {
                        "tm_player_id": candidate["player_id"],
                        "tm_current_market_value": candidate["current_market_value"],
                        "tm_highest_market_value": candidate["highest_market_value"],
                    }
                )

            matched_players = squads.apply(match_player, axis=1)
            squads = pd.concat([squads.reset_index(drop=True), matched_players.reset_index(drop=True)], axis=1)

            valuations = pd.read_csv(valuations_path, usecols=["player_id", "date", "market_value_in_eur"])
            valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
            valuations["market_value_in_eur"] = pd.to_numeric(valuations["market_value_in_eur"], errors="coerce")
            valuations = valuations.dropna(subset=["player_id", "date", "market_value_in_eur"])
            valuations["player_id"] = valuations["player_id"].astype(int)
            valuations["date_ord"] = valuations["date"].map(lambda value: value.toordinal())
            for player_id, group in valuations.sort_values("date_ord").groupby("player_id", sort=False):
                valuation_lookup[int(player_id)] = (
                    group["date_ord"].to_numpy(dtype=np.int64),
                    group["market_value_in_eur"].to_numpy(dtype=np.float64),
                )
            market_enabled = True

    sofifa_enabled = False
    sofifa_rating_columns = [
        "overall",
        "potential",
        "value_eur",
        "pace",
        "shooting",
        "passing",
        "dribbling",
        "defending",
        "physic",
        "movement_reactions",
        "mentality_composure",
    ]
    sofifa_lookup: dict[int, tuple[np.ndarray, pd.DataFrame]] = {}
    if use_sofifa_ratings and sofifa_ratings_path is not None:
        sofifa_ratings_path = Path(sofifa_ratings_path)
        if sofifa_ratings_path.exists():
            sofifa = pd.read_csv(
                sofifa_ratings_path,
                usecols=lambda col: col
                in {
                    "sofifa_id",
                    "available_from",
                    "short_name",
                    "long_name",
                    "dob",
                    "nationality_name",
                    *sofifa_rating_columns,
                },
                low_memory=False,
            )
            sofifa["available_from"] = pd.to_datetime(sofifa["available_from"], errors="coerce")
            sofifa["sofifa_id"] = pd.to_numeric(sofifa["sofifa_id"], errors="coerce")
            sofifa["dob_key"] = pd.to_datetime(sofifa["dob"], errors="coerce").dt.date
            sofifa["nationality_key"] = sofifa["nationality_name"].map(normalize_name)
            for col in sofifa_rating_columns:
                if col in sofifa.columns:
                    sofifa[col] = pd.to_numeric(sofifa[col], errors="coerce")
            sofifa = sofifa.dropna(subset=["sofifa_id", "available_from", "dob_key", "overall"]).copy()
            sofifa["sofifa_id"] = sofifa["sofifa_id"].astype(int)

            sofifa_candidates: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
            sofifa_candidates_by_dob: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            for row in sofifa.drop_duplicates(["sofifa_id", "long_name", "short_name", "dob_key"]).itertuples(index=False):
                for raw_name in (getattr(row, "long_name", ""), getattr(row, "short_name", "")):
                    player_key = normalize_person_name(raw_name)
                    if not player_key:
                        continue
                    candidate = {
                        "sofifa_id": int(row.sofifa_id),
                        "nationality_key": getattr(row, "nationality_key", ""),
                        "player_key": player_key,
                    }
                    sofifa_candidates[(player_key, row.dob_key)].append(candidate)
                    sofifa_candidates_by_dob[row.dob_key].append(candidate)

            def match_sofifa_player(row: pd.Series) -> pd.Series:
                candidates = sofifa_candidates.get((row["player_key"], row["dob_key"]), [])
                if not candidates:
                    fallback_candidates = sofifa_candidates_by_dob.get(row["dob_key"], [])
                    scored = []
                    for candidate in fallback_candidates:
                        score = person_name_match_score(row["player_key"], candidate["player_key"])
                        if candidate["nationality_key"] == row["team_key"]:
                            score += 0.03
                        if score >= 0.75:
                            scored.append((score, candidate))
                    if not scored:
                        return pd.Series({"sofifa_id": np.nan})
                    scored.sort(key=lambda item: item[0], reverse=True)
                    return pd.Series({"sofifa_id": scored[0][1]["sofifa_id"]})
                preferred = [candidate for candidate in candidates if candidate["nationality_key"] == row["team_key"]]
                candidate = preferred[0] if preferred else candidates[0]
                return pd.Series({"sofifa_id": candidate["sofifa_id"]})

            matched_sofifa = squads.apply(match_sofifa_player, axis=1)
            squads = pd.concat([squads.reset_index(drop=True), matched_sofifa.reset_index(drop=True)], axis=1)

            rating_cols_present = [col for col in sofifa_rating_columns if col in sofifa.columns]
            sofifa["date_ord"] = sofifa["available_from"].map(lambda value: value.toordinal())
            for sofifa_id, group in sofifa.sort_values("date_ord").groupby("sofifa_id", sort=False):
                sofifa_lookup[int(sofifa_id)] = (
                    group["date_ord"].to_numpy(dtype=np.int64),
                    group[rating_cols_present].reset_index(drop=True),
                )
            sofifa_enabled = True

    feature_cols = [
        "squad_player_count",
        "squad_avg_age",
        "squad_median_age",
        "squad_age_std",
        "squad_total_caps",
        "squad_avg_caps",
        "squad_top11_caps",
        "squad_total_goals",
        "squad_avg_goals",
        "squad_top5_goals",
        "squad_gk_count",
        "squad_df_count",
        "squad_mf_count",
        "squad_fw_count",
    ]
    market_feature_cols = [
        "squad_tm_matched_players",
        "squad_tm_match_rate",
        "squad_market_value_total_eur",
        "squad_market_value_avg_eur",
        "squad_market_value_top11_eur",
        "squad_market_value_top23_eur",
    ]
    if market_enabled:
        feature_cols.extend(market_feature_cols)
    sofifa_feature_cols = [
        "squad_sofifa_matched_players",
        "squad_sofifa_match_rate",
        "squad_sofifa_overall_avg",
        "squad_sofifa_overall_top11",
        "squad_sofifa_overall_top18",
        "squad_sofifa_overall_top23",
        "squad_sofifa_overall_top26",
        "squad_sofifa_potential_avg",
        "squad_sofifa_value_total_eur",
        "squad_sofifa_value_avg_eur",
        "squad_sofifa_value_top11_eur",
        "squad_sofifa_gk_overall_avg",
        "squad_sofifa_df_overall_avg",
        "squad_sofifa_mf_overall_avg",
        "squad_sofifa_fw_overall_avg",
        "squad_sofifa_attack_skill_avg",
        "squad_sofifa_midfield_skill_avg",
        "squad_sofifa_defensive_skill_avg",
        "squad_sofifa_keeper_skill_avg",
        "squad_sofifa_reactions_avg",
        "squad_sofifa_composure_avg",
    ]
    if sofifa_enabled:
        feature_cols.extend(sofifa_feature_cols)
    squad_lookup: dict[tuple[str, int, str], dict[str, float]] = {}
    squad_group_lookup: dict[tuple[str, int, str], pd.DataFrame] = {}
    for key, group in squads.groupby(["competition_key", "squad_year", "team_key"], dropna=False):
        squad_group_lookup[key] = group.copy()
        caps = group["caps"].to_numpy(dtype=float)
        goals = group["goals"].to_numpy(dtype=float)
        positions = group["position_key"].value_counts()
        squad_lookup[key] = {
            "squad_player_count": float(len(group)),
            "squad_avg_age": float(group["age"].mean()) if group["age"].notna().any() else np.nan,
            "squad_median_age": float(group["age"].median()) if group["age"].notna().any() else np.nan,
            "squad_age_std": float(group["age"].std(ddof=0)) if group["age"].notna().sum() > 1 else 0.0,
            "squad_total_caps": float(np.nansum(caps)),
            "squad_avg_caps": float(np.nanmean(caps)) if len(caps) else 0.0,
            "squad_top11_caps": float(np.nansum(np.sort(caps)[-11:])),
            "squad_total_goals": float(np.nansum(goals)),
            "squad_avg_goals": float(np.nanmean(goals)) if len(goals) else 0.0,
            "squad_top5_goals": float(np.nansum(np.sort(goals)[-5:])),
            "squad_gk_count": float(positions.get("gk", 0)),
            "squad_df_count": float(positions.get("df", 0)),
            "squad_mf_count": float(positions.get("mf", 0)),
            "squad_fw_count": float(positions.get("fw", 0)),
        }

    market_cache: dict[tuple[tuple[str, int, str], int], dict[str, float]] = {}
    sofifa_cache: dict[tuple[tuple[str, int, str], int], dict[str, float]] = {}

    def market_features_for_squad(squad_key: tuple[str, int, str], asof_date: pd.Timestamp) -> dict[str, float]:
        if not market_enabled or pd.isna(asof_date):
            return {}
        asof_ord = int(pd.Timestamp(asof_date).toordinal())
        cache_key = (squad_key, asof_ord)
        if cache_key in market_cache:
            return market_cache[cache_key]
        group = squad_group_lookup.get(squad_key)
        if group is None or "tm_player_id" not in group.columns:
            values = np.array([], dtype=float)
            matched_players = 0.0
            player_count = float(len(group)) if group is not None else 0.0
        else:
            historical_values = []
            player_ids = pd.to_numeric(group["tm_player_id"], errors="coerce").dropna().astype(int).tolist()
            matched_players = float(len(player_ids))
            player_count = float(len(group))
            for player_id in player_ids:
                valuation_data = valuation_lookup.get(player_id)
                if valuation_data is None:
                    historical_values.append(np.nan)
                    continue
                ordinals, values_for_player = valuation_data
                index = int(np.searchsorted(ordinals, asof_ord, side="right")) - 1
                historical_values.append(values_for_player[index] if index >= 0 else np.nan)
            values = np.array(historical_values, dtype=float)
        valid_values = values[~np.isnan(values)]
        features = {
            "squad_tm_matched_players": matched_players,
            "squad_tm_match_rate": safe_rate(matched_players, player_count),
            "squad_market_value_total_eur": float(np.nansum(valid_values)) if len(valid_values) else 0.0,
            "squad_market_value_avg_eur": float(np.nanmean(valid_values)) if len(valid_values) else 0.0,
            "squad_market_value_top11_eur": float(np.nansum(np.sort(valid_values)[-11:])) if len(valid_values) else 0.0,
            "squad_market_value_top23_eur": float(np.nansum(np.sort(valid_values)[-23:])) if len(valid_values) else 0.0,
        }
        market_cache[cache_key] = features
        return features

    def sofifa_features_for_squad(squad_key: tuple[str, int, str], asof_date: pd.Timestamp) -> dict[str, float]:
        if not sofifa_enabled or pd.isna(asof_date):
            return {}
        asof_ord = int(pd.Timestamp(asof_date).toordinal())
        cache_key = (squad_key, asof_ord)
        if cache_key in sofifa_cache:
            return sofifa_cache[cache_key]
        group = squad_group_lookup.get(squad_key)
        if group is None or "sofifa_id" not in group.columns:
            sofifa_rows = pd.DataFrame()
            matched_players = 0.0
            player_count = float(len(group)) if group is not None else 0.0
            positions = pd.Series(dtype=object)
        else:
            rating_rows = []
            positions = group["position_key"].reset_index(drop=True)
            player_count = float(len(group))
            for row_index, sofifa_id_value in enumerate(pd.to_numeric(group["sofifa_id"], errors="coerce")):
                if pd.isna(sofifa_id_value):
                    continue
                rating_data = sofifa_lookup.get(int(sofifa_id_value))
                if rating_data is None:
                    continue
                ordinals, ratings_for_player = rating_data
                index = int(np.searchsorted(ordinals, asof_ord, side="right")) - 1
                if index < 0:
                    continue
                item = ratings_for_player.iloc[index].copy()
                item["position_key"] = positions.iloc[row_index] if row_index < len(positions) else "other"
                rating_rows.append(item)
            sofifa_rows = pd.DataFrame(rating_rows)
            matched_players = float(len(sofifa_rows))
        if sofifa_rows.empty:
            features = {col: 0.0 for col in sofifa_feature_cols}
            features["squad_sofifa_match_rate"] = safe_rate(matched_players, player_count)
            sofifa_cache[cache_key] = features
            return features

        def top_sum(column: str, n: int) -> float:
            values = pd.to_numeric(sofifa_rows[column], errors="coerce").dropna().to_numpy(dtype=float)
            return float(np.sum(np.sort(values)[-n:])) if len(values) else 0.0

        overall = pd.to_numeric(sofifa_rows["overall"], errors="coerce")
        potential = pd.to_numeric(sofifa_rows.get("potential", np.nan), errors="coerce")
        value_eur = pd.to_numeric(sofifa_rows.get("value_eur", np.nan), errors="coerce")
        features = {
            "squad_sofifa_matched_players": matched_players,
            "squad_sofifa_match_rate": safe_rate(matched_players, player_count),
            "squad_sofifa_overall_avg": float(overall.mean()) if overall.notna().any() else 0.0,
            "squad_sofifa_overall_top11": top_sum("overall", 11),
            "squad_sofifa_overall_top18": top_sum("overall", 18),
            "squad_sofifa_overall_top23": top_sum("overall", 23),
            "squad_sofifa_overall_top26": top_sum("overall", 26),
            "squad_sofifa_potential_avg": float(potential.mean()) if potential.notna().any() else 0.0,
            "squad_sofifa_value_total_eur": float(value_eur.sum()) if value_eur.notna().any() else 0.0,
            "squad_sofifa_value_avg_eur": float(value_eur.mean()) if value_eur.notna().any() else 0.0,
            "squad_sofifa_value_top11_eur": top_sum("value_eur", 11) if "value_eur" in sofifa_rows.columns else 0.0,
        }
        for pos_key in ("gk", "df", "mf", "fw"):
            pos_overall = pd.to_numeric(
                sofifa_rows.loc[sofifa_rows["position_key"].eq(pos_key), "overall"],
                errors="coerce",
            )
            features[f"squad_sofifa_{pos_key}_overall_avg"] = (
                float(pos_overall.mean()) if pos_overall.notna().any() else 0.0
            )
        attack_cols = [col for col in ("shooting", "pace", "dribbling") if col in sofifa_rows.columns]
        midfield_cols = [col for col in ("passing", "dribbling", "physic") if col in sofifa_rows.columns]
        defensive_cols = [col for col in ("defending", "physic") if col in sofifa_rows.columns]
        keeper_cols = [col for col in ("overall",) if col in sofifa_rows.columns]
        skill_groups = {
            "attack": attack_cols,
            "midfield": midfield_cols,
            "defensive": defensive_cols,
            "keeper": keeper_cols,
        }
        for skill_name, columns in skill_groups.items():
            values = sofifa_rows[columns].apply(pd.to_numeric, errors="coerce") if columns else pd.DataFrame()
            features[f"squad_sofifa_{skill_name}_skill_avg"] = (
                float(values.mean(axis=1).mean()) if not values.empty else 0.0
            )
        reactions = pd.to_numeric(sofifa_rows.get("movement_reactions", np.nan), errors="coerce")
        composure = pd.to_numeric(sofifa_rows.get("mentality_composure", np.nan), errors="coerce")
        features["squad_sofifa_reactions_avg"] = float(reactions.mean()) if reactions.notna().any() else 0.0
        features["squad_sofifa_composure_avg"] = float(composure.mean()) if composure.notna().any() else 0.0
        sofifa_cache[cache_key] = features
        return features

    output = matches.copy()
    match_competition = output["tournament"].map(squad_competition_key)
    match_dates = pd.to_datetime(output["date"], errors="coerce")
    match_year = match_dates.dt.year
    tournament_start_dates = (
        pd.DataFrame({"competition_key": match_competition, "year": match_year, "date": match_dates})
        .dropna(subset=["date", "year"])
        .query("competition_key != ''")
        .groupby(["competition_key", "year"], dropna=False)["date"]
        .min()
        .to_dict()
    )

    for side in ("home", "away"):
        for col in feature_cols:
            output[f"{side}_{col}"] = np.nan
        for row_id, competition_key, year, team_name in zip(
            output.index,
            match_competition,
            match_year,
            output[f"{side}_team"],
        ):
            if not competition_key or pd.isna(year):
                continue
            team_key = normalize_name(team_name)
            if not team_key:
                continue
            year_int = int(year)
            squad_key = (competition_key, year_int, team_key)
            features = squad_lookup.get(squad_key)
            if features is None:
                squad_key = (competition_key, year_int - 1, team_key)
                features = squad_lookup.get(squad_key)
            if features is None:
                continue
            start_date = tournament_start_dates.get((competition_key, float(year_int)))
            if start_date is None:
                start_date = tournament_start_dates.get((competition_key, year_int))
            if market_enabled and start_date is not None:
                features = {
                    **features,
                    **market_features_for_squad(squad_key, pd.Timestamp(start_date) - pd.Timedelta(days=1)),
                }
            if sofifa_enabled and start_date is not None:
                features = {
                    **features,
                    **sofifa_features_for_squad(squad_key, pd.Timestamp(start_date) - pd.Timedelta(days=1)),
                }
            for col, value in features.items():
                output.at[row_id, f"{side}_{col}"] = value

    diff_features = {
        f"{col}_diff": output[f"home_{col}"].fillna(0.0) - output[f"away_{col}"].fillna(0.0)
        for col in feature_cols
    }
    output = pd.concat([output, pd.DataFrame(diff_features, index=output.index)], axis=1)

    matched = int(
        (
            output["home_squad_player_count"].fillna(0.0).gt(0)
            | output["away_squad_player_count"].fillna(0.0).gt(0)
        ).sum()
    )
    return output, matched


def attach_soccerbase_lineup_features(
    matches: pd.DataFrame,
    lineups_path: Path | None,
    use_player_ratings: bool,
    yearly_ratings_path: Path | None = None,
) -> tuple[pd.DataFrame, int]:
    if lineups_path is None:
        return matches, 0
    lineups_path = Path(lineups_path)
    if not lineups_path.exists():
        raise FileNotFoundError(f"Soccerbase lineups CSV not found: {lineups_path}")

    usecols = [
        "soccerbase_game_id",
        "date",
        "home_team",
        "away_team",
        "team",
        "player_id",
        "player_name",
        "is_starter",
        "is_sub_used",
        "soccerbase_position",
        "kaggle_player_id",
        "kaggle_overall_rating",
        "kaggle_positions",
    ]
    lineups = pd.read_csv(lineups_path, usecols=lambda col: col in usecols, low_memory=False)
    required = {"date", "home_team", "away_team", "team", "is_starter", "is_sub_used"}
    missing = required - set(lineups.columns)
    if missing:
        raise ValueError(f"Soccerbase lineups CSV is missing required columns: {sorted(missing)}")

    lineups["is_starter"] = pd.to_numeric(lineups["is_starter"], errors="coerce").fillna(0.0)
    lineups["is_sub_used"] = pd.to_numeric(lineups["is_sub_used"], errors="coerce").fillna(0.0)
    lineups = lineups[lineups["is_starter"].gt(0)].copy()
    if lineups.empty:
        return matches, 0
    lineups["source_order"] = np.arange(len(lineups), dtype=np.int64)

    lineups["date"] = pd.to_datetime(lineups["date"], errors="coerce")
    lineups["date_key"] = lineups["date"].dt.date.astype(str)
    lineups["home_key"] = lineups["home_team"].map(normalize_name)
    lineups["away_key"] = lineups["away_team"].map(normalize_name)
    lineups["pair_key"] = [team_pair_key(home, away) for home, away in zip(lineups["home_key"], lineups["away_key"])]
    lineups["team_key"] = lineups["team"].map(normalize_name)
    dedupe_player_col = "player_id" if "player_id" in lineups.columns else "player_name"
    lineups = lineups.drop_duplicates(
        subset=["date_key", "pair_key", "team_key", dedupe_player_col, "is_starter", "is_sub_used"],
        keep="first",
    )
    rating_columns = ["overall"]
    lineups["overall_rating"] = np.nan
    if yearly_ratings_path is not None:
        yearly_ratings_path = Path(yearly_ratings_path)
        if not yearly_ratings_path.exists():
            raise FileNotFoundError(f"SoFIFA yearly ratings CSV not found: {yearly_ratings_path}")
        yearly_rating_columns = [
            "overall",
            "pace",
            "shooting",
            "passing",
            "dribbling",
            "defending",
            "physic",
            "movement_reactions",
            "mentality_composure",
            "attacking_finishing",
            "attacking_short_passing",
        ]
        ratings = pd.read_csv(
            yearly_ratings_path,
            usecols=lambda col: col in {"sofifa_id", "available_from", *yearly_rating_columns},
        )
        ratings["available_from"] = pd.to_datetime(ratings["available_from"], errors="coerce")
        ratings["sofifa_id"] = pd.to_numeric(ratings["sofifa_id"], errors="coerce")
        ratings = ratings.dropna(subset=["available_from", "sofifa_id", "overall"]).copy()
        ratings["sofifa_id"] = ratings["sofifa_id"].astype(int)
        for col in yearly_rating_columns:
            if col in ratings.columns:
                ratings[col] = pd.to_numeric(ratings[col], errors="coerce")
        lineups["sofifa_id"] = pd.to_numeric(
            lineups["kaggle_player_id"] if "kaggle_player_id" in lineups.columns else np.nan,
            errors="coerce",
        )
        lineups["match_order"] = np.arange(len(lineups), dtype=np.int64)
        lineups = lineups.set_index("match_order", drop=False)
        enriched_parts = []
        for sofifa_id, player_rows in lineups[lineups["sofifa_id"].notna()].groupby("sofifa_id", sort=False):
            player_ratings = ratings[ratings["sofifa_id"].eq(int(sofifa_id))].sort_values("available_from")
            if player_ratings.empty:
                continue
            merged = pd.merge_asof(
                player_rows.sort_values("date"),
                player_ratings[["available_from"] + yearly_rating_columns],
                left_on="date",
                right_on="available_from",
                direction="backward",
            )
            enriched_parts.append(merged[["match_order"] + yearly_rating_columns])
        if enriched_parts:
            enriched = pd.concat(enriched_parts, ignore_index=True).set_index("match_order")
            for col in yearly_rating_columns:
                lineups.loc[enriched.index, f"{col}_rating"] = enriched[col]
        lineups["overall_rating"] = pd.to_numeric(lineups.get("overall_rating"), errors="coerce")
        if "overall_rating" not in lineups.columns or lineups["overall_rating"].isna().all():
            lineups["overall_rating"] = lineups.get("overall_rating", np.nan)
        for col in yearly_rating_columns:
            source_col = f"{col}_rating"
            if source_col in lineups.columns:
                lineups[source_col] = pd.to_numeric(lineups[source_col], errors="coerce")
        rating_columns = yearly_rating_columns
    elif "kaggle_overall_rating" in lineups.columns and use_player_ratings:
        lineups["overall_rating"] = pd.to_numeric(lineups["kaggle_overall_rating"], errors="coerce")
    if "overall_rating" not in lineups.columns:
        lineups["overall_rating"] = np.nan
    lineups["position_group"] = (
        lineups["kaggle_positions"].map(lineup_position_group)
        if "kaggle_positions" in lineups.columns
        else lineups.get("soccerbase_position", "other").map(lineup_position_group)
    )
    lineups = lineups[(lineups["team_key"] != "") & (lineups["home_key"] != "") & (lineups["away_key"] != "")]

    base_feature_cols = [
        "lineup_player_count",
        "lineup_starter_count",
        "lineup_used_sub_count",
        "lineup_gk_count",
        "lineup_df_count",
        "lineup_mf_count",
        "lineup_fw_count",
        "lineup_outfield_count",
        "lineup_defensive_share",
        "lineup_midfield_share",
        "lineup_attacking_share",
        "lineup_defense_attack_balance",
        "lineup_df_fw_ratio",
    ]
    rating_feature_cols = [
        "lineup_rating_count",
        "lineup_rating_coverage",
        "lineup_overall_avg",
        "lineup_overall_median",
        "lineup_overall_min",
        "lineup_overall_max",
        "lineup_overall_std",
        "lineup_overall_top5",
        "lineup_overall_top11",
        "lineup_gk_overall_avg",
        "lineup_df_overall_avg",
        "lineup_mf_overall_avg",
        "lineup_fw_overall_avg",
    ]
    yearly_rating_feature_cols = []
    if yearly_ratings_path is not None:
        for rating_col in rating_columns:
            if rating_col == "overall":
                continue
            yearly_rating_feature_cols.append(f"lineup_{rating_col}_avg")
            yearly_rating_feature_cols.append(f"lineup_{rating_col}_top11")
    feature_cols = base_feature_cols + (rating_feature_cols if use_player_ratings else [])
    feature_cols += yearly_rating_feature_cols

    rows = []
    group_cols = ["date_key", "pair_key", "team_key"]
    for key, group in lineups.groupby(group_cols, dropna=False):
        group = group.sort_values("source_order")
        if len(group) > 11:
            group = group.head(11)
        positions = group["position_group"].value_counts()
        outfield_count = float(positions.get("df", 0) + positions.get("mf", 0) + positions.get("fw", 0))
        row = {
            "date_key": key[0],
            "pair_key": key[1],
            "team_key": key[2],
            "lineup_player_count": float(len(group)),
            "lineup_starter_count": float(group["is_starter"].sum()),
            "lineup_used_sub_count": float(group["is_sub_used"].sum()),
            "lineup_gk_count": float(positions.get("gk", 0)),
            "lineup_df_count": float(positions.get("df", 0)),
            "lineup_mf_count": float(positions.get("mf", 0)),
            "lineup_fw_count": float(positions.get("fw", 0)),
            "lineup_outfield_count": outfield_count,
            "lineup_defensive_share": safe_rate(float(positions.get("df", 0)), outfield_count),
            "lineup_midfield_share": safe_rate(float(positions.get("mf", 0)), outfield_count),
            "lineup_attacking_share": safe_rate(float(positions.get("fw", 0)), outfield_count),
            "lineup_defense_attack_balance": float(positions.get("df", 0) - positions.get("fw", 0)),
            "lineup_df_fw_ratio": safe_rate(float(positions.get("df", 0)), float(positions.get("fw", 0))),
        }
        if use_player_ratings:
            ratings = group["overall_rating"].dropna().to_numpy(dtype=float)
            row.update(
                {
                    "lineup_rating_count": float(len(ratings)),
                    "lineup_rating_coverage": safe_rate(float(len(ratings)), float(len(group))),
                    "lineup_overall_avg": float(np.nanmean(ratings)) if len(ratings) else np.nan,
                    "lineup_overall_median": float(np.nanmedian(ratings)) if len(ratings) else np.nan,
                    "lineup_overall_min": float(np.nanmin(ratings)) if len(ratings) else np.nan,
                    "lineup_overall_max": float(np.nanmax(ratings)) if len(ratings) else np.nan,
                    "lineup_overall_std": float(np.nanstd(ratings)) if len(ratings) > 1 else 0.0,
                    "lineup_overall_top5": float(np.nansum(np.sort(ratings)[-5:])) if len(ratings) else 0.0,
                    "lineup_overall_top11": float(np.nansum(np.sort(ratings)[-11:])) if len(ratings) else 0.0,
                }
            )
            for position in ("gk", "df", "mf", "fw"):
                position_ratings = group.loc[group["position_group"].eq(position), "overall_rating"].dropna()
                row[f"lineup_{position}_overall_avg"] = (
                    float(position_ratings.mean()) if len(position_ratings) else np.nan
                )
            if yearly_ratings_path is not None:
                for rating_col in rating_columns:
                    if rating_col == "overall":
                        continue
                    source_col = f"{rating_col}_rating"
                    values = group[source_col].dropna().to_numpy(dtype=float) if source_col in group.columns else []
                    row[f"lineup_{rating_col}_avg"] = float(np.nanmean(values)) if len(values) else np.nan
                    row[f"lineup_{rating_col}_top11"] = (
                        float(np.nansum(np.sort(values)[-11:])) if len(values) else 0.0
                    )
        rows.append(row)

    side_features = pd.DataFrame(rows)
    output = matches.copy()
    output["date_key"] = pd.to_datetime(output["date"], errors="coerce").dt.date.astype(str)
    output["home_key"] = output["home_team"].map(normalize_name)
    output["away_key"] = output["away_team"].map(normalize_name)
    output["pair_key"] = [team_pair_key(home, away) for home, away in zip(output["home_key"], output["away_key"])]

    for side in ("home", "away"):
        team_key_col = f"{side}_key"
        lookup = output[["date_key", "pair_key", team_key_col]].rename(columns={team_key_col: "team_key"})
        lookup["row_id"] = output.index
        merged = lookup.merge(side_features, on=["date_key", "pair_key", "team_key"], how="left").set_index("row_id")
        for col in feature_cols:
            output[f"{side}_{col}"] = pd.to_numeric(merged[col], errors="coerce") if col in merged.columns else np.nan

    for col in feature_cols:
        output[f"{col}_diff"] = output[f"home_{col}"].fillna(0.0) - output[f"away_{col}"].fillna(0.0)

    matched = int(
        (
            output["home_lineup_player_count"].fillna(0.0).gt(0)
            | output["away_lineup_player_count"].fillna(0.0).gt(0)
        ).sum()
    )
    return output.drop(columns=["date_key", "home_key", "away_key", "pair_key"]), matched


def attach_soccerbase_rolling_stat_features(
    matches: pd.DataFrame,
    stats_path: Path | None,
    window_days: int,
    include_recency_features: bool = False,
) -> tuple[pd.DataFrame, int]:
    if stats_path is None:
        return matches, 0
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Soccerbase match stats CSV not found: {stats_path}")

    needed = [
        "date",
        "home_team",
        "away_team",
        "home_possession",
        "away_possession",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_shots_off_target",
        "away_shots_off_target",
        "home_corners",
        "away_corners",
    ]
    stats = pd.read_csv(stats_path, usecols=lambda col: col in needed, low_memory=False)
    missing = {"date", "home_team", "away_team"} - set(stats.columns)
    if missing:
        raise ValueError(f"Soccerbase match stats CSV is missing required columns: {sorted(missing)}")

    for col in needed:
        if col.startswith(("home_", "away_")) and col not in {"home_team", "away_team"} and col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors="coerce")

    metric_map = {
        "possession": ("home_possession", "away_possession"),
        "shots_on_target": ("home_shots_on_target", "away_shots_on_target"),
        "shots_off_target": ("home_shots_off_target", "away_shots_off_target"),
        "corners": ("home_corners", "away_corners"),
    }
    stat_value_cols = [col for pair in metric_map.values() for col in pair if col in stats.columns]
    stats = stats[pd.to_datetime(stats["date"], errors="coerce").notna()].copy()
    stats["date"] = pd.to_datetime(stats["date"], errors="coerce")
    stats = stats[stats[stat_value_cols].notna().any(axis=1)].copy()
    if stats.empty:
        return matches, 0

    rows = []
    for source_side, opponent_side, role in (("home", "away", "home"), ("away", "home", "away")):
        side = pd.DataFrame(
            {
                "date": stats["date"],
                "team_key": stats[f"{source_side}_team"].map(normalize_name),
                "role": role,
            }
        )
        for metric, (home_col, away_col) in metric_map.items():
            value_col = home_col if source_side == "home" else away_col
            against_col = away_col if source_side == "home" else home_col
            value = pd.to_numeric(stats[value_col], errors="coerce") if value_col in stats.columns else np.nan
            against = pd.to_numeric(stats[against_col], errors="coerce") if against_col in stats.columns else np.nan
            side[f"{metric}_for_sum"] = value.fillna(0.0)
            side[f"{metric}_for_count"] = value.notna().astype(float)
            side[f"{metric}_against_sum"] = against.fillna(0.0)
            side[f"{metric}_against_count"] = against.notna().astype(float)

        shots_for = side["shots_on_target_for_sum"] + side["shots_off_target_for_sum"]
        shots_for_count = (
            side["shots_on_target_for_count"].gt(0) | side["shots_off_target_for_count"].gt(0)
        ).astype(float)
        shots_against = side["shots_on_target_against_sum"] + side["shots_off_target_against_sum"]
        shots_against_count = (
            side["shots_on_target_against_count"].gt(0) | side["shots_off_target_against_count"].gt(0)
        ).astype(float)
        side["total_shots_for_sum"] = shots_for
        side["total_shots_for_count"] = shots_for_count
        side["total_shots_against_sum"] = shots_against
        side["total_shots_against_count"] = shots_against_count
        side["stat_match"] = 1.0
        rows.append(side)

    team_stats = pd.concat(rows, ignore_index=True)
    team_stats = team_stats[(team_stats["team_key"] != "") & team_stats["date"].notna()].copy()
    value_columns = [col for col in team_stats.columns if col.endswith(("_sum", "_count"))] + ["stat_match"]
    overall_lookup = RollingStatsLookup(team_stats, ["team_key"], value_columns, half_life_days=max(window_days, 1))
    role_lookup = RollingStatsLookup(team_stats, ["team_key", "role"], value_columns, half_life_days=max(window_days, 1))

    metrics = ["possession", "shots_on_target", "shots_off_target", "total_shots", "corners"]
    feature_cols = []
    for scope in ("recent", "all"):
        for role_scope in ("overall", "role"):
            feature_cols.append(f"sb_stats_{scope}_{role_scope}_matches")
            if include_recency_features:
                feature_cols.append(f"sb_stats_{scope}_{role_scope}_recency_matches")
            for metric in metrics:
                for direction in ("for", "against"):
                    feature_cols.append(f"sb_stats_{scope}_{role_scope}_{metric}_{direction}_avg")
                    if include_recency_features:
                        feature_cols.append(f"sb_stats_{scope}_{role_scope}_recency_{metric}_{direction}_avg")

    def summarize(stats_values: dict[str, float]) -> dict[str, float]:
        output: dict[str, float] = {"matches": float(stats_values.get("stat_match", 0.0))}
        if include_recency_features:
            output["recency_matches"] = float(stats_values.get("recency_stat_match", 0.0))
        for metric in metrics:
            for direction in ("for", "against"):
                total = float(stats_values.get(f"{metric}_{direction}_sum", 0.0))
                count = float(stats_values.get(f"{metric}_{direction}_count", 0.0))
                output[f"{metric}_{direction}_avg"] = safe_rate(total, count, np.nan)
                if include_recency_features:
                    recency_total = float(stats_values.get(f"recency_{metric}_{direction}_sum", 0.0))
                    recency_count = float(stats_values.get(f"recency_{metric}_{direction}_count", 0.0))
                    output[f"recency_{metric}_{direction}_avg"] = safe_rate(recency_total, recency_count, np.nan)
        return output

    output = matches.copy()
    feature_arrays: dict[str, np.ndarray] = {}
    for side in ("home", "away"):
        side_arrays = {
            f"{side}_{col}": np.full(len(output), np.nan, dtype=np.float64)
            for col in feature_cols
        }

        for position, row in enumerate(output[["date", f"{side}_team"]].itertuples(index=False)):
            match_date = pd.Timestamp(row.date)
            team_key = normalize_name(getattr(row, f"{side}_team"))
            recent_overall = summarize(overall_lookup.query(team_key, match_date, int(window_days)))
            recent_role = summarize(role_lookup.query((team_key, side), match_date, int(window_days)))
            all_overall = summarize(overall_lookup.query(team_key, match_date, 36500))
            all_role = summarize(role_lookup.query((team_key, side), match_date, 36500))
            groups = {
                "recent_overall": recent_overall,
                "recent_role": recent_role,
                "all_overall": all_overall,
                "all_role": all_role,
            }
            for group_name, values in groups.items():
                for key, value in values.items():
                    side_arrays[f"{side}_sb_stats_{group_name}_{key}"][position] = value
        feature_arrays.update(side_arrays)

    output = pd.concat([output, pd.DataFrame(feature_arrays, index=output.index)], axis=1)

    diff_features = {
        f"{col}_diff": output[f"home_{col}"].fillna(0.0) - output[f"away_{col}"].fillna(0.0)
        for col in feature_cols
    }
    output = pd.concat([output, pd.DataFrame(diff_features, index=output.index)], axis=1)

    matched = int(
        (
            output["home_sb_stats_recent_overall_matches"].fillna(0.0).gt(0)
            | output["away_sb_stats_recent_overall_matches"].fillna(0.0).gt(0)
        ).sum()
    )
    return output, matched


def attach_soccerbase_card_features(
    matches: pd.DataFrame,
    cards_path: Path | None,
    window_days: int,
    last_n: int,
) -> tuple[pd.DataFrame, int]:
    if cards_path is None:
        return matches, 0
    cards_path = Path(cards_path)
    if not cards_path.exists():
        raise FileNotFoundError(f"Soccerbase cards CSV not found: {cards_path}")

    needed = ["date", "home_team", "away_team", "team", "card_type", "player_id"]
    cards = pd.read_csv(cards_path, usecols=lambda col: col in needed, low_memory=False)
    missing = {"date", "home_team", "away_team", "team", "card_type"} - set(cards.columns)
    if missing:
        raise ValueError(f"Soccerbase cards CSV is missing required columns: {sorted(missing)}")

    cards["date"] = pd.to_datetime(cards["date"], errors="coerce")
    cards["home_key"] = cards["home_team"].map(normalize_name)
    cards["away_key"] = cards["away_team"].map(normalize_name)
    cards["team_key"] = cards["team"].map(normalize_name)
    cards["role"] = np.where(cards["team_key"].eq(cards["home_key"]), "home", "away")
    cards["card_type"] = cards["card_type"].astype(str).str.lower().str.strip()
    cards = cards[
        cards["date"].notna()
        & cards["team_key"].ne("")
        & cards["role"].isin(["home", "away"])
        & cards["card_type"].isin(["yellow", "red"])
    ].copy()
    if cards.empty:
        return matches, 0

    card_match_cols = ["date", "home_key", "away_key", "team_key", "role"]
    cards["yellow_cards"] = cards["card_type"].eq("yellow").astype(float)
    cards["red_cards"] = cards["card_type"].eq("red").astype(float)
    cards["card_points"] = cards["yellow_cards"] + 2.0 * cards["red_cards"]
    if "player_id" in cards.columns:
        cards["player_card_key"] = cards["player_id"].astype(str)
    else:
        cards["player_card_key"] = np.arange(len(cards)).astype(str)

    per_match = (
        cards.groupby(card_match_cols, dropna=False)
        .agg(
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
            card_points=("card_points", "sum"),
            players_carded=("player_card_key", "nunique"),
        )
        .reset_index()
    )
    per_match["card_match"] = 1.0

    value_columns = ["yellow_cards", "red_cards", "card_points", "players_carded", "card_match"]
    overall_lookup = RollingStatsLookup(per_match, ["team_key"], value_columns, half_life_days=max(window_days, 1))
    role_lookup = RollingStatsLookup(
        per_match,
        ["team_key", "role"],
        value_columns,
        half_life_days=max(window_days, 1),
    )

    scopes = ("recent", "last5", "all")
    metrics = ("yellow_cards", "red_cards", "card_points", "players_carded")
    feature_cols = []
    for scope in scopes:
        for role_scope in ("overall", "role"):
            feature_cols.append(f"sb_cards_{scope}_{role_scope}_matches")
            for metric in metrics:
                feature_cols.append(f"sb_cards_{scope}_{role_scope}_{metric}_per_match")

    def summarize(values: dict[str, float]) -> dict[str, float]:
        matches_count = float(values.get("card_match", 0.0))
        output = {"matches": matches_count}
        for metric in metrics:
            output[f"{metric}_per_match"] = safe_rate(float(values.get(metric, 0.0)), matches_count)
        return output

    output = matches.copy()
    feature_arrays: dict[str, np.ndarray] = {}
    for side in ("home", "away"):
        side_arrays = {
            f"{side}_{col}": np.full(len(output), np.nan, dtype=np.float64)
            for col in feature_cols
        }
        for position, row in enumerate(output[["date", f"{side}_team"]].itertuples(index=False)):
            match_date = pd.Timestamp(row.date)
            team_key = normalize_name(getattr(row, f"{side}_team"))
            groups = {
                "recent_overall": summarize(overall_lookup.query(team_key, match_date, int(window_days))),
                "recent_role": summarize(role_lookup.query((team_key, side), match_date, int(window_days))),
                "last5_overall": summarize(overall_lookup.query_last_n(team_key, match_date, int(last_n))),
                "last5_role": summarize(role_lookup.query_last_n((team_key, side), match_date, int(last_n))),
                "all_overall": summarize(overall_lookup.query(team_key, match_date, 36500)),
                "all_role": summarize(role_lookup.query((team_key, side), match_date, 36500)),
            }
            for group_name, values in groups.items():
                for key, value in values.items():
                    side_arrays[f"{side}_sb_cards_{group_name}_{key}"][position] = value
        feature_arrays.update(side_arrays)

    output = pd.concat([output, pd.DataFrame(feature_arrays, index=output.index)], axis=1)
    diff_features = {
        f"{col}_diff": output[f"home_{col}"].fillna(0.0) - output[f"away_{col}"].fillna(0.0)
        for col in feature_cols
    }
    output = pd.concat([output, pd.DataFrame(diff_features, index=output.index)], axis=1)

    matched = int(
        (
            output["home_sb_cards_recent_overall_matches"].fillna(0.0).gt(0)
            | output["away_sb_cards_recent_overall_matches"].fillna(0.0).gt(0)
        ).sum()
    )
    return output, matched


def transfermarkt_competition_weight(row: pd.Series) -> float:
    competition_id = str(row.get("competition_id", "")).upper()
    competition_type = str(row.get("type", "")).lower()
    sub_type = str(row.get("sub_type", "")).lower()

    if competition_id in {"CL"}:
        return 1.45
    if competition_id in {"GB1", "ES1", "IT1", "L1", "FR1"}:
        return 1.35
    if competition_id in {"EL"}:
        return 1.25
    if competition_type == "national_team_competition":
        return 1.25
    if competition_type == "international_cup":
        return 1.20
    if competition_type == "domestic_league" and sub_type == "first_tier":
        return 1.10
    if competition_type == "domestic_league":
        return 1.00
    if competition_type == "domestic_cup":
        return 0.85
    return 0.75


def position_group(value: Any) -> str:
    text = str(value).strip().lower()
    if "goalkeeper" in text:
        return "goalkeeper"
    if "back" in text or "defender" in text or "defence" in text:
        return "defender"
    if "midfield" in text:
        return "midfield"
    if "winger" in text or "forward" in text or "striker" in text or "attack" in text:
        return "attack"
    return "midfield"


class RollingStatsLookup:
    def __init__(
        self,
        daily: pd.DataFrame,
        key_columns: list[str],
        value_columns: list[str],
        half_life_days: float,
    ) -> None:
        self.key_columns = key_columns
        self.value_columns = value_columns
        self.half_life_days = float(half_life_days)
        self.reference_day = int(pd.Timestamp("2000-01-01").toordinal())
        self.data: dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        if daily.empty:
            return

        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily = daily.dropna(subset=["date"])
        daily["date_ord"] = daily["date"].map(lambda value: value.toordinal())

        group_key: str | list[str] = key_columns[0] if len(key_columns) == 1 else key_columns
        for key, group in daily.groupby(group_key, dropna=False):
            group = group.sort_values("date_ord")
            ordinals = group["date_ord"].to_numpy(dtype=np.int64)
            values = group[value_columns].to_numpy(dtype=np.float64)
            prefix = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
            age_factor = np.exp((ordinals - self.reference_day) / self.half_life_days)[:, None]
            prefix_exp = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values * age_factor, axis=0)])
            self.data[key] = (ordinals, prefix, prefix_exp)

    def query(self, key: Any, match_date: pd.Timestamp, window_days: int) -> dict[str, float]:
        empty = {col: 0.0 for col in self.value_columns}
        empty.update({f"recency_{col}": 0.0 for col in self.value_columns})
        if key not in self.data or pd.isna(match_date):
            return empty

        date_ord = int(pd.Timestamp(match_date).toordinal())
        window_start = date_ord - int(window_days)
        ordinals, prefix, prefix_exp = self.data[key]
        left = int(np.searchsorted(ordinals, window_start, side="left"))
        right = int(np.searchsorted(ordinals, date_ord, side="left"))
        if right <= left:
            return empty

        totals = prefix[right] - prefix[left]
        recency_totals = (prefix_exp[right] - prefix_exp[left]) * math.exp(
            -(date_ord - self.reference_day) / self.half_life_days
        )
        output = {col: float(value) for col, value in zip(self.value_columns, totals)}
        output.update({f"recency_{col}": float(value) for col, value in zip(self.value_columns, recency_totals)})
        return output

    def query_last_n(self, key: Any, match_date: pd.Timestamp, n: int) -> dict[str, float]:
        empty = {col: 0.0 for col in self.value_columns}
        if key not in self.data or pd.isna(match_date) or n <= 0:
            return empty

        date_ord = int(pd.Timestamp(match_date).toordinal())
        ordinals, prefix, _ = self.data[key]
        right = int(np.searchsorted(ordinals, date_ord, side="left"))
        left = max(0, right - int(n))
        if right <= left:
            return empty
        totals = prefix[right] - prefix[left]
        return {col: float(value) for col, value in zip(self.value_columns, totals)}


def build_transfermarkt_daily_stats(transfermarkt_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["appearances.csv", "players.csv", "competitions.csv", "game_lineups.csv"]
    missing = [name for name in required if not (transfermarkt_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Transfermarkt directory is missing: {missing}")

    cache_dir = transfermarkt_dir / "_feature_cache"
    country_cache = cache_dir / "country_daily_player_form.csv"
    position_cache = cache_dir / "position_daily_player_form.csv"
    if country_cache.exists() and position_cache.exists():
        country_daily = pd.read_csv(country_cache, parse_dates=["date"])
        position_daily = pd.read_csv(position_cache, parse_dates=["date"])
        return country_daily, position_daily

    players = pd.read_csv(
        transfermarkt_dir / "players.csv",
        usecols=["player_id", "country_of_citizenship", "position"],
    )
    players["country_key"] = players["country_of_citizenship"].map(normalize_name)
    players["position_group"] = players["position"].map(position_group)
    players = players.dropna(subset=["player_id", "country_key"])
    players = players[players["country_key"] != ""]

    competitions = pd.read_csv(
        transfermarkt_dir / "competitions.csv",
        usecols=["competition_id", "sub_type", "type"],
    )
    competitions["competition_weight"] = competitions.apply(transfermarkt_competition_weight, axis=1)
    competitions = competitions[["competition_id", "competition_weight"]]

    lineups = pd.read_csv(
        transfermarkt_dir / "game_lineups.csv",
        usecols=["game_id", "player_id", "type"],
    )
    lineups["lineup_start"] = (lineups["type"] == "starting_lineup").astype(np.int8)
    lineups["lineup_named_sub"] = (lineups["type"] == "substitutes").astype(np.int8)
    lineups = (
        lineups.groupby(["game_id", "player_id"], as_index=False)[["lineup_start", "lineup_named_sub"]]
        .max()
        .reset_index(drop=True)
    )

    appearances = pd.read_csv(
        transfermarkt_dir / "appearances.csv",
        usecols=[
            "game_id",
            "player_id",
            "date",
            "competition_id",
            "yellow_cards",
            "red_cards",
            "goals",
            "assists",
            "minutes_played",
        ],
    )
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    appearances = appearances.dropna(subset=["date", "player_id"])
    appearances = appearances.merge(players[["player_id", "country_key", "position_group"]], on="player_id", how="inner")
    appearances = appearances.merge(competitions, on="competition_id", how="left")
    appearances["competition_weight"] = appearances["competition_weight"].fillna(0.75)
    appearances = appearances.merge(lineups, on=["game_id", "player_id"], how="left")
    appearances[["lineup_start", "lineup_named_sub"]] = appearances[["lineup_start", "lineup_named_sub"]].fillna(0.0)

    numeric_cols = ["minutes_played", "goals", "assists", "yellow_cards", "red_cards", "lineup_start", "lineup_named_sub"]
    for col in numeric_cols:
        appearances[col] = pd.to_numeric(appearances[col], errors="coerce").fillna(0.0)

    appearances["appearances"] = 1.0
    appearances["goal_contrib"] = appearances["goals"] + appearances["assists"]
    weighted_cols = [
        "minutes_played",
        "goals",
        "assists",
        "goal_contrib",
        "yellow_cards",
        "red_cards",
        "lineup_start",
        "lineup_named_sub",
        "appearances",
    ]
    for col in weighted_cols:
        appearances[f"quality_{col}"] = appearances[col] * appearances["competition_weight"]

    value_columns = [
        "minutes_played",
        "goals",
        "assists",
        "goal_contrib",
        "yellow_cards",
        "red_cards",
        "lineup_start",
        "lineup_named_sub",
        "appearances",
        "quality_minutes_played",
        "quality_goals",
        "quality_assists",
        "quality_goal_contrib",
        "quality_lineup_start",
        "quality_lineup_named_sub",
        "quality_appearances",
    ]

    country_daily = appearances.groupby(["country_key", "date"], as_index=False)[value_columns].sum()
    position_daily = appearances.groupby(["country_key", "position_group", "date"], as_index=False)[
        [
            "minutes_played",
            "goal_contrib",
            "quality_minutes_played",
            "quality_goal_contrib",
            "lineup_start",
            "quality_lineup_start",
        ]
    ].sum()
    cache_dir.mkdir(parents=True, exist_ok=True)
    country_daily.to_csv(country_cache, index=False)
    position_daily.to_csv(position_cache, index=False)
    return country_daily, position_daily


def add_player_side_features(
    output: dict[str, float],
    prefix: str,
    stats: dict[str, float],
    position_stats: dict[str, dict[str, float]],
) -> None:
    minutes = stats["recency_minutes_played"]
    quality_minutes = stats["recency_quality_minutes_played"]
    appearances = stats["recency_appearances"]
    output.update(
        {
            f"{prefix}_player_minutes_2y": stats["minutes_played"],
            f"{prefix}_player_recency_minutes_2y": minutes,
            f"{prefix}_player_quality_minutes_2y": stats["quality_minutes_played"],
            f"{prefix}_player_recency_quality_minutes_2y": quality_minutes,
            f"{prefix}_player_goals_2y": stats["goals"],
            f"{prefix}_player_assists_2y": stats["assists"],
            f"{prefix}_player_goal_contrib_2y": stats["goal_contrib"],
            f"{prefix}_player_recency_goal_contrib_2y": stats["recency_goal_contrib"],
            f"{prefix}_player_quality_goal_contrib_2y": stats["quality_goal_contrib"],
            f"{prefix}_player_recency_quality_goal_contrib_2y": stats["recency_quality_goal_contrib"],
            f"{prefix}_player_starts_2y": stats["lineup_start"],
            f"{prefix}_player_recency_starts_2y": stats["recency_lineup_start"],
            f"{prefix}_player_named_subs_2y": stats["lineup_named_sub"],
            f"{prefix}_player_recency_named_subs_2y": stats["recency_lineup_named_sub"],
            f"{prefix}_player_apps_2y": stats["appearances"],
            f"{prefix}_player_recency_apps_2y": appearances,
            f"{prefix}_player_yellows_2y": stats["yellow_cards"],
            f"{prefix}_player_reds_2y": stats["red_cards"],
            f"{prefix}_player_goal_contrib_per90_2y": safe_rate(stats["goal_contrib"] * 90.0, stats["minutes_played"]),
            f"{prefix}_player_recency_goal_contrib_per90_2y": safe_rate(
                stats["recency_goal_contrib"] * 90.0, minutes
            ),
            f"{prefix}_player_quality_goal_contrib_per90_2y": safe_rate(
                stats["quality_goal_contrib"] * 90.0, stats["quality_minutes_played"]
            ),
            f"{prefix}_player_recency_quality_goal_contrib_per90_2y": safe_rate(
                stats["recency_quality_goal_contrib"] * 90.0, quality_minutes
            ),
            f"{prefix}_player_start_rate_2y": safe_rate(stats["lineup_start"], stats["appearances"]),
            f"{prefix}_player_recency_start_rate_2y": safe_rate(stats["recency_lineup_start"], appearances),
            f"{prefix}_player_quality_start_rate_2y": safe_rate(
                stats["quality_lineup_start"], stats["quality_appearances"]
            ),
        }
    )

    for group in POSITION_GROUPS:
        group_stats = position_stats[group]
        group_minutes = group_stats["recency_minutes_played"]
        group_quality_minutes = group_stats["recency_quality_minutes_played"]
        output.update(
            {
                f"{prefix}_{group}_recency_minutes_2y": group_minutes,
                f"{prefix}_{group}_recency_quality_minutes_2y": group_quality_minutes,
                f"{prefix}_{group}_recency_goal_contrib_2y": group_stats["recency_goal_contrib"],
                f"{prefix}_{group}_recency_quality_goal_contrib_2y": group_stats[
                    "recency_quality_goal_contrib"
                ],
                f"{prefix}_{group}_recency_goal_contrib_per90_2y": safe_rate(
                    group_stats["recency_goal_contrib"] * 90.0, group_minutes
                ),
                f"{prefix}_{group}_recency_quality_goal_contrib_per90_2y": safe_rate(
                    group_stats["recency_quality_goal_contrib"] * 90.0, group_quality_minutes
                ),
                f"{prefix}_{group}_recency_starts_2y": group_stats["recency_lineup_start"],
                f"{prefix}_{group}_recency_quality_starts_2y": group_stats["recency_quality_lineup_start"],
            }
        )


def resolve_transfermarkt_dir(transfermarkt_dir: Path) -> Path:
    versions_dir = transfermarkt_dir.parent
    if not versions_dir.exists():
        return transfermarkt_dir
    candidates = [
        path
        for path in versions_dir.iterdir()
        if path.is_dir()
        and (path / "players.csv").exists()
        and (path / "appearances.csv").exists()
        and (path / "competitions.csv").exists()
    ]
    if not candidates:
        return transfermarkt_dir
    latest = max(
        candidates,
        key=lambda path: (int(path.name) if path.name.isdigit() else -1, path.stat().st_mtime),
    )
    if latest != transfermarkt_dir:
        print(f"Using latest Transfermarkt player-scores version: {latest}")
    return latest


def attach_transfermarkt_player_features(
    matches: pd.DataFrame,
    transfermarkt_dir: Path | None,
    window_days: int,
    half_life_days: int,
) -> tuple[pd.DataFrame, bool]:
    if transfermarkt_dir is None:
        return matches, False
    transfermarkt_dir = resolve_transfermarkt_dir(Path(transfermarkt_dir))
    if not transfermarkt_dir.exists():
        print(f"Skipping player features: {transfermarkt_dir} does not exist.", file=sys.stderr)
        return matches, False

    print(f"Building Transfermarkt player-form features from {transfermarkt_dir} ...")
    country_daily, position_daily = build_transfermarkt_daily_stats(transfermarkt_dir)
    country_values = [
        "minutes_played",
        "goals",
        "assists",
        "goal_contrib",
        "yellow_cards",
        "red_cards",
        "lineup_start",
        "lineup_named_sub",
        "appearances",
        "quality_minutes_played",
        "quality_goals",
        "quality_assists",
        "quality_goal_contrib",
        "quality_lineup_start",
        "quality_lineup_named_sub",
        "quality_appearances",
    ]
    position_values = [
        "minutes_played",
        "goal_contrib",
        "quality_minutes_played",
        "quality_goal_contrib",
        "lineup_start",
        "quality_lineup_start",
    ]
    country_lookup = RollingStatsLookup(country_daily, ["country_key"], country_values, half_life_days)
    position_lookup = RollingStatsLookup(position_daily, ["country_key", "position_group"], position_values, half_life_days)

    feature_rows: list[dict[str, float]] = []
    for row in matches[["date", "home_team", "away_team"]].itertuples(index=False):
        match_date = pd.Timestamp(row.date)
        home_key = normalize_name(row.home_team)
        away_key = normalize_name(row.away_team)
        home_stats = country_lookup.query(home_key, match_date, window_days)
        away_stats = country_lookup.query(away_key, match_date, window_days)
        home_position_stats = {
            group: position_lookup.query((home_key, group), match_date, window_days) for group in POSITION_GROUPS
        }
        away_position_stats = {
            group: position_lookup.query((away_key, group), match_date, window_days) for group in POSITION_GROUPS
        }

        features: dict[str, float] = {}
        add_player_side_features(features, "home", home_stats, home_position_stats)
        add_player_side_features(features, "away", away_stats, away_position_stats)

        for suffix in [
            "player_recency_minutes_2y",
            "player_recency_quality_minutes_2y",
            "player_recency_goal_contrib_2y",
            "player_recency_quality_goal_contrib_2y",
            "player_recency_goal_contrib_per90_2y",
            "player_recency_quality_goal_contrib_per90_2y",
            "player_recency_starts_2y",
            "player_recency_start_rate_2y",
            "player_quality_start_rate_2y",
            "attack_recency_quality_goal_contrib_2y",
            "midfield_recency_quality_goal_contrib_2y",
            "defender_recency_quality_minutes_2y",
            "goalkeeper_recency_quality_minutes_2y",
        ]:
            features[f"{suffix}_diff"] = features[f"home_{suffix}"] - features[f"away_{suffix}"]
        feature_rows.append(features)

    player_features = pd.DataFrame(feature_rows, index=matches.index)
    matched_rows = int(
        (
            player_features["home_player_recency_apps_2y"].gt(0)
            | player_features["away_player_recency_apps_2y"].gt(0)
        ).sum()
    )
    print(f"Player-form features available for {matched_rows:,} of {len(matches):,} rows.")
    return pd.concat([matches.reset_index(drop=True), player_features.reset_index(drop=True)], axis=1), True


def ensure_data(data_dir: Path, skip_download: bool) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for file_name in DATA_FILES:
        path = data_dir / file_name
        if path.exists() or skip_download:
            continue
        url = f"{GITHUB_RAW_BASE}/{file_name}"
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
) -> Pipeline:
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=10)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=True)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            ("categorical", one_hot, categorical_features),
        ],
        remainder="drop",
    )

    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=3,
        n_estimators=450,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_lambda=2.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def build_score_pipeline(numeric_features: list[str], categorical_features: list[str], random_state: int) -> Pipeline:
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=10)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=True)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            ("categorical", one_hot, categorical_features),
        ],
        remainder="drop",
    )

    model = XGBRegressor(
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        n_estimators=360,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_lambda=2.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def align_feature_columns(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    output = frame.copy()
    for col in numeric_features:
        if col not in output.columns:
            output[col] = np.nan
    for col in categorical_features:
        if col not in output.columns:
            output[col] = "Unknown"
    return output[numeric_features + categorical_features]


def compute_recency_sample_weight(
    dates: pd.Series,
    half_life_days: int,
    min_weight: float,
) -> np.ndarray | None:
    if half_life_days <= 0:
        return None
    parsed = pd.to_datetime(dates, errors="coerce")
    max_date = parsed.max()
    if pd.isna(max_date):
        return None
    age_days = (max_date - parsed).dt.days.fillna(half_life_days).clip(lower=0).to_numpy(dtype=float)
    weights = np.exp(-math.log(2.0) * age_days / float(half_life_days))
    weights = np.maximum(weights, float(min_weight))
    return weights / np.mean(weights)


def prediction_frame(meta: pd.DataFrame, probabilities: np.ndarray, predicted_ids: np.ndarray) -> pd.DataFrame:
    output = meta.copy()
    output["predicted_outcome"] = [ID_TO_OUTCOME[int(value)] for value in predicted_ids]
    output["prob_away_win"] = probabilities[:, OUTCOME_TO_ID["away_win"]]
    output["prob_draw"] = probabilities[:, OUTCOME_TO_ID["draw"]]
    output["prob_home_win"] = probabilities[:, OUTCOME_TO_ID["home_win"]]
    return output


def poisson_pmf(lam: float, max_goals: int) -> np.ndarray:
    lam = float(np.clip(lam, 0.03, 12.0))
    probs = np.zeros(max_goals + 1, dtype=np.float64)
    probs[0] = math.exp(-lam)
    for goals in range(1, max_goals + 1):
        probs[goals] = probs[goals - 1] * lam / goals
    probs /= probs.sum()
    return probs


def most_likely_scoreline(home_expected: float, away_expected: float, max_goals: int) -> tuple[int, int, float]:
    home_probs = poisson_pmf(home_expected, max_goals)
    away_probs = poisson_pmf(away_expected, max_goals)
    matrix = np.outer(home_probs, away_probs)
    flat_index = int(matrix.argmax())
    home_goals, away_goals = np.unravel_index(flat_index, matrix.shape)
    return int(home_goals), int(away_goals), float(matrix[home_goals, away_goals])


def most_likely_scoreline_for_outcome(
    home_expected: float,
    away_expected: float,
    max_goals: int,
    required_outcome: str,
) -> tuple[int, int, float]:
    home_probs = poisson_pmf(home_expected, max_goals)
    away_probs = poisson_pmf(away_expected, max_goals)
    matrix = np.outer(home_probs, away_probs)
    home_grid, away_grid = np.indices(matrix.shape)
    if required_outcome == "home_win":
        mask = home_grid > away_grid
    elif required_outcome == "away_win":
        mask = home_grid < away_grid
    else:
        mask = home_grid == away_grid

    constrained = np.where(mask, matrix, -1.0)
    if constrained.max() < 0:
        return most_likely_scoreline(home_expected, away_expected, max_goals)
    flat_index = int(constrained.argmax())
    home_goals, away_goals = np.unravel_index(flat_index, constrained.shape)
    return int(home_goals), int(away_goals), float(matrix[home_goals, away_goals])


def score_prediction_frame(
    meta: pd.DataFrame,
    home_expected: np.ndarray,
    away_expected: np.ndarray,
    max_goals: int,
    forced_outcomes: list[str] | None = None,
) -> pd.DataFrame:
    output = meta.copy()
    output["expected_home_goals"] = np.clip(home_expected, 0.03, 12.0)
    output["expected_away_goals"] = np.clip(away_expected, 0.03, 12.0)
    scorelines = [
        most_likely_scoreline(home_xg, away_xg, max_goals)
        for home_xg, away_xg in zip(output["expected_home_goals"], output["expected_away_goals"])
    ]
    output["predicted_home_score"] = [home for home, _, _ in scorelines]
    output["predicted_away_score"] = [away for _, away, _ in scorelines]
    output["predicted_score"] = output["predicted_home_score"].astype(str) + "-" + output["predicted_away_score"].astype(str)
    output["predicted_score_probability"] = [probability for _, _, probability in scorelines]
    output["predicted_score_outcome"] = [
        outcome_label(home, away)
        for home, away in zip(output["predicted_home_score"], output["predicted_away_score"])
    ]
    if forced_outcomes is not None:
        pool_scorelines = [
            most_likely_scoreline_for_outcome(home_xg, away_xg, max_goals, required_outcome)
            for home_xg, away_xg, required_outcome in zip(
                output["expected_home_goals"],
                output["expected_away_goals"],
                forced_outcomes,
            )
        ]
        output["pool_predicted_home_score"] = [home for home, _, _ in pool_scorelines]
        output["pool_predicted_away_score"] = [away for _, away, _ in pool_scorelines]
        output["pool_predicted_score"] = (
            output["pool_predicted_home_score"].astype(str) + "-" + output["pool_predicted_away_score"].astype(str)
        )
        output["pool_predicted_score_probability"] = [probability for _, _, probability in pool_scorelines]
        output["pool_predicted_score_outcome"] = [
            outcome_label(home, away)
            for home, away in zip(output["pool_predicted_home_score"], output["pool_predicted_away_score"])
        ]
    if {"home_score", "away_score"}.issubset(output.columns):
        output["exact_score_correct"] = (
            output["predicted_home_score"].astype(int).eq(output["home_score"].astype(int))
            & output["predicted_away_score"].astype(int).eq(output["away_score"].astype(int))
        )
        output["score_outcome_correct"] = output["predicted_score_outcome"].eq(output["actual_outcome"])
        if forced_outcomes is not None:
            output["pool_exact_score_correct"] = (
                output["pool_predicted_home_score"].astype(int).eq(output["home_score"].astype(int))
                & output["pool_predicted_away_score"].astype(int).eq(output["away_score"].astype(int))
            )
            output["pool_score_outcome_correct"] = output["pool_predicted_score_outcome"].eq(output["actual_outcome"])
    return output


def score_metrics(
    y_home: np.ndarray,
    y_away: np.ndarray,
    home_expected: np.ndarray,
    away_expected: np.ndarray,
    score_predictions: pd.DataFrame,
) -> dict[str, float]:
    exact = score_predictions["exact_score_correct"].to_numpy()
    outcome = score_predictions["score_outcome_correct"].to_numpy()
    total_goals_true = y_home + y_away
    total_goals_expected = home_expected + away_expected
    home_rmse = float(np.sqrt(np.mean((home_expected - y_home) ** 2)))
    away_rmse = float(np.sqrt(np.mean((away_expected - y_away) ** 2)))
    total_rmse = float(np.sqrt(np.mean((total_goals_expected - total_goals_true) ** 2)))
    metrics = {
        "score_exact_accuracy": float(np.mean(exact)),
        "score_outcome_accuracy": float(np.mean(outcome)),
        "home_goals_mae": float(mean_absolute_error(y_home, home_expected)),
        "away_goals_mae": float(mean_absolute_error(y_away, away_expected)),
        "total_goals_mae": float(mean_absolute_error(total_goals_true, total_goals_expected)),
        "home_goals_rmse": home_rmse,
        "away_goals_rmse": away_rmse,
        "total_goals_rmse": total_rmse,
    }
    if {"pool_exact_score_correct", "pool_score_outcome_correct"}.issubset(score_predictions.columns):
        metrics["pool_score_exact_accuracy"] = float(np.mean(score_predictions["pool_exact_score_correct"].to_numpy()))
        metrics["pool_score_outcome_accuracy"] = float(
            np.mean(score_predictions["pool_score_outcome_correct"].to_numpy())
        )
    return metrics


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def save_feature_importance(pipeline: Pipeline, numeric: list[str], categorical: list[str], output_path: Path) -> None:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocess"]
    model: XGBClassifier = pipeline.named_steps["model"]
    feature_names = list(numeric)
    if categorical:
        encoder = preprocessor.named_transformers_["categorical"]
        try:
            cat_names = encoder.get_feature_names_out(categorical).tolist()
        except AttributeError:
            cat_names = encoder.get_feature_names(categorical).tolist()
        feature_names.extend(cat_names)

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(output_path, index=False)


def standardize_future_fixtures(path: Path, default_date: pd.Timestamp) -> pd.DataFrame:
    fixtures = pd.read_csv(path)
    columns = list(fixtures.columns)
    home_col = find_column(columns, ("home_team", "home team", "home"))
    away_col = find_column(columns, ("away_team", "away team", "away"))
    date_col = find_column(columns, ("date", "match_date", "fixture_date"))
    tournament_col = find_column(columns, ("tournament", "competition"))
    country_col = find_column(columns, ("country", "host_country"))
    city_col = find_column(columns, ("city", "venue_city"))
    neutral_col = find_column(columns, ("neutral", "is_neutral"))

    if home_col is None or away_col is None:
        raise ValueError("Future fixtures CSV needs home_team and away_team columns.")

    output = pd.DataFrame(
        {
            "date": pd.to_datetime(fixtures[date_col], errors="coerce") if date_col else default_date,
            "home_team": fixtures[home_col],
            "away_team": fixtures[away_col],
            "tournament": fixtures[tournament_col] if tournament_col else "FIFA World Cup",
            "city": fixtures[city_col] if city_col else "Unknown",
            "country": fixtures[country_col] if country_col else "Unknown",
            "neutral": fixtures[neutral_col] if neutral_col else True,
        }
    )
    output["date"] = output["date"].fillna(default_date)
    return output


def build_future_features(
    fixtures: pd.DataFrame,
    states: dict[str, TeamState],
    h2h: dict[tuple[str, str], dict[str, float]],
) -> pd.DataFrame:
    rows = [match_feature_row(row, states, h2h) for _, row in fixtures.iterrows()]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost model for international football 1X2 outcomes.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--rankings", type=Path, default=Path("fifa_ranking-2026-04-01.csv"))
    parser.add_argument("--external-elo", type=Path, default=DEFAULT_EXTERNAL_ELO)
    parser.add_argument("--odds-csv", type=Path, default=None)
    parser.add_argument("--future-fixtures", type=Path, default=None)
    parser.add_argument("--transfermarkt-dir", type=Path, default=DEFAULT_TRANSFERMARKT_DIR)
    parser.add_argument("--fjelstul-dir", type=Path, default=DEFAULT_FJELSTUL_DIR)
    parser.add_argument("--xfkz-snapshots", type=Path, default=DEFAULT_XFKZ_SNAPSHOTS)
    parser.add_argument("--geo-cities", type=Path, default=DEFAULT_GEO_CITY_LOCATIONS)
    parser.add_argument("--geo-country-reps", type=Path, default=DEFAULT_GEO_COUNTRY_REPS)
    parser.add_argument("--goalscorers", type=Path, default=DEFAULT_GOALSCORERS)
    parser.add_argument("--squads-csv", type=Path, default=DEFAULT_TOURNAMENT_SQUADS)
    parser.add_argument("--soccerbase-lineups", type=Path, default=DEFAULT_SOCCERBASE_LINEUPS)
    parser.add_argument("--soccerbase-match-stats", type=Path, default=DEFAULT_SOCCERBASE_MATCH_STATS)
    parser.add_argument("--soccerbase-cards", type=Path, default=DEFAULT_SOCCERBASE_CARDS)
    parser.add_argument("--sofifa-yearly-ratings", type=Path, default=DEFAULT_SOFIFA_YEARLY_RATINGS)
    parser.add_argument(
        "--use-geo-features",
        action="store_true",
        help="Use offline city/country geo features. Off by default because the first holdout test lowered 1X2 accuracy.",
    )
    parser.add_argument("--no-geo-features", action="store_true")
    parser.add_argument("--use-external-elo-features", action="store_true")
    parser.add_argument(
        "--use-xfkz-features",
        action="store_true",
        help="Use extracted xfkz country market/injury snapshots. Off by default because it improves calibration slightly but not holdout accuracy.",
    )
    parser.add_argument("--no-player-features", action="store_true")
    parser.add_argument("--player-window-days", type=int, default=730)
    parser.add_argument("--player-half-life-days", type=int, default=365)
    parser.add_argument("--no-goalscorer-features", action="store_true")
    parser.add_argument("--goalscorer-window-days", type=int, default=730)
    parser.add_argument("--goalscorer-half-life-days", type=int, default=365)
    parser.add_argument("--no-squad-features", action="store_true")
    parser.add_argument("--use-squad-market-values", action="store_true")
    parser.add_argument(
        "--use-squad-sofifa-ratings",
        action="store_true",
        help="Match tournament squads to historical SoFIFA rows and add squad quality aggregates.",
    )
    parser.add_argument("--use-soccerbase-lineup-features", action="store_true")
    parser.add_argument("--use-soccerbase-ratings", action="store_true")
    parser.add_argument("--use-sofifa-yearly-ratings", action="store_true")
    parser.add_argument("--use-soccerbase-stat-features", action="store_true")
    parser.add_argument("--use-soccerbase-stat-recency-features", action="store_true")
    parser.add_argument("--soccerbase-stat-window-days", type=int, default=730)
    parser.add_argument("--use-soccerbase-card-features", action="store_true")
    parser.add_argument("--soccerbase-card-window-days", type=int, default=730)
    parser.add_argument("--soccerbase-card-last-n", type=int, default=5)
    parser.add_argument("--train-from", default="1993-01-01")
    parser.add_argument("--test-from", default="2023-01-01")
    parser.add_argument("--recent-window", type=int, default=10)
    parser.add_argument("--score-max-goals", type=int, default=8)
    parser.add_argument("--sample-half-life-days", type=int, default=0)
    parser.add_argument("--sample-min-weight", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    ensure_data(args.data_dir, args.skip_download)
    results_path = args.results if args.results is not None else args.data_dir / "results.csv"
    results = load_results(results_path)
    print(
        f"Loaded {len(results):,} matches from {results['date'].min().date()} "
        f"to {results['date'].max().date()}."
    )

    features, states, h2h = build_historical_features(results, args.recent_window)
    features = attach_rankings(features, args.rankings)
    external_elo_rows_matched = 0
    if args.use_external_elo_features:
        features, external_elo_rows_matched = attach_external_elo_features(features, args.external_elo)
        print(f"Matched external Elo features for {external_elo_rows_matched:,} historical rows.")
    features = attach_context_features(features, args.rankings)
    geo_rows_matched = 0
    if args.use_geo_features and not args.no_geo_features:
        features, geo_rows_matched = attach_geo_features(features, args.geo_cities, args.geo_country_reps)
    goalscorer_rows_matched = 0
    if not args.no_goalscorer_features:
        features, goalscorer_rows_matched = attach_goalscorer_form_features(
            features,
            args.goalscorers,
            args.goalscorer_window_days,
            args.goalscorer_half_life_days,
        )
        print(f"Matched goalscorer form for {goalscorer_rows_matched:,} historical rows.")
    squad_rows_matched = 0
    if not args.no_squad_features:
        features, squad_rows_matched = attach_tournament_squad_features(
            features,
            args.squads_csv,
            args.transfermarkt_dir,
            args.use_squad_market_values,
            args.sofifa_yearly_ratings,
            args.use_squad_sofifa_ratings,
        )
        print(f"Matched tournament squad features for {squad_rows_matched:,} historical rows.")
    soccerbase_lineup_rows_matched = 0
    if args.use_soccerbase_lineup_features:
        features, soccerbase_lineup_rows_matched = attach_soccerbase_lineup_features(
            features,
            args.soccerbase_lineups,
            args.use_soccerbase_ratings or args.use_sofifa_yearly_ratings,
            args.sofifa_yearly_ratings if args.use_sofifa_yearly_ratings else None,
        )
        print(f"Matched Soccerbase lineup features for {soccerbase_lineup_rows_matched:,} historical rows.")
    soccerbase_stat_rows_matched = 0
    if args.use_soccerbase_stat_features:
        features, soccerbase_stat_rows_matched = attach_soccerbase_rolling_stat_features(
            features,
            args.soccerbase_match_stats,
            args.soccerbase_stat_window_days,
            args.use_soccerbase_stat_recency_features,
        )
        print(f"Matched Soccerbase rolling stat features for {soccerbase_stat_rows_matched:,} historical rows.")
    soccerbase_card_rows_matched = 0
    if args.use_soccerbase_card_features:
        features, soccerbase_card_rows_matched = attach_soccerbase_card_features(
            features,
            args.soccerbase_cards,
            args.soccerbase_card_window_days,
            args.soccerbase_card_last_n,
        )
        print(f"Matched Soccerbase rolling card features for {soccerbase_card_rows_matched:,} historical rows.")
    features, manager_feature_matches = attach_fjelstul_manager_features(features, args.fjelstul_dir)
    xfkz_snapshot_matches = 0
    if args.use_xfkz_features:
        features, xfkz_snapshot_matches = attach_xfkz_snapshot_features(features, args.xfkz_snapshots)
    player_features_enabled = False
    if not args.no_player_features:
        features, player_features_enabled = attach_transfermarkt_player_features(
            features,
            args.transfermarkt_dir,
            args.player_window_days,
            args.player_half_life_days,
        )
    features, odds_matches = attach_odds(features, args.odds_csv)
    if args.odds_csv is not None:
        print(f"Matched odds for {odds_matches:,} historical rows.")

    train_from = pd.Timestamp(args.train_from)
    test_from = pd.Timestamp(args.test_from)
    model_data = features[features["date"] >= train_from].copy()
    train = model_data[model_data["date"] < test_from].copy()
    test = model_data[model_data["date"] >= test_from].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"Train/test split produced empty data. train={len(train):,}, test={len(test):,}. "
            "Adjust --train-from or --test-from."
        )

    leakage_or_meta = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "actual_outcome",
        "target",
    }
    candidate_features = [col for col in model_data.columns if col not in leakage_or_meta]
    categorical_features = [
        col
        for col in candidate_features
        if col in {"tournament", "city", "country"}
        or not pd.api.types.is_numeric_dtype(model_data[col])
    ]
    numeric_features = [col for col in candidate_features if col not in categorical_features]

    pipeline = build_pipeline(numeric_features, categorical_features, args.random_state)
    x_train = align_feature_columns(train, numeric_features, categorical_features)
    y_train = train["target"].to_numpy()
    x_test = align_feature_columns(test, numeric_features, categorical_features)
    y_test = test["target"].to_numpy()
    train_sample_weight = compute_recency_sample_weight(
        train["date"],
        args.sample_half_life_days,
        args.sample_min_weight,
    )

    print(f"Training rows: {len(train):,}; test rows: {len(test):,}; features: {len(candidate_features):,}.")
    fit_kwargs = {"model__sample_weight": train_sample_weight} if train_sample_weight is not None else {}
    pipeline.fit(x_train, y_train, **fit_kwargs)
    probabilities = pipeline.predict_proba(x_test)
    predictions = probabilities.argmax(axis=1)

    home_score_pipeline = build_score_pipeline(numeric_features, categorical_features, args.random_state)
    away_score_pipeline = build_score_pipeline(numeric_features, categorical_features, args.random_state + 1)
    y_home_train = train["home_score"].to_numpy(dtype=float)
    y_away_train = train["away_score"].to_numpy(dtype=float)
    y_home_test = test["home_score"].to_numpy(dtype=float)
    y_away_test = test["away_score"].to_numpy(dtype=float)
    home_score_pipeline.fit(x_train, y_home_train, **fit_kwargs)
    away_score_pipeline.fit(x_train, y_away_train, **fit_kwargs)
    home_expected = np.clip(home_score_pipeline.predict(x_test), 0.03, 12.0)
    away_expected = np.clip(away_score_pipeline.predict(x_test), 0.03, 12.0)

    meta_columns = ["date", "home_team", "away_team", "tournament", "home_score", "away_score", "actual_outcome"]
    score_predictions = score_prediction_frame(
        test[meta_columns].reset_index(drop=True),
        home_expected,
        away_expected,
        args.score_max_goals,
        [ID_TO_OUTCOME[int(value)] for value in predictions],
    )
    score_metric_values = score_metrics(
        y_home_test,
        y_away_test,
        home_expected,
        away_expected,
        score_predictions,
    )

    metrics = {
        "train_from": args.train_from,
        "test_from": args.test_from,
        "rows_total": int(len(model_data)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "features": int(len(candidate_features)),
        "odds_rows_matched": int(odds_matches),
        "external_elo_rows_matched": int(external_elo_rows_matched),
        "external_elo_features_enabled": bool(args.use_external_elo_features),
        "geo_rows_matched": int(geo_rows_matched),
        "goalscorer_feature_rows_matched": int(goalscorer_rows_matched),
        "goalscorer_window_days": None if args.no_goalscorer_features else int(args.goalscorer_window_days),
        "goalscorer_half_life_days": None if args.no_goalscorer_features else int(args.goalscorer_half_life_days),
        "squad_feature_rows_matched": int(squad_rows_matched),
        "squad_market_values_enabled": bool(not args.no_squad_features and args.use_squad_market_values),
        "squad_sofifa_ratings_enabled": bool(not args.no_squad_features and args.use_squad_sofifa_ratings),
        "soccerbase_lineup_rows_matched": int(soccerbase_lineup_rows_matched),
        "soccerbase_lineup_features_enabled": bool(args.use_soccerbase_lineup_features),
        "soccerbase_ratings_enabled": bool(args.use_soccerbase_ratings),
        "sofifa_yearly_ratings_enabled": bool(args.use_sofifa_yearly_ratings),
        "soccerbase_stat_rows_matched": int(soccerbase_stat_rows_matched),
        "soccerbase_stat_features_enabled": bool(args.use_soccerbase_stat_features),
        "soccerbase_stat_recency_features_enabled": bool(args.use_soccerbase_stat_recency_features),
        "soccerbase_stat_window_days": int(args.soccerbase_stat_window_days)
        if args.use_soccerbase_stat_features
        else None,
        "soccerbase_card_rows_matched": int(soccerbase_card_rows_matched),
        "soccerbase_card_features_enabled": bool(args.use_soccerbase_card_features),
        "soccerbase_card_window_days": int(args.soccerbase_card_window_days)
        if args.use_soccerbase_card_features
        else None,
        "soccerbase_card_last_n": int(args.soccerbase_card_last_n)
        if args.use_soccerbase_card_features
        else None,
        "manager_feature_side_matches": int(manager_feature_matches),
        "xfkz_snapshot_rows_matched": int(xfkz_snapshot_matches),
        "player_features_enabled": bool(player_features_enabled),
        "player_window_days": int(args.player_window_days) if player_features_enabled else None,
        "player_half_life_days": int(args.player_half_life_days) if player_features_enabled else None,
        "sample_half_life_days": int(args.sample_half_life_days),
        "sample_min_weight": float(args.sample_min_weight),
        "score_max_goals": int(args.score_max_goals),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1, 2])),
        "brier_score": multiclass_brier(y_test, probabilities),
        **score_metric_values,
        "labels": ID_TO_OUTCOME,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=[ID_TO_OUTCOME[i] for i in [0, 1, 2]],
            output_dict=True,
            zero_division=0,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    test_predictions = prediction_frame(test[meta_columns].reset_index(drop=True), probabilities, predictions)
    test_predictions = test_predictions.merge(
        score_predictions[
            [
                "date",
                "home_team",
                "away_team",
                "expected_home_goals",
                "expected_away_goals",
                "predicted_home_score",
                "predicted_away_score",
                "predicted_score",
                "predicted_score_probability",
                "predicted_score_outcome",
                "exact_score_correct",
                "score_outcome_correct",
                "pool_predicted_home_score",
                "pool_predicted_away_score",
                "pool_predicted_score",
                "pool_predicted_score_probability",
                "pool_predicted_score_outcome",
                "pool_exact_score_correct",
                "pool_score_outcome_correct",
            ]
        ],
        on=["date", "home_team", "away_team"],
        how="left",
    )
    test_predictions.to_csv(args.output_dir / "test_predictions_xgboost.csv", index=False)
    score_predictions.to_csv(args.output_dir / "test_score_predictions_xgboost.csv", index=False)
    save_feature_importance(
        pipeline,
        numeric_features,
        categorical_features,
        args.output_dir / "feature_importance_xgboost.csv",
    )

    final_pipeline = build_pipeline(numeric_features, categorical_features, args.random_state)
    final_home_score_pipeline = build_score_pipeline(numeric_features, categorical_features, args.random_state)
    final_away_score_pipeline = build_score_pipeline(numeric_features, categorical_features, args.random_state + 1)
    final_x = align_feature_columns(model_data, numeric_features, categorical_features)
    final_sample_weight = compute_recency_sample_weight(
        model_data["date"],
        args.sample_half_life_days,
        args.sample_min_weight,
    )
    final_fit_kwargs = {"model__sample_weight": final_sample_weight} if final_sample_weight is not None else {}
    final_pipeline.fit(final_x, model_data["target"].to_numpy(), **final_fit_kwargs)
    final_home_score_pipeline.fit(final_x, model_data["home_score"].to_numpy(dtype=float), **final_fit_kwargs)
    final_away_score_pipeline.fit(final_x, model_data["away_score"].to_numpy(dtype=float), **final_fit_kwargs)
    model_payload = {
        "pipeline": final_pipeline,
        "home_score_pipeline": final_home_score_pipeline,
        "away_score_pipeline": final_away_score_pipeline,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "train_from": args.train_from,
        "results_max_date": str(results["date"].max().date()),
        "labels": ID_TO_OUTCOME,
        "score_max_goals": int(args.score_max_goals),
        "sample_half_life_days": int(args.sample_half_life_days),
        "sample_min_weight": float(args.sample_min_weight),
        "external_elo_features_enabled": bool(args.use_external_elo_features),
        "goalscorer_features_enabled": not bool(args.no_goalscorer_features),
        "goalscorer_window_days": None if args.no_goalscorer_features else int(args.goalscorer_window_days),
        "goalscorer_half_life_days": None if args.no_goalscorer_features else int(args.goalscorer_half_life_days),
        "squad_features_enabled": not bool(args.no_squad_features),
        "squad_market_values_enabled": bool(not args.no_squad_features and args.use_squad_market_values),
        "soccerbase_lineup_features_enabled": bool(args.use_soccerbase_lineup_features),
        "soccerbase_ratings_enabled": bool(args.use_soccerbase_ratings),
        "sofifa_yearly_ratings_enabled": bool(args.use_sofifa_yearly_ratings),
        "soccerbase_stat_features_enabled": bool(args.use_soccerbase_stat_features),
        "soccerbase_stat_recency_features_enabled": bool(args.use_soccerbase_stat_recency_features),
        "soccerbase_stat_window_days": int(args.soccerbase_stat_window_days)
        if args.use_soccerbase_stat_features
        else None,
        "soccerbase_card_features_enabled": bool(args.use_soccerbase_card_features),
        "soccerbase_card_window_days": int(args.soccerbase_card_window_days)
        if args.use_soccerbase_card_features
        else None,
        "soccerbase_card_last_n": int(args.soccerbase_card_last_n)
        if args.use_soccerbase_card_features
        else None,
    }
    joblib.dump(model_payload, args.model_dir / "worldcup_xgboost_model.joblib")

    if args.future_fixtures is not None:
        future_default_date = results["date"].max() + pd.Timedelta(days=1)
        fixtures = standardize_future_fixtures(args.future_fixtures, future_default_date)
        future_features = build_future_features(fixtures, states, h2h)
        future_features = attach_rankings(future_features, args.rankings)
        if args.use_external_elo_features:
            future_features, _ = attach_external_elo_features(future_features, args.external_elo)
        future_features = attach_context_features(future_features, args.rankings)
        if args.use_geo_features and not args.no_geo_features:
            future_features, _ = attach_geo_features(future_features, args.geo_cities, args.geo_country_reps)
        if not args.no_goalscorer_features:
            future_features, _ = attach_goalscorer_form_features(
                future_features,
                args.goalscorers,
                args.goalscorer_window_days,
                args.goalscorer_half_life_days,
            )
        if not args.no_squad_features:
            future_features, _ = attach_tournament_squad_features(
                future_features,
                args.squads_csv,
                args.transfermarkt_dir,
                args.use_squad_market_values,
                args.sofifa_yearly_ratings,
                args.use_squad_sofifa_ratings,
            )
        if args.use_soccerbase_lineup_features:
            future_features, _ = attach_soccerbase_lineup_features(
                future_features,
                args.soccerbase_lineups,
                args.use_soccerbase_ratings or args.use_sofifa_yearly_ratings,
                args.sofifa_yearly_ratings if args.use_sofifa_yearly_ratings else None,
            )
        if args.use_soccerbase_stat_features:
            future_features, _ = attach_soccerbase_rolling_stat_features(
                future_features,
                args.soccerbase_match_stats,
                args.soccerbase_stat_window_days,
                args.use_soccerbase_stat_recency_features,
            )
        if args.use_soccerbase_card_features:
            future_features, _ = attach_soccerbase_card_features(
                future_features,
                args.soccerbase_cards,
                args.soccerbase_card_window_days,
                args.soccerbase_card_last_n,
            )
        if args.use_xfkz_features:
            future_features, _ = attach_xfkz_snapshot_features(future_features, args.xfkz_snapshots)
        future_features, _ = attach_odds(future_features, args.odds_csv)
        future_probabilities = final_pipeline.predict_proba(
            align_feature_columns(future_features, numeric_features, categorical_features)
        )
        future_x = align_feature_columns(future_features, numeric_features, categorical_features)
        future_home_expected = np.clip(final_home_score_pipeline.predict(future_x), 0.03, 12.0)
        future_away_expected = np.clip(final_away_score_pipeline.predict(future_x), 0.03, 12.0)
        future_score_predictions = score_prediction_frame(
            future_features[["date", "home_team", "away_team", "tournament"]].reset_index(drop=True),
            future_home_expected,
            future_away_expected,
            args.score_max_goals,
            [ID_TO_OUTCOME[int(value)] for value in future_probabilities.argmax(axis=1)],
        )
        future_predictions = prediction_frame(
            future_features[["date", "home_team", "away_team", "tournament"]].reset_index(drop=True),
            future_probabilities,
            future_probabilities.argmax(axis=1),
        )
        future_predictions = future_predictions.merge(
            future_score_predictions[
                [
                    "date",
                    "home_team",
                    "away_team",
                    "expected_home_goals",
                    "expected_away_goals",
                    "predicted_home_score",
                    "predicted_away_score",
                    "predicted_score",
                    "predicted_score_probability",
                    "predicted_score_outcome",
                    "pool_predicted_home_score",
                    "pool_predicted_away_score",
                    "pool_predicted_score",
                    "pool_predicted_score_probability",
                    "pool_predicted_score_outcome",
                ]
            ],
            on=["date", "home_team", "away_team"],
            how="left",
        )
        future_predictions.to_csv(args.output_dir / "future_predictions_xgboost.csv", index=False)
        print(f"Saved future predictions for {len(future_predictions):,} fixtures.")

    print("\nScores on time-based test set:")
    print(f"  accuracy:          {metrics['accuracy']:.4f}")
    print(f"  balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  log_loss:          {metrics['log_loss']:.4f}")
    print(f"  brier_score:       {metrics['brier_score']:.4f}")
    print(f"  exact_score:       {metrics['score_exact_accuracy']:.4f}")
    print(f"  score_outcome:     {metrics['score_outcome_accuracy']:.4f}")
    print(f"  pool_exact_score:  {metrics['pool_score_exact_accuracy']:.4f}")
    print(f"  pool_score_outcome:{metrics['pool_score_outcome_accuracy']:.4f}")
    print(f"  total_goals_mae:   {metrics['total_goals_mae']:.4f}")
    print(f"\nSaved metrics to {args.output_dir / 'model_metrics.json'}")
    print(f"Saved test predictions to {args.output_dir / 'test_predictions_xgboost.csv'}")
    print(f"Saved test score predictions to {args.output_dir / 'test_score_predictions_xgboost.csv'}")
    print(f"Saved model to {args.model_dir / 'worldcup_xgboost_model.joblib'}")


if __name__ == "__main__":
    main()
