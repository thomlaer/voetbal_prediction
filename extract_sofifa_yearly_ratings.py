#!/usr/bin/env python
"""Normalize local SoFIFA/FIFA player-rating datasets into one time-aware CSV."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_ROOT = Path("data/kagglehub/datasets")
OUTPUT_PATH = Path("data/extracted/sofifa_yearly_player_ratings.csv")


OUTPUT_COLUMNS = [
    "fifa_year",
    "available_from",
    "source_file",
    "sofifa_id",
    "short_name",
    "long_name",
    "dob",
    "age",
    "nationality_name",
    "club_name",
    "league_name",
    "player_positions",
    "overall",
    "potential",
    "value_eur",
    "wage_eur",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
    "attacking_crossing",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_short_passing",
    "skill_dribbling",
    "skill_long_passing",
    "movement_acceleration",
    "movement_sprint_speed",
    "movement_agility",
    "movement_reactions",
    "power_stamina",
    "power_strength",
    "mentality_aggression",
    "mentality_interceptions",
    "mentality_positioning",
    "mentality_vision",
    "mentality_composure",
    "defending_marking_awareness",
    "defending_standing_tackle",
    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes",
]


ALIASES = {
    "sofifa_id": ("sofifa_id", "player_id", "Player ID", "ID"),
    "short_name": ("short_name", "name", "Name", "Player"),
    "long_name": ("long_name", "full_name", "Player", "Name"),
    "dob": ("dob",),
    "age": ("age", "Age"),
    "nationality_name": ("nationality_name", "country_name", "Nation"),
    "club_name": ("club_name", "Team"),
    "league_name": ("league_name", "League"),
    "player_positions": ("player_positions", "positions", "Position", "Alternative positions"),
    "overall": ("overall", "overall_rating", "Overall Score", "OVR"),
    "potential": ("potential", "Potential Score"),
    "value_eur": ("value_eur", "value", "Value"),
    "wage_eur": ("wage_eur", "wage", "Wage"),
    "pace": ("pace", "PAC"),
    "shooting": ("shooting", "SHO"),
    "passing": ("passing", "PAS"),
    "dribbling": ("dribbling", "DRI"),
    "defending": ("defending", "DEF"),
    "physic": ("physic", "PHY"),
    "attacking_crossing": ("attacking_crossing", "Crossing", "kaggle_crossing"),
    "attacking_finishing": ("attacking_finishing", "Finishing"),
    "attacking_heading_accuracy": ("attacking_heading_accuracy", "Heading Accuracy"),
    "attacking_short_passing": ("attacking_short_passing", "Short Passing"),
    "skill_dribbling": ("skill_dribbling", "Dribbling"),
    "skill_long_passing": ("skill_long_passing", "Long Passing"),
    "movement_acceleration": ("movement_acceleration", "Acceleration"),
    "movement_sprint_speed": ("movement_sprint_speed", "Sprint Speed"),
    "movement_agility": ("movement_agility", "Agility"),
    "movement_reactions": ("movement_reactions", "Reactions"),
    "power_stamina": ("power_stamina", "Stamina"),
    "power_strength": ("power_strength", "Strength"),
    "mentality_aggression": ("mentality_aggression", "Aggression"),
    "mentality_interceptions": ("mentality_interceptions", "Interceptions"),
    "mentality_positioning": ("mentality_positioning", "Positioning", "Attack Position"),
    "mentality_vision": ("mentality_vision", "Vision"),
    "mentality_composure": ("mentality_composure", "Composure"),
    "defending_marking_awareness": ("defending_marking_awareness", "Def Awareness", "Defensive Awareness"),
    "defending_standing_tackle": ("defending_standing_tackle", "Standing Tackle"),
    "goalkeeping_diving": ("goalkeeping_diving", "GK Diving"),
    "goalkeeping_handling": ("goalkeeping_handling", "GK Handling"),
    "goalkeeping_kicking": ("goalkeeping_kicking", "GK Kicking"),
    "goalkeeping_positioning": ("goalkeeping_positioning", "GK Positioning"),
    "goalkeeping_reflexes": ("goalkeeping_reflexes", "GK Reflexes"),
}


def first_existing(columns: set[str], names: tuple[str, ...]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for name in names:
        if name in columns:
            return name
        found = lowered.get(name.lower())
        if found is not None:
            return found
    return None


def euro_to_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("€", "").replace(",", "")
    if not text or text.lower() in {"nan", "none"}:
        return np.nan
    multiplier = 1.0
    if text[-1:].upper() == "M":
        multiplier = 1_000_000.0
        text = text[:-1]
    elif text[-1:].upper() == "K":
        multiplier = 1_000.0
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def fifa_year_from_path(path: Path) -> int | None:
    name = path.name
    match = re.search(r"players_(\d{2})\.csv$", name)
    if match and not name.startswith("female_"):
        return 2000 + int(match.group(1))
    if path.name == "male_players (legacy).csv" and "fifa-23-complete-player-dataset" in str(path):
        return 2023
    if path.name == "male_players.csv" and "fifa-23-complete-player-dataset" in str(path):
        return None
    if path.name == "all_fifa_players.csv":
        return 2024
    if path.name == "player-data-full-2025-june.csv":
        return 2025
    if path.name == "EAFC26-Men.csv":
        return 2026
    return None


def default_available_from(fifa_year: int) -> str:
    if fifa_year == 2025:
        return "2025-06-03"
    if fifa_year == 2026:
        return "2026-05-13"
    return f"{fifa_year - 1}-09-01"


def normalize_file(path: Path) -> pd.DataFrame | None:
    fifa_year = fifa_year_from_path(path)
    if fifa_year is None:
        return None
    frame = pd.read_csv(path, low_memory=False)
    if "GENDER" in frame.columns:
        frame = frame[frame["GENDER"].astype(str).str.lower().isin({"male", "m"})].copy()
    columns = set(frame.columns)

    out = pd.DataFrame(index=frame.index)
    out["fifa_year"] = fifa_year
    if "fifa_update_date" in frame.columns:
        out["available_from"] = pd.to_datetime(frame["fifa_update_date"], errors="coerce").dt.date.astype(str)
        out.loc[out["available_from"].eq("NaT"), "available_from"] = default_available_from(fifa_year)
    else:
        out["available_from"] = default_available_from(fifa_year)
    out["source_file"] = str(path)

    for target, names in ALIASES.items():
        source = first_existing(columns, names)
        out[target] = frame[source] if source is not None else np.nan

    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out["sofifa_id"] = pd.to_numeric(out["sofifa_id"], errors="coerce")
    numeric_cols = [col for col in OUTPUT_COLUMNS if col not in {"available_from", "source_file", "short_name", "long_name", "dob", "nationality_name", "club_name", "league_name", "player_positions"}]
    for col in numeric_cols:
        if col in {"value_eur", "wage_eur"}:
            out[col] = out[col].map(euro_to_float)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["sofifa_id", "overall"]).copy()
    out["sofifa_id"] = out["sofifa_id"].astype(int)
    return out[OUTPUT_COLUMNS]


def main() -> None:
    paths = sorted(DATA_ROOT.rglob("*.csv"))
    pieces = []
    for path in paths:
        normalized = normalize_file(path)
        if normalized is not None:
            pieces.append(normalized)
            print(f"Loaded {len(normalized):,} rows from {path}", flush=True)

    if not pieces:
        raise SystemExit("No SoFIFA/FIFA player rating files found.")

    output = pd.concat(pieces, ignore_index=True)
    output["available_from"] = pd.to_datetime(output["available_from"], errors="coerce")
    output = output.dropna(subset=["available_from"])
    output = output.sort_values(["sofifa_id", "available_from", "fifa_year"])
    output = output.drop_duplicates(subset=["sofifa_id", "available_from"], keep="last")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(output):,} rows to {OUTPUT_PATH}", flush=True)
    print(output.groupby("fifa_year").size().to_string(), flush=True)


if __name__ == "__main__":
    main()
