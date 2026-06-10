"""
Extract compact geo lookup tables from the SimpleMaps worldcities file.

The training script uses these CSVs as an offline, API-free source for venue
coordinates, rough team-base coordinates, travel distance and approximate
timezone shifts.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from train_xgboost_worldcup import normalize_name


DEFAULT_ZIP = Path("simplemaps_worldcities_basicv1.901.zip")
DEFAULT_CSV = Path("simplemaps_worldcities_basicv1.901/worldcities.csv")
DEFAULT_OUTPUT_DIR = Path("data/extracted")

GEO_COUNTRY_ALIASES = {
    "congo kinshasa": "congo dr",
    "congo brazzaville": "congo",
    "cote d’ivoire": "ivory coast",
    "cote d ivoire": "ivory coast",
    "korea south": "south korea",
    "korea north": "north korea",
    "cape verde": "cape verde",
    "turkiye": "turkey",
}

TEAM_BASE_OVERRIDES = {
    "england": "London",
    "scotland": "Edinburgh",
    "wales": "Cardiff",
    "northern ireland": "Belfast",
    "gibraltar": "Gibraltar",
    "faroe islands": "Torshavn",
    "hong kong": "Hong Kong",
    "macau": "Macau",
    "taiwan": "Taipei",
}


def normalize_geo_country(value: object) -> str:
    key = normalize_name(value)
    return GEO_COUNTRY_ALIASES.get(key, key)


def capital_priority(value: object) -> int:
    text = str(value).strip().lower()
    if text == "primary":
        return 0
    if text == "admin":
        return 1
    if text == "minor":
        return 2
    return 3


def read_worldcities_csv(handle: object) -> pd.DataFrame:
    return pd.read_csv(
        handle,
        usecols=[
            "city",
            "city_ascii",
            "lat",
            "lng",
            "country",
            "iso2",
            "iso3",
            "admin_name",
            "capital",
            "population",
            "id",
        ],
    )


def load_worldcities(zip_path: Path | None, csv_path: Path | None) -> pd.DataFrame:
    if csv_path is not None and csv_path.exists():
        cities = read_worldcities_csv(csv_path)
    elif zip_path is not None and zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open("worldcities.csv") as handle:
                cities = read_worldcities_csv(handle)
    else:
        raise FileNotFoundError(
            "Worldcities source not found. Expected either "
            f"{csv_path or DEFAULT_CSV} or {zip_path or DEFAULT_ZIP}."
        )
    cities["city_key"] = cities["city_ascii"].fillna(cities["city"]).map(normalize_name)
    cities["country_key"] = cities["country"].map(normalize_geo_country)
    cities["population"] = pd.to_numeric(cities["population"], errors="coerce").fillna(0.0)
    cities["capital_priority"] = cities["capital"].map(capital_priority)
    cities = cities.dropna(subset=["lat", "lng", "city_key", "country_key"])
    return cities


def build_city_locations(cities: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "city",
        "city_ascii",
        "city_key",
        "country",
        "country_key",
        "iso2",
        "iso3",
        "admin_name",
        "capital",
        "population",
        "lat",
        "lng",
    ]
    ranked = cities.sort_values(["city_key", "country_key", "population"], ascending=[True, True, False])
    return ranked.drop_duplicates(["city_key", "country_key"], keep="first")[columns].reset_index(drop=True)


def build_country_representatives(cities: pd.DataFrame, city_locations: pd.DataFrame) -> pd.DataFrame:
    ranked = cities.sort_values(
        ["country_key", "capital_priority", "population"],
        ascending=[True, True, False],
    )
    reps = ranked.drop_duplicates("country_key", keep="first").copy()

    override_rows = []
    city_by_name = city_locations.sort_values("population", ascending=False).drop_duplicates("city_key", keep="first")
    city_lookup = city_by_name.set_index("city_key")
    for country_key, city_name in TEAM_BASE_OVERRIDES.items():
        city_key = normalize_name(city_name)
        if city_key not in city_lookup.index:
            continue
        row = city_lookup.loc[city_key].copy()
        row["country_key"] = country_key
        row["country"] = country_key.title()
        row["capital"] = "override"
        override_rows.append(row)

    if override_rows:
        reps = pd.concat([reps, pd.DataFrame(override_rows)], ignore_index=True)
        reps = reps.drop_duplicates("country_key", keep="last")

    columns = [
        "country",
        "country_key",
        "city",
        "city_ascii",
        "city_key",
        "iso2",
        "iso3",
        "admin_name",
        "capital",
        "population",
        "lat",
        "lng",
    ]
    return reps[columns].sort_values("country_key").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract offline geo lookup tables from SimpleMaps worldcities.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cities = load_worldcities(args.zip, args.csv)
    city_locations = build_city_locations(cities)
    country_reps = build_country_representatives(cities, city_locations)

    city_path = args.output_dir / "worldcities_city_locations.csv"
    country_path = args.output_dir / "worldcities_country_representatives.csv"
    city_locations.to_csv(city_path, index=False)
    country_reps.to_csv(country_path, index=False)

    print(f"Wrote {len(city_locations):,} city rows to {city_path}")
    print(f"Wrote {len(country_reps):,} country representatives to {country_path}")


if __name__ == "__main__":
    main()
