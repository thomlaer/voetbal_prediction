#!/usr/bin/env python
"""Fill Soccerbase match stat columns from the match-info AJAX endpoint.

The main Soccerbase tournament pages only include disabled placeholder bars.
The expanded "i" match-info panel is loaded from:

    /matches/additional_information.sd?id_game=<soccerbase_game_id>

That endpoint contains the actual possession, shots, and corners bars when
Soccerbase has them.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import sys
import tempfile
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import requests


BASE_URL = "https://www.soccerbase.com"
DEFAULT_INPUT = "data/extracted/soccerbase_match_stats.csv"
DEFAULT_REPORT = "outputs/soccerbase_additional_stats_report.csv"

STAT_COLUMNS = [
    "home_possession",
    "away_possession",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_shots_off_target",
    "away_shots_off_target",
    "home_corners",
    "away_corners",
]

BAR_LABELS = {
    "possession": ("home_possession", "away_possession"),
    "shots on target": ("home_shots_on_target", "away_shots_on_target"),
    "shots off target": ("home_shots_off_target", "away_shots_off_target"),
    "corners": ("home_corners", "away_corners"),
}

THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Soccerbase expanded match stats.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Existing soccerbase_match_stats CSV.")
    parser.add_argument(
        "--output",
        default=DEFAULT_INPUT,
        help="Output CSV. Defaults to overwriting the input atomically.",
    )
    parser.add_argument("--report-output", default=DEFAULT_REPORT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay-ms", type=int, default=100)
    parser.add_argument("--max-games", type=int, default=0, help="Only fetch the first N unique game ids.")
    parser.add_argument("--force", action="store_true", help="Refetch rows even when all stat fields are filled.")
    return parser.parse_args()


def request_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def get_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = request_session()
        THREAD_LOCAL.session = session
    return session


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(label: str) -> str:
    label = normalize_text(label).lower()
    label = label.replace("%", "")
    return re.sub(r"\s+", " ", label).strip()


def stat_value(value: str) -> str:
    value = normalize_text(value)
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return match.group(0) if match else ""


def parse_stat_bars(page_html: str) -> dict[str, str]:
    stats: dict[str, str] = {}
    for bar_match in re.finditer(r'<div[^>]+class="[^"]*\bbar\b[^"]*"[^>]*>(.*?)</div><!-- \.barWrapper -->', page_html, flags=re.I | re.S):
        bar_html = bar_match.group(1)
        if re.search(r"No data available", bar_html, flags=re.I):
            continue
        label_match = re.search(r'<div[^>]+class="[^"]*\blabel\b[^"]*"[^>]*>(.*?)</div>', bar_html, flags=re.I | re.S)
        left_match = re.search(r'<div[^>]+class="[^"]*\bleft\b[^"]*"[^>]*>(.*?)</div>', bar_html, flags=re.I | re.S)
        right_match = re.search(r'<div[^>]+class="[^"]*\bright\b[^"]*"[^>]*>(.*?)</div>', bar_html, flags=re.I | re.S)
        if not (label_match and left_match and right_match):
            continue
        label = normalize_label(label_match.group(1))
        for needle, (home_col, away_col) in BAR_LABELS.items():
            if needle in label:
                stats[home_col] = stat_value(left_match.group(1))
                stats[away_col] = stat_value(right_match.group(1))
                break
    return stats


def additional_info_url(game_id: str) -> str:
    return urllib.parse.urljoin(BASE_URL, f"/matches/additional_information.sd?id_game={urllib.parse.quote(game_id)}")


def fetch_stats(game: dict[str, str], delay_ms: int) -> dict[str, object]:
    game_id = game["soccerbase_game_id"]
    url = additional_info_url(game_id)
    headers = {}
    if game.get("source_url"):
        headers["Referer"] = game["source_url"]
    if delay_ms:
        time.sleep(delay_ms / 1000)
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = get_session().get(url, headers=headers, timeout=45)
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            stats = parse_stat_bars(response.text)
            status = "ok" if stats else "no_stats_in_additional_information"
            return {
                "soccerbase_game_id": game_id,
                "status": status,
                "stats": stats,
                "error": "",
                "url": url,
            }
        except requests.RequestException as error:
            last_error = error
            time.sleep(0.75 * attempt * attempt)
    return {
        "soccerbase_game_id": game_id,
        "status": "error",
        "stats": {},
        "error": str(last_error),
        "url": url,
    }


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        columns = list(reader.fieldnames or [])
    return rows, columns


def atomic_write_csv(path: Path, rows: Iterable[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def rows_need_fetch(rows: list[dict[str, str]], force: bool) -> list[dict[str, str]]:
    seen = set()
    games = []
    for row in rows:
        game_id = row.get("soccerbase_game_id", "").strip()
        if not game_id or game_id in seen:
            continue
        seen.add(game_id)
        if not force and all(row.get(column, "").strip() for column in STAT_COLUMNS):
            continue
        games.append(row)
    return games


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "soccerbase_game_id",
        "date",
        "home_team",
        "away_team",
        "status",
        "fields_found",
        "error",
        "url",
    ]
    atomic_write_csv(path, rows, columns)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report_output)

    rows, columns = read_csv(input_path)
    for column in STAT_COLUMNS + ["stats_source"]:
        if column not in columns:
            columns.append(column)

    games = rows_need_fetch(rows, args.force)
    if args.max_games:
        games = games[: args.max_games]

    print(f"Loaded {len(rows):,} rows from {input_path}")
    print(f"Fetching expanded Soccerbase stats for {len(games):,} unique games")

    results: dict[str, dict[str, object]] = {}
    report_rows: list[dict[str, object]] = []
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(fetch_stats, game, args.delay_ms): game for game in games}
        for index, future in enumerate(as_completed(future_map), 1):
            game = future_map[future]
            result = future.result()
            game_id = str(result["soccerbase_game_id"])
            results[game_id] = result
            stats = result["stats"]
            report_rows.append(
                {
                    "soccerbase_game_id": game_id,
                    "date": game.get("date", ""),
                    "home_team": game.get("home_team", ""),
                    "away_team": game.get("away_team", ""),
                    "status": result["status"],
                    "fields_found": len(stats) if isinstance(stats, dict) else 0,
                    "error": result["error"],
                    "url": result["url"],
                }
            )
            if index == 1 or index % 100 == 0 or index == len(games):
                ok_count = sum(1 for item in results.values() if item["status"] == "ok")
                print(f"[{index:,}/{len(games):,}] stats found for {ok_count:,} games")

    changed = 0
    no_stats = 0
    errors = 0
    for row in rows:
        game_id = row.get("soccerbase_game_id", "").strip()
        result = results.get(game_id)
        if not result:
            continue
        if result["status"] == "ok":
            for column, value in result["stats"].items():
                row[column] = value
            row["stats_source"] = "additional_information"
            changed += 1
        elif result["status"] == "error":
            row["stats_source"] = "additional_information_error"
            errors += 1
        else:
            row["stats_source"] = "additional_information_no_stats"
            no_stats += 1

    atomic_write_csv(output_path, rows, columns)
    write_report(report_path, report_rows)

    print(f"Wrote {len(rows):,} rows to {output_path}")
    print(f"Updated stats for {changed:,} rows; no stats for {no_stats:,}; errors for {errors:,}")
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
