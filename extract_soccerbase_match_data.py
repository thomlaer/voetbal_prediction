#!/usr/bin/env python
"""Extract Soccerbase tournament matches, lineups, cards, and available stats."""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
import sys
import time
import unicodedata
import urllib.parse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


BASE_URL = "https://www.soccerbase.com"
DEFAULT_URL_FILE = "data/source_lists/soccerbase_sources.txt"
KAGGLE_DATASET = "aniss7/fifa-player-data-from-sofifa-2025-06-03"

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

TEAM_ALIASES = {
    "czechia": "czech republic",
    "china pr": "china",
    "cote d ivoire": "ivory coast",
    "cote divoire": "ivory coast",
    "dr congo": "congo dr",
    "iran": "iran",
    "ir iran": "iran",
    "korea republic": "south korea",
    "north korea": "korea dpr",
    "south korea": "korea republic",
    "turkiye": "turkey",
    "usa": "united states",
}


@dataclass
class MatchRow:
    soccerbase_game_id: str
    source_url: str
    tournament_url: str
    competition: str
    season: str
    tab: str
    stage: str
    date: str
    home_team: str
    away_team: str
    home_team_id: str
    away_team_id: str
    home_score: str
    away_score: str
    neutral: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Soccerbase lineups/cards/stats CSVs.")
    parser.add_argument("--url", action="append", default=[], help="Soccerbase tournament or competition URL.")
    parser.add_argument(
        "--url-file",
        action="append",
        default=[],
        help="Newline-delimited Soccerbase URL list. Lines starting with # are ignored.",
    )
    parser.add_argument("--lineups-output", default="data/extracted/soccerbase_lineups.csv")
    parser.add_argument("--stats-output", default="data/extracted/soccerbase_match_stats.csv")
    parser.add_argument("--cards-output", default="data/extracted/soccerbase_cards_events.csv")
    parser.add_argument("--report-output", default="outputs/soccerbase_extraction_report.csv")
    parser.add_argument("--fetched-games-cache", default="data/extracted/soccerbase_fetched_games.csv")
    parser.add_argument("--player-data", default="", help="Path to Kaggle fifa_players.csv or its directory.")
    parser.add_argument("--download-kaggle", action="store_true", help="Download Kaggle player data via kagglehub.")
    parser.add_argument("--delay-ms", type=int, default=200)
    parser.add_argument("--max-tournaments", type=int, default=0)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument(
        "--max-fetch-games",
        type=int,
        default=0,
        help="Cap the number of missing games fetched after incremental filtering. Use 0 for no cap.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Preserve existing output CSV rows and only fetch Soccerbase game ids that are not present yet.",
    )
    parser.add_argument(
        "--incremental-lookback-days",
        type=int,
        default=60,
        help="In incremental mode, only fetch missing completed matches from the last N days. Use 0 for no date window.",
    )
    parser.add_argument(
        "--worldcup-date-pages",
        default="",
        help="Optional YYYY-MM-DD:YYYY-MM-DD range; add Soccerbase daily result pages and keep only World Cup rows.",
    )
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def load_urls(args: argparse.Namespace) -> list[str]:
    urls = list(args.url)
    url_files = args.url_file
    if not urls and not url_files and Path(DEFAULT_URL_FILE).exists():
        url_files = [DEFAULT_URL_FILE]
    for file_name in url_files:
        for line in Path(file_name).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    urls.extend(worldcup_date_urls(getattr(args, "worldcup_date_pages", "")))
    return list(dict.fromkeys(urls))


def request_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def fetch_text(session: requests.Session, url: str, referer: str | None = None) -> str:
    headers = {}
    if referer:
        headers["Referer"] = referer
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = session.get(url, headers=headers, timeout=45)
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text
        except requests.RequestException as error:
            last_error = error
            if attempt == 4:
                break
            time.sleep(0.75 * attempt * attempt)
    raise RuntimeError(f"Could not fetch {url}: {last_error}") from last_error


def absolute_url(url: str, base: str = BASE_URL) -> str:
    return urllib.parse.urljoin(base, html.unescape(url))


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(value: object) -> str:
    text = normalize_text(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return TEAM_ALIASES.get(text, text)


def page_title(page_html: str) -> str:
    match = re.search(r"<title>(.*?)</title>", page_html, flags=re.I | re.S)
    return normalize_text(match.group(1)) if match else ""


def query_param(url: str, name: str) -> str:
    return urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get(name, [""])[0]


def is_results_date_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.path.endswith("/matches/results.sd") and bool(query_param(url, "date"))


def worldcup_date_urls(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    if ":" in text:
        start_text, end_text = text.split(":", 1)
    elif "," in text:
        start_text, end_text = text.split(",", 1)
    else:
        start_text = end_text = text
    start = date.fromisoformat(start_text.strip())
    end = date.fromisoformat(end_text.strip())
    if end < start:
        start, end = end, start
    return [
        f"{BASE_URL}/matches/results.sd?date={(start + timedelta(days=offset)).isoformat()}"
        for offset in range((end - start).days + 1)
    ]


def select_options(page_html: str, select_id: str) -> list[tuple[str, str]]:
    match = re.search(
        rf'<select[^>]+id="{re.escape(select_id)}"[^>]*>(.*?)</select>',
        page_html,
        flags=re.I | re.S,
    )
    if not match:
        return []
    options = []
    for option in re.finditer(r'<option[^>]+value="([^"]*)"[^>]*>(.*?)</option>', match.group(1), flags=re.I | re.S):
        options.append((html.unescape(option.group(1)), normalize_text(option.group(2))))
    return options


def season_year_value(label: str) -> int:
    match = re.search(r"\b(19|20)\d{2}\b", label)
    return int(match.group(0)) if match else 0


def discover_tournament_urls(session: requests.Session, source_url: str, delay_ms: int) -> list[tuple[str, str]]:
    source_url = source_url.split("#", 1)[0]
    text = fetch_text(session, source_url)
    if delay_ms:
        time.sleep(delay_ms / 1000)

    if is_results_date_url(source_url):
        return [(source_url, query_param(source_url, "date") or page_title(text))]

    source_tourn_id = query_param(source_url, "tourn_id")
    if source_tourn_id:
        title = page_title(text)
        label = title.split("|")[1].strip() if "|" in title else source_tourn_id
        return [(absolute_url(f"/tournaments/tournament.sd?tourn_id={source_tourn_id}"), label)]

    urls = []
    for value, label in select_options(text, "seasonSelect"):
        if not value or value == "0" or "all years" in label.lower():
            continue
        urls.append((absolute_url(f"/tournaments/tournament.sd?tourn_id={value}"), label))
    return urls


def discover_tab_urls(tournament_url: str, page_html: str) -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = [(tournament_url, "default")]
    for match in re.finditer(
        r'<a[^>]+href="([^"]*?/tournaments/tournament\.sd\?tourn_id=\d+&amp;tab=tab-\d+)"[^>]*>(.*?)</a>',
        page_html,
        flags=re.I | re.S,
    ):
        urls.append((absolute_url(match.group(1)), normalize_text(match.group(2))))
    return list(dict.fromkeys(urls))


def discover_round_urls(tab_url: str, page_html: str) -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    for match in re.finditer(
        r'<a[^>]+href="([^"]*?/tournaments/tournament_tab\.sd\?tourn_id=\d+&amp;tab_id=\d+&amp;roundTab=roundTab_\d+)"[^>]*>(.*?)</a>',
        page_html,
        flags=re.I | re.S,
    ):
        urls.append((absolute_url(match.group(1), tab_url), normalize_text(match.group(2))))
    return list(dict.fromkeys(urls))


def parse_match_date(row_html: str) -> str:
    date_href = re.search(r"/matches/results\.sd\?date=(\d{4}-\d{2}-\d{2})", row_html)
    if date_href:
        return date_href.group(1)
    text = normalize_text(row_html)
    match = re.search(r"\b(?:Mo|Tu|We|Th|Fr|Sa|Su)?\s*(\d{1,2})([A-Za-z]{3})(\d{4})\b", text)
    if not match:
        return ""
    month = MONTHS.get(match.group(2).lower())
    if not month:
        return ""
    return date(int(match.group(3)), month, int(match.group(1))).isoformat()


def parse_team(row_html: str, side: str) -> tuple[str, str]:
    match = re.search(
        rf'<td[^>]+class="[^"]*\bteam\b[^"]*\b{re.escape(side)}Team\b[^"]*"[^>]*>.*?'
        r'<a[^>]+team_id=(\d+)[^>]*>(.*?)</a>',
        row_html,
        flags=re.I | re.S,
    )
    if not match:
        return "", ""
    return normalize_text(match.group(2)), match.group(1)


def parse_score(row_html: str) -> tuple[str, str]:
    score_match = re.search(r'<td[^>]+class="[^"]*\bscore\b[^"]*"[^>]*>(.*?)</td>', row_html, flags=re.I | re.S)
    if not score_match:
        return "", ""
    values = re.findall(r"<em>(.*?)</em>", score_match.group(1), flags=re.I | re.S)
    if len(values) >= 2:
        return normalize_text(values[0]), normalize_text(values[1])
    text = normalize_text(score_match.group(1))
    match = re.search(r"(\d+)\s*-\s*(\d+)", text)
    return (match.group(1), match.group(2)) if match else ("", "")


def table_match_rows(page_html: str) -> Iterable[tuple[str, str, str]]:
    for table in re.finditer(
        r'<table[^>]+class="[^"]*\bsoccerGrid\b[^"]*"[^>]+id="([^"]+)"[^>]*>(.*?)</table>',
        page_html,
        flags=re.I | re.S,
    ):
        table_id = html.unescape(table.group(1))
        table_html = table.group(2)
        stage = ""
        for row in re.finditer(r"<tr\b[^>]*>.*?</tr>", table_html, flags=re.I | re.S):
            row_html = row.group(0)
            heading = re.search(r"<h2[^>]*>(.*?)</h2>", row_html, flags=re.I | re.S)
            if heading:
                stage = normalize_text(heading.group(1))
            if re.search(r'class="[^"]*\bmatch\b', row_html, flags=re.I):
                yield table_id, stage, row_html


def game_id_from_row(table_id: str, row_html: str) -> str:
    row_id_match = re.search(r'id="([^"]+)"', row_html, flags=re.I)
    if not row_id_match:
        return ""
    row_id = html.unescape(row_id_match.group(1))
    if row_id.startswith(table_id):
        return row_id[len(table_id) :]
    digits = re.findall(r"\d+", row_id)
    if not digits:
        return ""
    return digits[-1][-6:]


def parse_matches(page_html: str, source_url: str, tournament_url: str, competition: str, season: str, tab: str) -> list[MatchRow]:
    matches = []
    for table_id, stage, row_html in table_match_rows(page_html):
        game_id = game_id_from_row(table_id, row_html)
        if not game_id:
            continue
        home_team, home_team_id = parse_team(row_html, "home")
        away_team, away_team_id = parse_team(row_html, "away")
        home_score, away_score = parse_score(row_html)
        matches.append(
            MatchRow(
                soccerbase_game_id=game_id,
                source_url=source_url,
                tournament_url=tournament_url,
                competition=competition,
                season=season,
                tab=tab,
                stage=stage,
                date=parse_match_date(row_html),
                home_team=home_team,
                away_team=away_team,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_score=home_score,
                away_score=away_score,
                neutral="1" if "neutralVenues" in row_html else "0",
            )
        )
    return matches


def marker_text(fragment: str) -> str:
    def img_repl(match: re.Match[str]) -> str:
        img = match.group(0).lower()
        if "red-and-yellow" in img:
            return " [card:red_yellow] "
        if "red.png" in img or "red card" in img:
            return " [card:red] "
        if "yellow.png" in img or "yellow card" in img:
            return " [card:yellow] "
        return " "

    def anchor_repl(match: re.Match[str]) -> str:
        href = html.unescape(match.group(1))
        name = normalize_text(match.group(2))
        player_id = query_param(urllib.parse.urljoin(BASE_URL, href), "player_id")
        if not player_id:
            player_id_match = re.search(r"player_id=(\d+)", href)
            player_id = player_id_match.group(1) if player_id_match else ""
        return f" [[{player_id}|{name}]] "

    text = re.sub(r"<img\b[^>]*>", img_repl, fragment, flags=re.I | re.S)
    text = re.sub(r'<a[^>]+href="([^"]*player\.sd\?player_id=\d+[^"]*)"[^>]*>(.*?)</a>', anchor_repl, text, flags=re.I | re.S)
    text = re.sub(r"<strong[^>]*>.*?Subs not used:.*?</strong>", " Subs not used: ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def split_top_level_commas(text: str) -> list[str]:
    parts = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def card_markers(text: str) -> list[str]:
    return re.findall(r"\[card:([a-z_]+)\]", text)


def player_markers(text: str) -> list[tuple[str, str]]:
    return [(player_id, name.strip()) for player_id, name in re.findall(r"\[\[([^|]*)\|([^\]]+)\]\]", text)]


def base_lineup_row(match: MatchRow, team: str, opponent: str, is_home: bool, player_id: str, player_name: str) -> dict[str, object]:
    return {
        "source": "soccerbase",
        "soccerbase_game_id": match.soccerbase_game_id,
        "source_url": match.source_url,
        "tournament_url": match.tournament_url,
        "competition": match.competition,
        "season": match.season,
        "tab": match.tab,
        "stage": match.stage,
        "date": match.date,
        "home_team": match.home_team,
        "away_team": match.away_team,
        "team": team,
        "opponent": opponent,
        "is_home": int(is_home),
        "player_id": player_id,
        "player_name": player_name,
        "soccerbase_position": "",
        "shirt_number": "",
        "referee": "",
    }


def parse_active_players(match: MatchRow, cell_html: str, team: str, opponent: str, is_home: bool) -> tuple[list[dict], list[dict]]:
    lineups: list[dict] = []
    cards: list[dict] = []
    for segment in split_top_level_commas(marker_text(cell_html)):
        players = player_markers(segment)
        if not players:
            continue
        starter_id, starter_name = players[0]
        starter = base_lineup_row(match, team, opponent, is_home, starter_id, starter_name)
        before_sub = segment.split("(", 1)[0]
        starter_cards = card_markers(before_sub)
        starter.update(
            {
                "squad_status": "starter",
                "is_starter": 1,
                "is_sub_used": 0,
                "is_unused_sub": 0,
                "sub_on_minute": "",
                "sub_off_minute": "",
                "card_summary": ";".join(starter_cards),
            }
        )
        if len(players) > 1 and "(" in segment:
            sub_id, sub_name = players[1]
            parenthetical = segment.split("(", 1)[1]
            minute_match = re.search(r"\b(\d{1,3})\b", parenthetical)
            minute = minute_match.group(1) if minute_match else ""
            starter["sub_off_minute"] = minute
            sub_cards = card_markers(parenthetical)
            sub = base_lineup_row(match, team, opponent, is_home, sub_id, sub_name)
            sub.update(
                {
                    "squad_status": "sub_used",
                    "is_starter": 0,
                    "is_sub_used": 1,
                    "is_unused_sub": 0,
                    "sub_on_minute": minute,
                    "sub_off_minute": "",
                    "card_summary": ";".join(sub_cards),
                }
            )
            lineups.append(sub)
            for card_type in sub_cards:
                cards.append(card_event(match, team, opponent, is_home, sub_id, sub_name, card_type))
        lineups.append(starter)
        for card_type in starter_cards:
            cards.append(card_event(match, team, opponent, is_home, starter_id, starter_name, card_type))
    return lineups, cards


def parse_unused_players(match: MatchRow, cell_html: str, team: str, opponent: str, is_home: bool) -> list[dict]:
    output = []
    text = marker_text(cell_html)
    if "Subs not used:" in text:
        text = text.split("Subs not used:", 1)[1]
    for player_id, player_name in player_markers(text):
        row = base_lineup_row(match, team, opponent, is_home, player_id, player_name)
        row.update(
            {
                "squad_status": "unused_sub",
                "is_starter": 0,
                "is_sub_used": 0,
                "is_unused_sub": 1,
                "sub_on_minute": "",
                "sub_off_minute": "",
                "card_summary": "",
            }
        )
        output.append(row)
    return output


def card_event(match: MatchRow, team: str, opponent: str, is_home: bool, player_id: str, player_name: str, card_type: str) -> dict[str, object]:
    return {
        "source": "soccerbase",
        "soccerbase_game_id": match.soccerbase_game_id,
        "source_url": match.source_url,
        "competition": match.competition,
        "season": match.season,
        "tab": match.tab,
        "stage": match.stage,
        "date": match.date,
        "home_team": match.home_team,
        "away_team": match.away_team,
        "team": team,
        "opponent": opponent,
        "is_home": int(is_home),
        "player_id": player_id,
        "player_name": player_name,
        "card_type": card_type,
        "minute": "",
        "event_source": "lineup_bubble",
    }


def row_by_class(page_html: str, class_name: str) -> str:
    match = re.search(rf'<tr[^>]+class="[^"]*\b{re.escape(class_name)}\b[^"]*"[^>]*>(.*?)</tr>', page_html, flags=re.I | re.S)
    return match.group(1) if match else ""


def cell_by_class(row_html: str, class_name: str) -> str:
    match = re.search(rf'<td[^>]+class="[^"]*\b{re.escape(class_name)}\b[^"]*"[^>]*>(.*?)</td>', row_html, flags=re.I | re.S)
    return match.group(1) if match else ""


def parse_referee(page_html: str) -> str:
    match = re.search(r"<dt>\s*Referee:\s*</dt>\s*<dd>(.*?)</dd>", page_html, flags=re.I | re.S)
    return normalize_text(match.group(1)) if match else ""


def parse_lineup_bubble(match: MatchRow, page_html: str) -> tuple[list[dict], list[dict], str]:
    referee = parse_referee(page_html)
    players_row = row_by_class(page_html, "players")
    subs_row = row_by_class(page_html, "subs")
    output_lineups: list[dict] = []
    output_cards: list[dict] = []
    for class_name, team, opponent, is_home in [
        ("right", match.home_team, match.away_team, True),
        ("left", match.away_team, match.home_team, False),
    ]:
        active_rows, active_cards = parse_active_players(match, cell_by_class(players_row, class_name), team, opponent, is_home)
        unused_rows = parse_unused_players(match, cell_by_class(subs_row, class_name), team, opponent, is_home)
        output_lineups.extend(active_rows)
        output_lineups.extend(unused_rows)
        output_cards.extend(active_cards)
    for row in output_lineups:
        row["referee"] = referee
    return output_lineups, output_cards, referee


def blank_stats_row(match: MatchRow) -> dict[str, object]:
    return {
        "source": "soccerbase",
        "soccerbase_game_id": match.soccerbase_game_id,
        "source_url": match.source_url,
        "competition": match.competition,
        "season": match.season,
        "tab": match.tab,
        "stage": match.stage,
        "date": match.date,
        "home_team": match.home_team,
        "away_team": match.away_team,
        "home_score": match.home_score,
        "away_score": match.away_score,
        "home_possession": "",
        "away_possession": "",
        "home_shots_on_target": "",
        "away_shots_on_target": "",
        "home_shots_off_target": "",
        "away_shots_off_target": "",
        "home_corners": "",
        "away_corners": "",
        "stats_source": "not_available_in_static_html",
    }


class KaggleMatcher:
    def __init__(self, csv_path: Path | None):
        self.rows: list[dict[str, object]] = []
        self.columns: list[str] = []
        self.index: dict[tuple[str, str], list[int]] = {}
        self.name_index: dict[str, list[int]] = {}
        if csv_path and csv_path.exists():
            self._load(csv_path)

    def _load(self, csv_path: Path) -> None:
        df = pd.read_csv(csv_path, low_memory=False)
        self.columns = list(df.columns)
        self.rows = df.where(pd.notna(df), "").to_dict("records")
        for idx, row in enumerate(self.rows):
            team_keys = {
                normalize_key(row.get("national_team", "")),
                normalize_key(row.get("nationality", "")),
                normalize_key(row.get("country_name", "")),
            }
            team_keys.discard("")
            for key in self._candidate_keys(row):
                self.name_index.setdefault(key, []).append(idx)
                for team_key in team_keys:
                    self.index.setdefault((key, team_key), []).append(idx)

    def _candidate_keys(self, row: dict[str, object]) -> set[str]:
        keys = set()
        for value in [row.get("name", ""), row.get("full_name", "")]:
            norm = normalize_key(value)
            if norm:
                keys.add(norm)
                parts = norm.split()
                if len(parts) >= 2:
                    keys.add(f"{parts[0][0]} {parts[-1]}")
                    keys.add(" ".join(parts[-2:]))
                    for token in parts[1:]:
                        if len(token) >= 3:
                            keys.add(f"{parts[0][0]} {token}")
                            keys.add(f"{parts[0]} {token}")
        return keys

    def player_keys(self, player_name: str) -> set[str]:
        norm = normalize_key(player_name)
        keys = {norm} if norm else set()
        parts = norm.split()
        if len(parts) >= 2:
            if len(parts[0]) == 1:
                keys.add(f"{parts[0]} {parts[-1]}")
                keys.add(" ".join(parts[1:]))
            keys.add(f"{parts[0][0]} {parts[-1]}")
            keys.add(" ".join(parts[-2:]))
        return {key for key in keys if key}

    def match(self, player_name: str, team_name: str) -> tuple[dict[str, object], str]:
        team_key = normalize_key(team_name)
        player_parts = normalize_key(player_name).split()
        allow_best_rating_fallback = bool(player_parts and len(player_parts[0]) == 1)
        for key in self.player_keys(player_name):
            candidates = self.index.get((key, team_key), [])
            if len(candidates) == 1:
                return self.rows[candidates[0]], "name_team"
        for key in self.player_keys(player_name):
            candidates = self.name_index.get(key, [])
            unique = list(dict.fromkeys(candidates))
            if len(unique) == 1:
                return self.rows[unique[0]], "name_unique"
            if allow_best_rating_fallback and 1 < len(unique) <= 25:
                return self.best_rating_row(unique), "name_best_rating_fallback"
        return {}, ""

    def best_rating_row(self, indexes: list[int]) -> dict[str, object]:
        def rating(index: int) -> float:
            value = self.rows[index].get("overall_rating", "")
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = -1.0
            return number

        return self.rows[max(indexes, key=rating)]

    def enrich(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        if not self.rows:
            return rows
        output = []
        for row in rows:
            kaggle_row, quality = self.match(str(row.get("player_name", "")), str(row.get("team", "")))
            enriched = dict(row)
            enriched["kaggle_match_quality"] = quality
            enriched["kaggle_rating_snapshot_note"] = (
                "sofifa_2025_snapshot_not_historical" if kaggle_row else ""
            )
            for column in self.columns:
                enriched[f"kaggle_{column}"] = kaggle_row.get(column, "") if kaggle_row else ""
            output.append(enriched)
        return output


def resolve_player_data(args: argparse.Namespace) -> Path | None:
    if args.player_data:
        path = Path(args.player_data)
        if path.is_dir():
            candidates = list(path.rglob("player-data-full-*.csv")) or list(path.rglob("fifa_players.csv"))
            return candidates[0] if candidates else None
        return path
    if not args.download_kaggle:
        return None
    import kagglehub

    dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    candidates = list(dataset_path.rglob("player-data-full-*.csv")) or list(dataset_path.rglob("fifa_players.csv"))
    return candidates[0] if candidates else None


def write_csv(path: str, rows: list[dict[str, object]], preferred: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = preferred + [col for col in dict.fromkeys(k for row in rows for k in row) if col not in preferred]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_existing_csv(path: str) -> list[dict[str, object]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def existing_soccerbase_game_ids(paths: Iterable[str]) -> set[str]:
    game_ids: set[str] = set()
    for path in paths:
        csv_path = Path(path)
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "soccerbase_game_id" not in (reader.fieldnames or []):
                continue
            fieldnames = set(reader.fieldnames or [])
            if not {"home_score", "away_score"}.issubset(fieldnames):
                # Lineups/cards and older cache rows do not prove that final
                # post-match data was fetched. They must not block a refresh.
                continue
            for row in reader:
                if not str(row.get("home_score", "")).strip() or not str(row.get("away_score", "")).strip():
                    continue
                game_id = str(row.get("soccerbase_game_id", "")).strip()
                if game_id:
                    game_ids.add(game_id)
    return game_ids


def existing_lineup_game_ids(path: str, min_rows_per_game: int = 18) -> set[str]:
    """Game ids with enough stored lineup rows to count as lineup-fetched.

    Match stats can arrive before the lineup bubble is available or before our
    parser understands it. Those stats rows must not block a later lineup retry.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        return set()
    counts: dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "soccerbase_game_id" not in fieldnames or "player_name" not in fieldnames:
            return set()
        for row in reader:
            game_id = str(row.get("soccerbase_game_id", "")).strip()
            player_name = str(row.get("player_name", "")).strip()
            if not game_id or not player_name:
                continue
            counts[game_id] = counts.get(game_id, 0) + 1
    return {game_id for game_id, count in counts.items() if count >= min_rows_per_game}


def has_completed_score(row: dict[str, object]) -> bool:
    return bool(str(row.get("home_score", "")).strip() and str(row.get("away_score", "")).strip())


def parsed_match_date(match: MatchRow) -> date | None:
    value = pd.to_datetime(match.date, errors="coerce")
    if pd.isna(value):
        return None
    return value.date()


def filter_incremental_matches(
    matches: list[MatchRow],
    skip_game_ids: set[str],
    lookback_days: int,
) -> tuple[list[MatchRow], dict[str, int]]:
    today = date.today()
    cutoff = today - timedelta(days=lookback_days) if lookback_days > 0 else None

    filtered: list[MatchRow] = []
    counts = {
        "existing": 0,
        "future": 0,
        "older_than_window": 0,
        "missing_completed": 0,
        "unknown_date": 0,
    }
    for match in matches:
        if str(match.soccerbase_game_id) in skip_game_ids:
            counts["existing"] += 1
            continue
        match_day = parsed_match_date(match)
        if match_day is None:
            counts["unknown_date"] += 1
            continue
        if match_day > today:
            counts["future"] += 1
            continue
        if cutoff is not None and match_day < cutoff:
            counts["older_than_window"] += 1
            continue
        filtered.append(match)
        counts["missing_completed"] += 1
    return filtered, counts


def merge_rows(
    existing: list[dict[str, object]],
    new_rows: list[dict[str, object]],
    key_fields: list[str],
    replace_blank_score: bool = False,
) -> list[dict[str, object]]:
    output = list(existing)
    seen = {}
    for index, row in enumerate(existing):
        seen[tuple(str(row.get(field, "")).strip() for field in key_fields)] = index
    for row in new_rows:
        key = tuple(str(row.get(field, "")).strip() for field in key_fields)
        if key in seen:
            existing_index = seen[key]
            if replace_blank_score and has_completed_score(row) and not has_completed_score(output[existing_index]):
                output[existing_index] = row
            continue
        output.append(row)
        seen[key] = len(output) - 1
    return output


def cache_rows_from_lineups(lineups: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for row in lineups:
        game_id = str(row.get("soccerbase_game_id", "")).strip()
        if not game_id or game_id in seen:
            continue
        seen.add(game_id)
        rows.append(
            {
                "soccerbase_game_id": game_id,
                "date": row.get("date", ""),
                "home_team": row.get("home_team", ""),
                "away_team": row.get("away_team", ""),
                "status": "lineup_fetched",
            }
        )
    return rows


def scrape(args: argparse.Namespace, skip_game_ids: set[str] | None = None) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    session = request_session()
    tournament_targets: list[tuple[str, str]] = []
    for source_url in load_urls(args):
        print(f"Discovering {source_url}")
        try:
            tournament_targets.extend(discover_tournament_urls(session, source_url, args.delay_ms))
        except Exception as error:
            if not args.skip_errors:
                raise
            print(f"  skipped discovery: {error}", file=sys.stderr)
    tournament_targets = list(dict.fromkeys(tournament_targets))
    if args.max_tournaments:
        date_targets = [target for target in tournament_targets if is_results_date_url(target[0])]
        regular_targets = [target for target in tournament_targets if not is_results_date_url(target[0])]
        tournament_targets = regular_targets[: args.max_tournaments] + date_targets

    matches_by_id: dict[str, MatchRow] = {}
    report_rows: list[dict] = []
    for index, (tournament_url, season_label) in enumerate(tournament_targets, 1):
        print(f"[{index}/{len(tournament_targets)}] Fetching tournament {tournament_url}")
        try:
            tournament_html = fetch_text(session, tournament_url)
            title = page_title(tournament_html)
            competition = title.split(" Betting |", 1)[0].strip() if " Betting |" in title else title
            tab_urls = discover_tab_urls(tournament_url, tournament_html)
            seen_tab_urls = set()
            tournament_match_count = 0
            for tab_url, tab_label in tab_urls:
                if tab_url in seen_tab_urls:
                    continue
                seen_tab_urls.add(tab_url)
                tab_html = tournament_html if tab_url == tournament_url else fetch_text(session, tab_url, tournament_url)
                pages = [(tab_url, tab_label, tab_html)]
                for round_url, round_label in discover_round_urls(tab_url, tab_html):
                    if round_url not in seen_tab_urls:
                        seen_tab_urls.add(round_url)
                        pages.append((round_url, f"{tab_label}: {round_label}", fetch_text(session, round_url, tab_url)))
                        if args.delay_ms:
                            time.sleep(args.delay_ms / 1000)
                for page_url, page_label, page_html in pages:
                    page_matches = parse_matches(page_html, page_url, tournament_url, competition, season_label, page_label)
                    if is_results_date_url(page_url):
                        for match in page_matches:
                            match.competition = match.stage
                        page_matches = [
                            match for match in page_matches if normalize_key(match.competition) == "world cup"
                        ]
                    for match in page_matches:
                        matches_by_id.setdefault(match.soccerbase_game_id, match)
                    tournament_match_count += len(page_matches)
                if args.delay_ms:
                    time.sleep(args.delay_ms / 1000)
            report_rows.append(
                {
                    "url": tournament_url,
                    "season": season_label,
                    "competition": competition,
                    "tabs": len(tab_urls),
                    "matches_found": tournament_match_count,
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as error:
            if not args.skip_errors:
                raise
            report_rows.append(
                {
                    "url": tournament_url,
                    "season": season_label,
                    "competition": "",
                    "tabs": 0,
                    "matches_found": 0,
                    "status": "error",
                    "error": str(error),
                }
            )
            print(f"  skipped tournament: {error}", file=sys.stderr)
        if args.delay_ms:
            time.sleep(args.delay_ms / 1000)

    matches = list(matches_by_id.values())
    if args.max_games:
        matches = matches[: args.max_games]
    if args.incremental:
        original_count = len(matches)
        matches, incremental_counts = filter_incremental_matches(
            matches,
            skip_game_ids or set(),
            args.incremental_lookback_days,
        )
        if args.max_fetch_games and len(matches) > args.max_fetch_games:
            matches = sorted(
                matches,
                key=lambda match: (parsed_match_date(match) or date.min, match.soccerbase_game_id),
                reverse=True,
            )[: args.max_fetch_games]
        report_rows.append(
            {
                "url": "",
                "season": "",
                "competition": "incremental_cache",
                "tabs": "",
                "matches_found": original_count,
                "status": "incremental_filter",
                "error": (
                    f"existing={incremental_counts['existing']}; "
                    f"future={incremental_counts['future']}; "
                    f"older_than_window={incremental_counts['older_than_window']}; "
                    f"unknown_date={incremental_counts['unknown_date']}; "
                    f"missing_completed={incremental_counts['missing_completed']}"
                ),
            }
        )
        report_rows.append(
            {
                "url": "",
                "season": "",
                "competition": "incremental_cache",
                "tabs": "",
                "matches_found": len(matches),
                "status": "fetch_missing_completed_games",
                "error": "",
            }
        )
        print(
            "Incremental mode: "
            f"existing={incremental_counts['existing']:,}, "
            f"future={incremental_counts['future']:,}, "
            f"older_than_window={incremental_counts['older_than_window']:,}, "
            f"missing_completed={incremental_counts['missing_completed']:,}. "
            f"Fetching {len(matches):,} missing completed games."
        )
    lineups: list[dict] = []
    cards: list[dict] = []
    stats = [blank_stats_row(match) for match in matches]
    for index, match in enumerate(matches, 1):
        print(f"[{index}/{len(matches)}] Lineup {match.soccerbase_game_id} {match.home_team} - {match.away_team}")
        try:
            bubble_url = absolute_url(f"/teams/lineup_bubble.sd?id_game={match.soccerbase_game_id}")
            bubble_html = fetch_text(session, bubble_url, match.source_url)
            match_lineups, match_cards, _ = parse_lineup_bubble(match, bubble_html)
            lineups.extend(match_lineups)
            cards.extend(match_cards)
        except Exception as error:
            if not args.skip_errors:
                raise
            report_rows.append(
                {
                    "url": match.source_url,
                    "season": match.season,
                    "competition": match.competition,
                    "tabs": "",
                    "matches_found": "",
                    "status": "lineup_error",
                    "error": f"{match.soccerbase_game_id}: {error}",
                }
            )
            print(f"  skipped lineup: {error}", file=sys.stderr)
        if args.delay_ms:
            time.sleep(args.delay_ms / 1000)
    return lineups, stats, cards, report_rows


def main() -> int:
    args = parse_args()
    player_data = resolve_player_data(args)
    matcher = KaggleMatcher(player_data)
    if player_data:
        print(f"Using player data: {player_data}")
    else:
        print("No Kaggle player data loaded.")

    existing_lineups = read_existing_csv(args.lineups_output) if args.incremental else []
    existing_stats = read_existing_csv(args.stats_output) if args.incremental else []
    existing_cards = read_existing_csv(args.cards_output) if args.incremental else []
    existing_cache = read_existing_csv(args.fetched_games_cache) if args.incremental else []
    skip_game_ids = existing_lineup_game_ids(args.lineups_output) if args.incremental else set()

    lineups, stats, cards, report_rows = scrape(args, skip_game_ids)
    lineups = matcher.enrich(lineups)
    if args.incremental:
        lineups = merge_rows(
            existing_lineups,
            lineups,
            ["soccerbase_game_id", "team", "player_id", "player_name", "squad_status"],
        )
        stats = merge_rows(existing_stats, stats, ["soccerbase_game_id"], replace_blank_score=True)
        cards = merge_rows(
            existing_cards,
            cards,
            ["soccerbase_game_id", "team", "player_id", "player_name", "card_type", "minute"],
        )
        fetched_cache = merge_rows(
            existing_cache,
            cache_rows_from_lineups(lineups),
            ["soccerbase_game_id"],
        )
    else:
        fetched_cache = cache_rows_from_lineups(lineups)

    write_csv(
        args.lineups_output,
        lineups,
        [
            "source",
            "soccerbase_game_id",
            "date",
            "competition",
            "season",
            "tab",
            "stage",
            "home_team",
            "away_team",
            "team",
            "opponent",
            "is_home",
            "player_id",
            "player_name",
            "squad_status",
            "is_starter",
            "is_sub_used",
            "is_unused_sub",
            "sub_on_minute",
            "sub_off_minute",
            "soccerbase_position",
            "shirt_number",
            "referee",
            "card_summary",
            "kaggle_match_quality",
            "kaggle_rating_snapshot_note",
        ],
    )
    write_csv(
        args.stats_output,
        stats,
        [
            "source",
            "soccerbase_game_id",
            "date",
            "competition",
            "season",
            "tab",
            "stage",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "home_possession",
            "away_possession",
            "home_shots_on_target",
            "away_shots_on_target",
            "home_shots_off_target",
            "away_shots_off_target",
            "home_corners",
            "away_corners",
            "stats_source",
        ],
    )
    write_csv(
        args.cards_output,
        cards,
        [
            "source",
            "soccerbase_game_id",
            "date",
            "competition",
            "season",
            "tab",
            "stage",
            "home_team",
            "away_team",
            "team",
            "opponent",
            "is_home",
            "player_id",
            "player_name",
            "card_type",
            "minute",
            "event_source",
        ],
    )
    write_csv(
        args.report_output,
        report_rows,
        ["url", "season", "competition", "tabs", "matches_found", "status", "error"],
    )
    write_csv(
        args.fetched_games_cache,
        fetched_cache,
        ["soccerbase_game_id", "date", "home_team", "away_team", "status"],
    )
    print(f"Wrote {len(lineups):,} lineup/player rows to {args.lineups_output}")
    print(f"Wrote {len(stats):,} match stat rows to {args.stats_output}")
    print(f"Wrote {len(cards):,} card rows to {args.cards_output}")
    print(f"Wrote report to {args.report_output}")
    print(f"Wrote {len(fetched_cache):,} fetched-game cache rows to {args.fetched_games_cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
