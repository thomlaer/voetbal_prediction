#!/usr/bin/env python
"""Extract national-team tournament squads from Wikipedia squad pages."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd
from lxml import html as lxml_html


DEFAULT_URLS = [
    "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup_squads",
    "https://en.wikipedia.org/wiki/2018_FIFA_World_Cup_squads",
    "https://en.wikipedia.org/wiki/2014_FIFA_World_Cup_squads",
    "https://en.wikipedia.org/wiki/UEFA_Euro_2020_squads",
    "https://en.wikipedia.org/wiki/2015_Copa_America_squads",
    "https://en.wikipedia.org/wiki/2019_Africa_Cup_of_Nations_squads",
    "https://en.wikipedia.org/wiki/2011_AFC_Asian_Cup_squads",
]

HEADERS = [
    "source",
    "source_url",
    "resolved_url",
    "source_page_title",
    "competition",
    "year",
    "team",
    "source_heading",
    "shirt_number",
    "position",
    "player",
    "date_of_birth",
    "age",
    "caps",
    "goals",
    "club",
    "raw_date_of_birth_age",
    "raw_player",
    "raw_club",
]


TEAM_HEADING_DENYLIST = {
    "contents",
    "references",
    "external links",
    "notes",
    "statistics",
    "group stage",
    "knockout stage",
    "squads",
    "venues",
    "match officials",
    "draw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract tournament squad tables from Wikipedia to CSV.",
    )
    parser.add_argument("--url", action="append", default=[], help="Wikipedia page URL.")
    parser.add_argument(
        "--url-file",
        action="append",
        default=[],
        help="Newline-delimited URL file. Lines starting with # are ignored.",
    )
    parser.add_argument(
        "--output",
        default="data/extracted/wikipedia_tournament_squads.csv",
        help="Output squad CSV.",
    )
    parser.add_argument(
        "--report",
        default="outputs/wikipedia_tournament_squads_report.csv",
        help="Output page-level report CSV.",
    )
    parser.add_argument("--delay-ms", type=int, default=250, help="Delay between page requests.")
    parser.add_argument("--skip-errors", action="store_true", help="Continue after fetch/parse errors.")
    return parser.parse_args()


def load_urls(args: argparse.Namespace) -> list[str]:
    urls = list(args.url)
    for file_name in args.url_file:
        for line in Path(file_name).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    if not urls:
        urls = DEFAULT_URLS
    return list(dict.fromkeys(urls))


def fetch_html(url: str) -> tuple[str, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.geturl(), response.read().decode(charset, errors="replace")
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt == 4:
                break
            time.sleep(0.75 * attempt * attempt)
    raise RuntimeError(f"Could not fetch {url}: {last_error}") from last_error


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_key(value: object) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def page_title(document: lxml_html.HtmlElement) -> str:
    title_values = document.xpath("//title/text()")
    if not title_values:
        return ""
    return clean_text(title_values[0].replace(" - Wikipedia", ""))


def competition_and_year(title: str, resolved_url: str) -> tuple[str, str]:
    base = re.sub(r"\s+squads?$", "", title, flags=re.IGNORECASE).strip()
    year_match = re.search(r"\b(19|20)\d{2}\b", base)
    year = year_match.group(0) if year_match else ""
    competition = base
    if year:
        competition = re.sub(rf"\b{re.escape(year)}\b", "", base).strip()
    competition = re.sub(r"\s+", " ", competition).strip(" -")
    if not competition:
        slug = urllib.parse.unquote(urllib.parse.urlparse(resolved_url).path.rsplit("/", 1)[-1])
        competition = re.sub(r"_squads?$", "", slug, flags=re.IGNORECASE).replace("_", " ")
    return competition, year


def discover_squad_url(document: lxml_html.HtmlElement, base_url: str) -> str | None:
    links = document.xpath('//a[contains(@href, "squads")]/@href')
    for href in links:
        if not href.startswith("/wiki/") or ":" in href:
            continue
        if href.lower().endswith("_squads") or "_squads#" in href.lower():
            return urllib.parse.urljoin(base_url, href.split("#", 1)[0])
    return None


def flatten_columns(df: pd.DataFrame) -> list[str]:
    if isinstance(df.columns, pd.MultiIndex):
        flattened = []
        for column in df.columns:
            parts = [clean_text(part) for part in column if "Unnamed:" not in str(part)]
            flattened.append(" ".join(dict.fromkeys(part for part in parts if part)))
        return flattened
    return [clean_text(column) for column in df.columns]


def find_column(columns: list[str], patterns: list[str]) -> str | None:
    normalized = {column: normalize_key(column) for column in columns}
    for pattern in patterns:
        for column, key in normalized.items():
            if re.search(pattern, key):
                return column
    return None


def nearest_heading(table: lxml_html.HtmlElement) -> str:
    headings = table.xpath("(preceding::h2 | preceding::h3 | preceding::h4)[last()]")
    if not headings:
        return ""
    heading = clean_text(headings[0].text_content().replace("[edit]", ""))
    heading = re.sub(r"^\d+\s+", "", heading).strip()
    return heading


def looks_like_squad_table(columns: list[str]) -> bool:
    keys = [normalize_key(column) for column in columns]
    has_player = any(key == "player" or key.endswith(" player") for key in keys)
    has_club = any(key == "club" or key.endswith(" club") for key in keys)
    has_position = any(key in {"pos", "position"} for key in keys)
    has_number = any(key in {"no", "number", "shirt number"} for key in keys)
    return has_player and has_club and (has_position or has_number)


def parse_birth_age(value: str) -> tuple[str, str]:
    raw = clean_text(value)
    age_match = re.search(r"\b(?:aged?|age)\s*(\d{1,2})\b", raw, flags=re.IGNORECASE)
    age = age_match.group(1) if age_match else ""
    dob = re.sub(r"\s*\((?:aged?|age)\s*\d{1,2}[^)]*\)", "", raw, flags=re.IGNORECASE).strip()
    return dob, age


def integer_text(value: object) -> str:
    text = clean_text(value)
    match = re.search(r"-?\d+", text)
    return match.group(0) if match else ""


def table_to_rows(
    table: lxml_html.HtmlElement,
    source_url: str,
    resolved_url: str,
    title: str,
    competition: str,
    year: str,
) -> list[dict[str, str]]:
    table_html = lxml_html.tostring(table, encoding="unicode")
    frames = pd.read_html(StringIO(table_html))
    if not frames:
        return []
    df = frames[0]
    df.columns = flatten_columns(df)
    columns = list(df.columns)
    if not looks_like_squad_table(columns):
        return []

    player_col = find_column(columns, [r"^player$"])
    club_col = find_column(columns, [r"^club$"])
    position_col = find_column(columns, [r"^(pos|position)$"])
    number_col = find_column(columns, [r"^(no|number|shirt number)$"])
    birth_col = find_column(columns, [r"date of birth", r"birth"])
    caps_col = find_column(columns, [r"^caps?$", r"appearances"])
    goals_col = find_column(columns, [r"^goals?$"])
    if not player_col or not club_col:
        return []

    team = nearest_heading(table)
    if normalize_key(team) in TEAM_HEADING_DENYLIST:
        team = ""

    rows: list[dict[str, str]] = []
    for _, item in df.iterrows():
        player = clean_text(item.get(player_col, ""))
        if not player or normalize_key(player) in {"player", "total", "head coach", "coach"}:
            continue
        club = clean_text(item.get(club_col, ""))
        raw_dob_age = clean_text(item.get(birth_col, "")) if birth_col else ""
        dob, age = parse_birth_age(raw_dob_age)
        rows.append(
            {
                "source": "wikipedia",
                "source_url": source_url,
                "resolved_url": resolved_url,
                "source_page_title": title,
                "competition": competition,
                "year": year,
                "team": team,
                "source_heading": team,
                "shirt_number": integer_text(item.get(number_col, "")) if number_col else "",
                "position": clean_text(item.get(position_col, "")) if position_col else "",
                "player": player,
                "date_of_birth": dob,
                "age": age,
                "caps": integer_text(item.get(caps_col, "")) if caps_col else "",
                "goals": integer_text(item.get(goals_col, "")) if goals_col else "",
                "club": club,
                "raw_date_of_birth_age": raw_dob_age,
                "raw_player": clean_text(item.get(player_col, "")),
                "raw_club": club,
            }
        )
    return rows


def extract_page(url: str) -> tuple[list[dict[str, str]], dict[str, object]]:
    resolved_url, text = fetch_html(url)
    document = lxml_html.fromstring(text)
    title = page_title(document)

    if "squad" not in title.lower():
        squad_url = discover_squad_url(document, resolved_url)
        if squad_url and squad_url != resolved_url:
            resolved_url, text = fetch_html(squad_url)
            document = lxml_html.fromstring(text)
            title = page_title(document)

    competition, year = competition_and_year(title, resolved_url)
    tables = document.xpath('//table[contains(concat(" ", normalize-space(@class), " "), " wikitable ")]')
    rows: list[dict[str, str]] = []
    squad_tables = 0
    for table in tables:
        try:
            parsed_rows = table_to_rows(table, url, resolved_url, title, competition, year)
        except ValueError:
            parsed_rows = []
        if parsed_rows:
            squad_tables += 1
            rows.extend(parsed_rows)

    report = {
        "source_url": url,
        "resolved_url": resolved_url,
        "source_page_title": title,
        "competition": competition,
        "year": year,
        "tables_seen": len(tables),
        "squad_tables": squad_tables,
        "rows": len(rows),
        "status": "ok" if rows else "no_squad_rows",
        "warning": "" if rows else "No recognizable squad tables found.",
    }
    return rows, report


def write_csv(file_name: str, rows: list[dict[str, object]], headers: list[str] | None = None) -> None:
    path = Path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if headers is None:
        headers = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    urls = load_urls(args)
    all_rows: list[dict[str, str]] = []
    report_rows: list[dict[str, object]] = []

    for url in urls:
        print(f"Fetching {url}")
        try:
            rows, report = extract_page(url)
        except Exception as error:  # noqa: BLE001 - CLI report should capture individual page failures.
            if not args.skip_errors:
                raise
            report = {
                "source_url": url,
                "resolved_url": "",
                "source_page_title": "",
                "competition": "",
                "year": "",
                "tables_seen": 0,
                "squad_tables": 0,
                "rows": 0,
                "status": "error",
                "warning": str(error),
            }
            rows = []
            print(f"  skipping: {error}", file=sys.stderr)
        all_rows.extend(rows)
        report_rows.append(report)
        print(f"  rows: {len(rows)}")
        if args.delay_ms > 0:
            time.sleep(args.delay_ms / 1000)

    write_csv(args.output, all_rows, HEADERS)
    write_csv(
        args.report,
        report_rows,
        [
            "source_url",
            "resolved_url",
            "source_page_title",
            "competition",
            "year",
            "tables_seen",
            "squad_tables",
            "rows",
            "status",
            "warning",
        ],
    )
    print(f"Wrote {len(all_rows):,} squad rows to {args.output}")
    print(f"Wrote report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
