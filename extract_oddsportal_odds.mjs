#!/usr/bin/env node

/*
Extract 1X2 historical odds from OddsPortal tournament result pages.

The training pipeline can consume the aligned CSV written by this script via:
  python train_xgboost_worldcup.py --odds-csv data/extracted/oddsportal_worldcup_closing_1x2.csv

OddsPortal serves the table through an encrypted JSON endpoint. This script
derives that endpoint from the public page HTML and the ajax-user-data script,
then decrypts the response with the same AES-CBC parameters used by the site.
*/

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";

const ORIGIN = "https://www.oddsportal.com";
const DEFAULT_URLS = [
  "https://www.oddsportal.com/football/world/world-cup-2026/results/",
  "https://www.oddsportal.com/football/world/world-cup-2022/results/",
  "https://www.oddsportal.com/football/world/world-cup-2018/results/",
  "https://www.oddsportal.com/football/world/world-cup-2014/results/",
];

const DEFAULT_OUTPUT = "data/extracted/oddsportal_worldcup_closing_1x2.csv";
const DEFAULT_RAW_OUTPUT = "data/extracted/oddsportal_worldcup_closing_1x2_raw.csv";
const DEFAULT_REPORT = "outputs/oddsportal_worldcup_match_report.csv";
const DEFAULT_RESULTS = "data/results.csv";

const DECRYPTION_KEY_TEXT = "J*8sQ!p$7aD_fR2yW@gHn*3bVp#sAdLd_k";
const DECRYPTION_SALT_TEXT = "5b9a8f2c3e6d1a4b7c8e9d0f1a2b3c4d";

const TEAM_ALIASES = new Map(
  Object.entries({
    "cote d ivoire": "ivory coast",
    "cote divoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "cabo verde": "cape verde",
    czechia: "czech republic",
    "dr congo": "congo dr",
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
    "serbia and montenegro": "serbia",
    "soviet union": "russia",
    ussr: "russia",
    turkiye: "turkey",
    usa: "united states",
    "united states": "united states",
    "viet nam": "vietnam",
    "west germany": "germany",
    zaire: "congo dr",
  }),
);

function parseArgs(argv) {
  const args = {
    urls: [],
    urlFiles: [],
    output: DEFAULT_OUTPUT,
    rawOutput: DEFAULT_RAW_OUTPUT,
    report: DEFAULT_REPORT,
    results: DEFAULT_RESULTS,
    delayMs: 500,
    maxPages: null,
    discoverArchives: false,
    skipErrors: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--url") {
      args.urls.push(argv[++i]);
    } else if (arg === "--url-file") {
      args.urlFiles.push(argv[++i]);
    } else if (arg === "--output") {
      args.output = argv[++i];
    } else if (arg === "--raw-output") {
      args.rawOutput = argv[++i];
    } else if (arg === "--match-report") {
      args.report = argv[++i];
    } else if (arg === "--results") {
      args.results = argv[++i];
    } else if (arg === "--delay-ms") {
      args.delayMs = Number(argv[++i]);
    } else if (arg === "--max-pages") {
      args.maxPages = Number(argv[++i]);
    } else if (arg === "--discover-archives" || arg === "--discover-archive-years") {
      args.discoverArchives = true;
    } else if (arg === "--skip-errors") {
      args.skipErrors = true;
    } else if (arg === "--help" || arg === "-h") {
      printHelpAndExit();
    } else if (arg.startsWith("--")) {
      throw new Error(`Unknown argument: ${arg}`);
    } else {
      args.urls.push(arg);
    }
  }

  for (const urlFile of args.urlFiles) {
    const lines = fs
      .readFileSync(urlFile, "utf8")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#"));
    args.urls.push(...lines);
  }
  if (args.urls.length === 0) {
    args.urls = DEFAULT_URLS;
  }
  args.urls = [...new Set(args.urls)];
  return args;
}

function printHelpAndExit() {
  console.log(`Usage: node extract_oddsportal_odds.mjs [options] [url ...]

Options:
  --url URL              Add an OddsPortal tournament results URL.
  --url-file PATH        Read OddsPortal URLs from a newline-delimited text file.
  --output PATH          Aligned CSV for train_xgboost_worldcup.py.
  --raw-output PATH      Raw scraped OddsPortal CSV.
  --match-report PATH    Match coverage report CSV.
  --results PATH         martj42 results.csv used for home/away alignment.
  --delay-ms N           Delay between endpoint requests. Default: 500.
  --max-pages N          Limit pages per tournament for testing.
  --discover-archives    Also scrape archive-year URLs linked from each input URL.
  --skip-errors          Keep scraping if a URL is missing or temporarily unavailable.
`);
  process.exit(0);
}

function ensureParentDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function decodeHtmlEntities(value) {
  return value
    .replace(/&quot;/g, '"')
    .replace(/&#039;/g, "'")
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

function unescapeJsonUrl(value) {
  return value.replace(/\\\//g, "/");
}

async function fetchText(url, referer = ORIGIN) {
  let lastError = null;
  for (let attempt = 1; attempt <= 5; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: {
          "user-agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
          accept: "application/json, text/plain, */*",
          "accept-language": "en-US,en;q=0.9",
          referer,
          "x-requested-with": "XMLHttpRequest",
        },
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} for ${url}`);
      }
      return await response.text();
    } catch (error) {
      lastError = error;
      if (attempt === 5) {
        break;
      }
      await sleep(750 * attempt * attempt);
    }
  }
  throw lastError;
}

function extractPageConfig(html, pageUrl) {
  const decoded = decodeHtmlEntities(html);
  const oddsRequestMatch = decoded.match(
    /"oddsRequest":\{"url":"([^"]+)","urlPartTz":([^,}]+),"urlPartQs":"([^"]+)"\}/,
  );
  if (!oddsRequestMatch) {
    throw new Error(`Could not find oddsRequest in ${pageUrl}`);
  }

  const userDataMatch = html.match(/<script[^>]+src="([^"]*ajax-user-data[^"]+)"/);
  if (!userDataMatch) {
    throw new Error(`Could not find ajax-user-data script in ${pageUrl}`);
  }

  const pageH1 = capture(decoded, /"pageH1":"([^"]+)"/);
  const tournamentId = capture(decoded, /"tournamentId":(\d+)/);
  const encodedTournamentId = capture(decoded, /"encodedTurnamentId":"([^"]+)"/);
  const activeTab = capture(decoded, /"activeTab":"([^"]+)"/);
  const context = capture(decoded, /"context":"([^"]+)"/);

  return {
    pageUrl,
    userDataUrl: absoluteUrl(userDataMatch[1], pageUrl),
    oddsRequest: {
      url: unescapeJsonUrl(oddsRequestMatch[1]),
      urlPartTz: oddsRequestMatch[2].replace(/^"|"$/g, ""),
      urlPartQs: unescapeJsonUrl(oddsRequestMatch[3]),
    },
    pageH1: pageH1 ? decodeHtmlEntities(pageH1) : "",
    tournamentId: tournamentId ? Number(tournamentId) : null,
    encodedTournamentId,
    activeTab,
    context,
  };
}

function capture(value, regex) {
  const match = value.match(regex);
  return match ? match[1] : "";
}

function absoluteUrl(value, base) {
  return new URL(value, base).href;
}

function oddsPortalCompetitionMeta(pageUrl) {
  const url = new URL(pageUrl);
  const parts = url.pathname.split("/").filter(Boolean);
  const resultsIndex = parts.lastIndexOf("results");
  if (parts[0] !== "football" || resultsIndex < 3) {
    return null;
  }
  const slug = parts[resultsIndex - 1];
  return {
    region: parts[1],
    slug,
    family: slug.replace(/-\d{4}$/g, ""),
  };
}

function sameCompetitionFamily(candidateUrl, baseUrl) {
  const candidate = oddsPortalCompetitionMeta(candidateUrl);
  const base = oddsPortalCompetitionMeta(baseUrl);
  return (
    Boolean(candidate) &&
    Boolean(base) &&
    candidate.region === base.region &&
    candidate.family === base.family
  );
}

function archiveSortValue(url) {
  const meta = oddsPortalCompetitionMeta(url);
  const year = meta?.slug.match(/-(\d{4})$/)?.[1];
  return year ? Number(year) : 9999;
}

function discoverArchiveUrls(html, pageUrl) {
  const decoded = decodeHtmlEntities(html);
  const candidates = new Set([pageUrl]);
  const patterns = [
    /"url":"(\\?\/football\/[^"]+?\/results\/)"/g,
    /"href":"(\\?\/football\/[^"]+?\/results\/)"/g,
    /href="(\/football\/[^"]+?\/results\/)"/g,
  ];

  for (const pattern of patterns) {
    for (const match of decoded.matchAll(pattern)) {
      const candidate = absoluteUrl(unescapeJsonUrl(match[1]), pageUrl);
      if (sameCompetitionFamily(candidate, pageUrl)) {
        candidates.add(candidate);
      }
    }
  }

  return [...candidates].sort((left, right) => archiveSortValue(right) - archiveSortValue(left));
}

function extractUserData(scriptText) {
  const matches = [...scriptText.matchAll(/JSON\.parse\("((?:\\.|[^"\\])*)"\)/g)];
  for (const match of matches) {
    const jsonText = JSON.parse(`"${match[1]}"`);
    const parsed = JSON.parse(jsonText);
    if (parsed.bookiehash) {
      return parsed;
    }
  }
  throw new Error("Could not find bookiehash in ajax-user-data script");
}

function decryptPayload(responseText, encrypted) {
  if (!encrypted) {
    return JSON.parse(responseText);
  }

  const decoded = Buffer.from(responseText.trim(), "base64").toString("utf8");
  const [payloadBase64, ivHex] = decoded.split(":");
  if (!payloadBase64 || !ivHex) {
    throw new Error("Unexpected encrypted OddsPortal payload format");
  }

  const key = crypto.pbkdf2Sync(
    DECRYPTION_KEY_TEXT,
    DECRYPTION_SALT_TEXT,
    1000,
    32,
    "sha256",
  );
  const decipher = crypto.createDecipheriv("aes-256-cbc", key, Buffer.from(ivHex, "hex"));
  let output = Buffer.concat([
    decipher.update(Buffer.from(payloadBase64, "base64")),
    decipher.final(),
  ]);

  if (output.length >= 2 && output[0] === 0x1f && output[1] === 0x8b) {
    output = zlib.gunzipSync(output);
  }
  return JSON.parse(output.toString("utf8"));
}

function endpointForPage(config, userData, pageNumber) {
  const base = absoluteUrl(config.oddsRequest.url, config.pageUrl);
  const pagePart = pageNumber > 1 ? `/page/${pageNumber}/` : "";
  return `${base}${userData.bookiehash}/${userData.usePremium}/${config.oddsRequest.urlPartTz}${pagePart}${config.oddsRequest.urlPartQs}${Date.now()}`;
}

function dateFromTimestamp(seconds) {
  const timestamp = Number(seconds);
  if (!Number.isFinite(timestamp)) {
    return "";
  }
  return new Date(timestamp * 1000).toISOString().slice(0, 10);
}

function numericOrBlank(value) {
  if (value === null || value === undefined || value === "") {
    return "";
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : "";
}

function hasNumericValue(value) {
  return value !== "" && value !== null && value !== undefined && Number.isFinite(Number(value));
}

function oddsByColumn(row) {
  const columns = String(row.cols || "1|X|2").split("|");
  const output = {};
  const odds = Array.isArray(row.odds) ? row.odds : [];
  for (let i = 0; i < columns.length && i < odds.length; i += 1) {
    const column = columns[i];
    const item = odds[i] || {};
    const target =
      column === "1" ? "home" : column.toUpperCase() === "X" ? "draw" : column === "2" ? "away" : "";
    if (!target) {
      continue;
    }
    output[`${target}_odds`] = numericOrBlank(item.avgOdds);
    output[`max_${target}_odds`] = numericOrBlank(item.maxOdds);
    output[`n_odds_${target}_win`] = numericOrBlank(item.cntActive);
    output[`${target}_outcome_result_id`] = numericOrBlank(item.outcomeResultId);
  }
  return output;
}

function rowToRawRecord(row, config, pageNumber) {
  const mappedOdds = oddsByColumn(row);
  if (
    !hasNumericValue(mappedOdds.home_odds) ||
    !hasNumericValue(mappedOdds.draw_odds) ||
    !hasNumericValue(mappedOdds.away_odds)
  ) {
    return null;
  }

  return {
    source: "oddsportal",
    source_competition: row["tournament-name"] || config.pageH1 || "",
    source_url: config.pageUrl,
    source_page: pageNumber,
    oddsportal_event_id: row.id ?? "",
    oddsportal_encoded_event_id: row.encodeEventId ?? "",
    date: dateFromTimestamp(row["date-start-base"] || row["date-start-timestamp"]),
    event_timestamp_utc: row["date-start-base"] || row["date-start-timestamp"] || "",
    home_team: row["home-name"] || "",
    away_team: row["away-name"] || "",
    home_score: row.homeResult ?? "",
    away_score: row.awayResult ?? "",
    result_text: stripHtml(row.result || row.postmatchResult || ""),
    tournament: row["tournament-name"] || "",
    tournament_stage: String(row["tournament-stage-name"] || "").trim(),
    event_stage: row["event-stage-name"] || "",
    venue: row.venue || "",
    venue_town: row.venueTown || "",
    venue_country: row.venueCountry || "",
    home_odds: mappedOdds.home_odds,
    draw_odds: mappedOdds.draw_odds,
    away_odds: mappedOdds.away_odds,
    max_home_odds: mappedOdds.max_home_odds,
    max_draw_odds: mappedOdds.max_draw_odds,
    max_away_odds: mappedOdds.max_away_odds,
    n_odds_home_win: mappedOdds.n_odds_home_win,
    n_odds_draw: mappedOdds.n_odds_draw_win,
    n_odds_away_win: mappedOdds.n_odds_away_win,
    odds_result_home: mappedOdds.home_outcome_result_id,
    odds_result_draw: mappedOdds.draw_outcome_result_id,
    odds_result_away: mappedOdds.away_outcome_result_id,
    detail_url: row.url ? absoluteUrl(row.url, ORIGIN) : "",
  };
}

function stripHtml(value) {
  return decodeHtmlEntities(String(value).replace(/<[^>]+>/g, "").replace(/&nbsp;/g, " ")).trim();
}

async function scrapeCompetition(pageUrl, args) {
  console.log(`Fetching ${pageUrl}`);
  const html = await fetchText(pageUrl);
  const config = extractPageConfig(html, pageUrl);
  const userDataScript = await fetchText(config.userDataUrl, pageUrl);
  let userData = extractUserData(userDataScript);
  const encrypted = userData.encriptedResponse !== false;
  const rows = [];

  let pageCount = 1;
  for (let pageNumber = 1; pageNumber <= pageCount; pageNumber += 1) {
    if (args.maxPages && pageNumber > args.maxPages) {
      break;
    }
    let data = {};
    let pageRows = [];
    for (let attempt = 1; attempt <= 8; attempt += 1) {
      const endpoint = endpointForPage(config, userData, pageNumber);
      const text = await fetchText(endpoint, pageUrl);
      const payload = decryptPayload(text, encrypted);
      data = payload.d || {};
      pageRows = Array.isArray(data.rows) ? data.rows : [];
      if (pageRows.length > 0 || pageNumber === pageCount || attempt === 8) {
        break;
      }
      if (attempt % 3 === 0) {
        const refreshedUserDataScript = await fetchText(config.userDataUrl, pageUrl);
        userData = extractUserData(refreshedUserDataScript);
      }
      await sleep(1500 * attempt);
    }
    if (pageNumber === 1) {
      pageCount = Number(data.pagination?.pageCount) || Math.ceil(Number(data.total || 0) / Number(data.onePage || 50)) || 1;
    }
    for (const row of pageRows) {
      const record = rowToRawRecord(row, config, pageNumber);
      if (record) {
        rows.push(record);
      }
    }
    console.log(`  page ${pageNumber}/${pageCount}: ${pageRows.length} rows`);
    if (pageNumber < pageCount && args.delayMs > 0) {
      await sleep(args.delayMs);
    }
  }
  return rows;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (char === '"' && next === '"') {
        value += '"';
        i += 1;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        value += char;
      }
    } else if (char === '"') {
      inQuotes = true;
    } else if (char === ",") {
      row.push(value);
      value = "";
    } else if (char === "\n") {
      row.push(value);
      rows.push(row);
      row = [];
      value = "";
    } else if (char !== "\r") {
      value += char;
    }
  }
  if (value.length || row.length) {
    row.push(value);
    rows.push(row);
  }
  if (rows.length === 0) {
    return [];
  }
  const headers = rows.shift();
  return rows
    .filter((items) => items.length > 1)
    .map((items) => Object.fromEntries(headers.map((header, index) => [header, items[index] ?? ""])));
}

function normalizeName(value) {
  const normalized = String(value || "")
    .trim()
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[.,\-_()&]/g, " ")
    .replace(/'/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return TEAM_ALIASES.get(normalized) || normalized;
}

function matchKey(date, homeTeam, awayTeam) {
  return `${date}|${normalizeName(homeTeam)}|${normalizeName(awayTeam)}`;
}

function loadResultsIndex(resultsPath) {
  if (!fs.existsSync(resultsPath)) {
    return null;
  }
  const text = fs.readFileSync(resultsPath, "utf8");
  const rows = parseCsv(text);
  const exact = new Map();
  const reversed = new Map();
  for (const row of rows) {
    if (!row.date || !row.home_team || !row.away_team) {
      continue;
    }
    exact.set(matchKey(row.date, row.home_team, row.away_team), row);
    reversed.set(matchKey(row.date, row.away_team, row.home_team), row);
  }
  return { exact, reversed };
}

function alignToResults(rawRows, resultsIndex) {
  if (!resultsIndex) {
    return { aligned: rawRows, reportRows: [], unmatched: rawRows, duplicates: 0 };
  }
  const aligned = [];
  const unmatched = [];
  const seen = new Set();
  let exactMatches = 0;
  let reversedMatches = 0;
  let duplicates = 0;

  for (const row of rawRows) {
    const key = matchKey(row.date, row.home_team, row.away_team);
    const reverseKey = matchKey(row.date, row.away_team, row.home_team);
    const exactResult = resultsIndex.exact.get(key);
    const reverseResult = resultsIndex.exact.get(reverseKey);

    if (exactResult) {
      const alignedKey = matchKey(exactResult.date, exactResult.home_team, exactResult.away_team);
      if (seen.has(alignedKey)) {
        duplicates += 1;
        continue;
      }
      seen.add(alignedKey);
      exactMatches += 1;
      aligned.push({
        ...row,
        date: exactResult.date,
        home_team: exactResult.home_team,
        away_team: exactResult.away_team,
        martj42_tournament: exactResult.tournament || "",
        martj42_home_score: exactResult.home_score || "",
        martj42_away_score: exactResult.away_score || "",
        source_match_orientation: "same",
      });
    } else if (reverseResult) {
      const alignedKey = matchKey(reverseResult.date, reverseResult.home_team, reverseResult.away_team);
      if (seen.has(alignedKey)) {
        duplicates += 1;
        continue;
      }
      seen.add(alignedKey);
      reversedMatches += 1;
      aligned.push({
        ...row,
        date: reverseResult.date,
        home_team: reverseResult.home_team,
        away_team: reverseResult.away_team,
        home_score: row.away_score,
        away_score: row.home_score,
        home_odds: row.away_odds,
        away_odds: row.home_odds,
        max_home_odds: row.max_away_odds,
        max_away_odds: row.max_home_odds,
        n_odds_home_win: row.n_odds_away_win,
        n_odds_away_win: row.n_odds_home_win,
        odds_result_home: row.odds_result_away,
        odds_result_away: row.odds_result_home,
        martj42_tournament: reverseResult.tournament || "",
        martj42_home_score: reverseResult.home_score || "",
        martj42_away_score: reverseResult.away_score || "",
        source_match_orientation: "reversed",
      });
    } else {
      unmatched.push(row);
    }
  }

  const reportRows = buildReportRows(rawRows, aligned, unmatched, exactMatches, reversedMatches, duplicates);
  return { aligned, reportRows, unmatched, duplicates };
}

function buildReportRows(rawRows, alignedRows, unmatchedRows, exactMatches, reversedMatches, duplicates) {
  const byCompetition = new Map();
  for (const row of rawRows) {
    const key = `${row.source_competition}|${row.source_url}`;
    if (!byCompetition.has(key)) {
      byCompetition.set(key, {
        source_competition: row.source_competition,
        source_url: row.source_url,
        raw_rows: 0,
        aligned_rows: 0,
        unmatched_rows: 0,
      });
    }
    byCompetition.get(key).raw_rows += 1;
  }
  for (const row of alignedRows) {
    const key = `${row.source_competition}|${row.source_url}`;
    if (byCompetition.has(key)) {
      byCompetition.get(key).aligned_rows += 1;
    }
  }
  for (const row of unmatchedRows) {
    const key = `${row.source_competition}|${row.source_url}`;
    if (byCompetition.has(key)) {
      byCompetition.get(key).unmatched_rows += 1;
    }
  }

  const rows = [...byCompetition.values()];
  rows.push({
    source_competition: "__TOTAL__",
    source_url: "",
    raw_rows: rawRows.length,
    aligned_rows: alignedRows.length,
    unmatched_rows: unmatchedRows.length,
    exact_matches: exactMatches,
    reversed_matches: reversedMatches,
    duplicate_aligned_rows: duplicates,
  });
  return rows;
}

function csvEscape(value) {
  const text = value === null || value === undefined ? "" : String(value);
  if (/[",\r\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function writeCsv(filePath, rows, preferredColumns = []) {
  ensureParentDir(filePath);
  const columns = [
    ...preferredColumns,
    ...[...new Set(rows.flatMap((row) => Object.keys(row)))].filter(
      (column) => !preferredColumns.includes(column),
    ),
  ];
  const lines = [columns.join(",")];
  for (const row of rows) {
    lines.push(columns.map((column) => csvEscape(row[column])).join(","));
  }
  fs.writeFileSync(filePath, `${lines.join("\n")}\n`, "utf8");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.discoverArchives) {
    const expandedUrls = new Set();
    for (const url of args.urls) {
      console.log(`Discovering archive years for ${url}`);
      const html = await fetchText(url);
      const discovered = discoverArchiveUrls(html, url);
      console.log(`  found ${discovered.length} URL(s) in this competition family`);
      for (const discoveredUrl of discovered) {
        expandedUrls.add(discoveredUrl);
      }
      if (args.delayMs > 0) {
        await sleep(args.delayMs);
      }
    }
    args.urls = [...expandedUrls].sort((left, right) => {
      const leftMeta = oddsPortalCompetitionMeta(left);
      const rightMeta = oddsPortalCompetitionMeta(right);
      const familyCompare = `${leftMeta?.region || ""}/${leftMeta?.family || ""}`.localeCompare(
        `${rightMeta?.region || ""}/${rightMeta?.family || ""}`,
      );
      return familyCompare || archiveSortValue(right) - archiveSortValue(left);
    });
    console.log(`Expanded to ${args.urls.length} unique OddsPortal URL(s).`);
  }

  const rawRows = [];
  const errorRows = [];
  for (const url of args.urls) {
    try {
      const rows = await scrapeCompetition(url, args);
      rawRows.push(...rows);
    } catch (error) {
      if (!args.skipErrors) {
        throw error;
      }
      console.warn(`Skipping ${url}: ${error.message}`);
      errorRows.push({
        source_competition: "__ERROR__",
        source_url: url,
        raw_rows: 0,
        aligned_rows: 0,
        unmatched_rows: 0,
        exact_matches: 0,
        reversed_matches: 0,
        duplicate_aligned_rows: 0,
        status: "error",
        error: error.message,
      });
    }
  }

  const rawColumns = [
    "source",
    "source_competition",
    "source_url",
    "source_page",
    "oddsportal_event_id",
    "oddsportal_encoded_event_id",
    "date",
    "home_team",
    "away_team",
    "home_odds",
    "draw_odds",
    "away_odds",
    "max_home_odds",
    "max_draw_odds",
    "max_away_odds",
    "n_odds_home_win",
    "n_odds_draw",
    "n_odds_away_win",
  ];
  writeCsv(args.rawOutput, rawRows, rawColumns);

  const resultsIndex = loadResultsIndex(args.results);
  const { aligned, reportRows, unmatched } = alignToResults(rawRows, resultsIndex);
  writeCsv(args.output, aligned, [
    "date",
    "home_team",
    "away_team",
    "home_odds",
    "draw_odds",
    "away_odds",
    "max_home_odds",
    "max_draw_odds",
    "max_away_odds",
    "n_odds_home_win",
    "n_odds_draw",
    "n_odds_away_win",
    "source_match_orientation",
    "martj42_tournament",
    "source_competition",
    "source_url",
    "oddsportal_event_id",
  ]);
  const allReportRows = [...reportRows, ...errorRows];
  if (allReportRows.length) {
    writeCsv(args.report, allReportRows, [
      "source_competition",
      "source_url",
      "raw_rows",
      "aligned_rows",
      "unmatched_rows",
      "exact_matches",
      "reversed_matches",
      "duplicate_aligned_rows",
      "status",
      "error",
    ]);
    writeCsv(args.report.replace(/\.csv$/i, "_unmatched_sample.csv"), unmatched.slice(0, 200), rawColumns);
  }

  console.log(`Wrote ${rawRows.length.toLocaleString("en-US")} raw rows to ${args.rawOutput}`);
  console.log(`Wrote ${aligned.length.toLocaleString("en-US")} aligned rows to ${args.output}`);
  if (allReportRows.length) {
    console.log(`Wrote match report to ${args.report}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
