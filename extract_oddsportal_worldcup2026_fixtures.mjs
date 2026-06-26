#!/usr/bin/env node

/*
Create a World Cup 2026 fixture schedule with current OddsPortal 1X2 odds.

OddsPortal exposes the currently listed World Cup fixtures through the
"next matches" page. ESPN's public scoreboard is used as a fixture hydrator:
when knockout slots such as 1A, 2B, W73, etc. are resolved to real teams,
those team names are merged into the local 104-match skeleton before odds are
matched.
*/

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";

const DEFAULT_URL = "https://www.oddsportal.com/football/world/world-cup/";
const DEFAULT_FIXTURES = "data/extracted/worldcup2026_future_fixtures.csv";
const DEFAULT_OUTPUT = "data/extracted/oddsportal_worldcup2026_fixture_odds_schedule.csv";
const DEFAULT_RAW_OUTPUT = "data/extracted/oddsportal_worldcup2026_fixture_odds_raw.csv";
const DEFAULT_REPORT = "outputs/oddsportal_worldcup2026_fixture_odds_report.csv";
const DEFAULT_ESPN_SCOREBOARD_URL =
  "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?limit=200&dates=20260611-20260720";
const DEFAULT_ESPN_FIXTURES_OUTPUT = "data/extracted/espn_worldcup2026_fixtures.csv";

const DECRYPTION_KEY_TEXT = "J*8sQ!p$7aD_fR2yW@gHn*3bVp#sAdLd_k";
const DECRYPTION_SALT_TEXT = "5b9a8f2c3e6d1a4b7c8e9d0f1a2b3c4d";

const TEAM_ALIASES = new Map(
  Object.entries({
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Czech Rep": "Czech Republic",
    Curacao: "Curaçao",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "D.R. Congo": "DR Congo",
    Turkiye: "Turkey",
    "TÃ¼rkiye": "Turkey",
    USA: "United States",
    "Winner UEFA Playoff A": "Bosnia and Herzegovina",
    "Winner UEFA Playoff B": "Sweden",
    "Winner UEFA Playoff C": "Turkey",
    "Winner UEFA Playoff D": "Czech Republic",
  }),
);

function parseArgs(argv) {
  const args = {
    url: DEFAULT_URL,
    fixtures: DEFAULT_FIXTURES,
    output: DEFAULT_OUTPUT,
    rawOutput: DEFAULT_RAW_OUTPUT,
    report: DEFAULT_REPORT,
    espnScoreboardUrl: DEFAULT_ESPN_SCOREBOARD_URL,
    espnFixturesOutput: DEFAULT_ESPN_FIXTURES_OUTPUT,
    skipEspnFixtures: false,
    skipOddsScrape: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--url") {
      args.url = argv[++i];
    } else if (arg === "--fixtures") {
      args.fixtures = argv[++i];
    } else if (arg === "--output") {
      args.output = argv[++i];
    } else if (arg === "--raw-output") {
      args.rawOutput = argv[++i];
    } else if (arg === "--report") {
      args.report = argv[++i];
    } else if (arg === "--espn-scoreboard-url") {
      args.espnScoreboardUrl = argv[++i];
    } else if (arg === "--espn-fixtures-output") {
      args.espnFixturesOutput = argv[++i];
    } else if (arg === "--skip-espn-fixtures") {
      args.skipEspnFixtures = true;
    } else if (arg === "--skip-odds-scrape") {
      args.skipOddsScrape = true;
    } else if (arg === "--help" || arg === "-h") {
      printHelpAndExit();
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

function printHelpAndExit() {
  console.log(`Usage: node extract_oddsportal_worldcup2026_fixtures.mjs [options]

Options:
  --url URL          OddsPortal World Cup next-matches URL.
  --fixtures PATH   Local full 104-match fixture skeleton CSV.
  --output PATH     Output schedule with odds where available.
  --raw-output PATH Raw OddsPortal fixture odds.
  --report PATH     Coverage report CSV.
  --espn-scoreboard-url URL
                    ESPN scoreboard URL used to resolve real knockout teams.
  --espn-fixtures-output PATH
                    Output parsed ESPN fixture rows for audit/debugging.
  --skip-espn-fixtures
                    Do not use ESPN to hydrate fixture placeholders.
  --skip-odds-scrape
                    Debug mode: build the schedule without live OddsPortal odds.
`);
  process.exit(0);
}

function ensureParentDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

function decodeHtmlEntities(value) {
  return String(value)
    .replace(/&quot;/g, '"')
    .replace(/&#039;/g, "'")
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

async function fetchText(url, referer = "https://www.oddsportal.com") {
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
      await new Promise((resolve) => setTimeout(resolve, 750 * attempt * attempt));
    }
  }
  throw lastError;
}

async function fetchJson(url) {
  return JSON.parse(await fetchText(url, "https://www.espn.com"));
}

function extractNextMatchesData(html) {
  const match = html.match(/<next-matches\b[^>]*:comp-data="([^"]+)"/);
  if (!match) {
    throw new Error("Could not find OddsPortal next-matches component data.");
  }
  return JSON.parse(decodeHtmlEntities(match[1]));
}

function extractUserData(scriptText) {
  const matches = [...scriptText.matchAll(/JSON\.parse\("((?:\\.|[^"\\])*)"\)/g)];
  for (const match of matches) {
    const jsonText = JSON.parse(`"${match[1]}"`);
    const parsed = JSON.parse(jsonText);
    if (parsed.bookiehash || parsed.encriptedResponse !== undefined) {
      return parsed;
    }
  }
  throw new Error("Could not find OddsPortal user data.");
}

function extractUserDataUrl(html, pageUrl) {
  const match = html.match(/<script[^>]+src="([^"]*ajax-user-data[^"]+)"/);
  if (!match) {
    throw new Error("Could not find OddsPortal ajax-user-data script.");
  }
  return new URL(match[1], pageUrl).href;
}

function decryptPayload(responseText, encrypted) {
  if (!encrypted) {
    return JSON.parse(responseText);
  }
  const decoded = Buffer.from(responseText.trim(), "base64").toString("utf8");
  const [payloadBase64, ivHex] = decoded.split(":");
  if (!payloadBase64 || !ivHex) {
    throw new Error("Unexpected encrypted OddsPortal payload format.");
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

function numericOrBlank(value) {
  if (value === null || value === undefined || value === "") {
    return "";
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : "";
}

function dateFromTimestamp(seconds) {
  const timestamp = Number(seconds);
  if (!Number.isFinite(timestamp)) {
    return "";
  }
  return new Date(timestamp * 1000).toISOString().slice(0, 10);
}

function isoFromTimestamp(seconds) {
  const timestamp = Number(seconds);
  if (!Number.isFinite(timestamp)) {
    return "";
  }
  return new Date(timestamp * 1000).toISOString();
}

function modelTeamName(value) {
  const text = String(value || "").trim();
  return TEAM_ALIASES.get(text) || text;
}

function normalizeTeamKey(value) {
  const text = modelTeamName(value)
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
  const aliases = new Map(
    Object.entries({
      "bosnia hz": "bosnia and herzegovina",
      "cote d ivoire": "ivory coast",
      "czech rep": "czech republic",
      "czechia": "czech republic",
      "congo dr": "congo dr",
      "d r congo": "congo dr",
      "dr congo": "congo dr",
      "korea rep": "south korea",
      "korea republic": "south korea",
      "turkiye": "turkey",
      "usa": "united states",
    }),
  );
  return aliases.get(text) || text;
}

function isPlaceholderTeam(value) {
  const text = String(value || "").trim().toLowerCase();
  return (
    !text ||
    text === "nan" ||
    text.startsWith("winner ") ||
    text.startsWith("group ") ||
    /^[123][a-l]+$/i.test(text) ||
    /^w\d+/.test(text) ||
    /^ru\d+/.test(text)
  );
}

function slotFromEspnLabel(value) {
  const text = String(value || "").trim();
  let match = text.match(/^([123])([A-L]+)$/i);
  if (match) {
    return `${match[1]}${match[2].toUpperCase()}`;
  }
  match = text.match(/^Group\s+([A-L])\s+Winner$/i);
  if (match) {
    return `1${match[1].toUpperCase()}`;
  }
  match = text.match(/^Group\s+([A-L])\s+(?:2nd|Second)\s+Place$/i);
  if (match) {
    return `2${match[1].toUpperCase()}`;
  }
  match = text.match(/^Third\s+Place\s+Group\s+([A-L](?:\s*\/\s*[A-L])+)$/i);
  if (match) {
    return `3${match[1].replace(/[^A-L]/gi, "").toUpperCase()}`;
  }
  match = text.match(/^Round\s+of\s+32\s+(\d+)\s+Winner$/i);
  if (match) {
    return `W${72 + Number(match[1])}`;
  }
  match = text.match(/^Round\s+of\s+16\s+(\d+)\s+Winner$/i);
  if (match) {
    return `W${88 + Number(match[1])}`;
  }
  match = text.match(/^Quarterfinal\s+(\d+)\s+Winner$/i);
  if (match) {
    return `W${96 + Number(match[1])}`;
  }
  match = text.match(/^Semifinal\s+(\d+)\s+Winner$/i);
  if (match) {
    return `W${100 + Number(match[1])}`;
  }
  match = text.match(/^Semifinal\s+(\d+)\s+Loser$/i);
  if (match) {
    return `RU${100 + Number(match[1])}`;
  }
  return "";
}

function espnTeamInfo(competitor) {
  const team = competitor?.team || {};
  const labels = [
    team.displayName,
    team.shortDisplayName,
    team.name,
    team.abbreviation,
    team.location,
  ].filter(Boolean);
  const slot = labels.map(slotFromEspnLabel).find(Boolean) || "";
  const label = team.displayName || team.shortDisplayName || team.name || team.location || "";
  const placeholder = Boolean(slot) || (team.isActive === false && /group|winner|place|round|quarter|semi/i.test(label));
  return {
    team: placeholder ? slot || label : modelTeamName(label),
    slot,
    is_placeholder: placeholder,
    display_name: label,
    abbreviation: team.abbreviation || "",
  };
}

function stageFromEspnSeason(event) {
  const slug = String(event?.season?.slug || "").toLowerCase();
  const map = new Map(
    Object.entries({
      "group-stage": "Group Stage",
      "round-of-32": "Round of 32",
      "round-of-16": "Round of 16",
      quarterfinals: "Quarterfinals",
      semifinals: "Semifinals",
      "third-place": "Third Place Playoff",
      "3rd-place-match": "Third Place Playoff",
      final: "Final",
    }),
  );
  return map.get(slug) || event?.season?.slug || "";
}

function parseEspnFixtureRows(payload) {
  const events = Array.isArray(payload?.events) ? payload.events : [];
  return events
    .map((event, index) => {
      const competition = Array.isArray(event.competitions) ? event.competitions[0] : {};
      const competitors = Array.isArray(competition?.competitors) ? competition.competitors : [];
      const home = competitors.find((item) => item.homeAway === "home") || competitors[0] || {};
      const away = competitors.find((item) => item.homeAway === "away") || competitors[1] || {};
      const homeInfo = espnTeamInfo(home);
      const awayInfo = espnTeamInfo(away);
      return {
        source: "espn",
        espn_event_id: event.id || competition?.id || "",
        match_number: String(index + 1),
        date: String(event.date || competition?.date || "").slice(0, 10),
        kickoff_utc: event.date || competition?.date || "",
        stage: stageFromEspnSeason(event),
        home_team: homeInfo.team,
        away_team: awayInfo.team,
        home_slot: homeInfo.slot,
        away_slot: awayInfo.slot,
        home_is_placeholder: boolString(homeInfo.is_placeholder),
        away_is_placeholder: boolString(awayInfo.is_placeholder),
        espn_home_team: homeInfo.display_name,
        espn_away_team: awayInfo.display_name,
        espn_home_abbreviation: homeInfo.abbreviation,
        espn_away_abbreviation: awayInfo.abbreviation,
        espn_name: event.name || "",
      };
    })
    .filter((row) => row.espn_event_id && row.home_team && row.away_team);
}

async function fetchEspnFixtures(scoreboardUrl) {
  return parseEspnFixtureRows(await fetchJson(scoreboardUrl));
}

function dateDiffDays(left, right) {
  const leftTime = Date.parse(`${left}T00:00:00Z`);
  const rightTime = Date.parse(`${right}T00:00:00Z`);
  if (!Number.isFinite(leftTime) || !Number.isFinite(rightTime)) {
    return Number.POSITIVE_INFINITY;
  }
  return Math.abs(Math.round((leftTime - rightTime) / 86_400_000));
}

function oddsFitsFixture(fixture, odds) {
  if (!odds || dateDiffDays(fixture.date, odds.oddsportal_date_utc) > 1) {
    return false;
  }
  const homePlaceholder = isPlaceholderTeam(fixture.home_team);
  const awayPlaceholder = isPlaceholderTeam(fixture.away_team);
  if (!homePlaceholder && normalizeTeamKey(fixture.home_team) !== normalizeTeamKey(odds.home_team)) {
    return false;
  }
  if (!awayPlaceholder && normalizeTeamKey(fixture.away_team) !== normalizeTeamKey(odds.away_team)) {
    return false;
  }
  return !homePlaceholder || !awayPlaceholder;
}

function scoreOddsFixtureMatch(fixture, odds) {
  const diff = dateDiffDays(fixture.date, odds.oddsportal_date_utc);
  const homeExact =
    !isPlaceholderTeam(fixture.home_team) &&
    normalizeTeamKey(fixture.home_team) === normalizeTeamKey(odds.home_team);
  const awayExact =
    !isPlaceholderTeam(fixture.away_team) &&
    normalizeTeamKey(fixture.away_team) === normalizeTeamKey(odds.away_team);
  return (homeExact ? 3 : 0) + (awayExact ? 3 : 0) + (1 - diff) * 0.5;
}

function findOddsForFixture(fixture, oddsRows, usedOddsIndexes) {
  let best = null;
  for (let index = 0; index < oddsRows.length; index += 1) {
    if (usedOddsIndexes.has(index) || !oddsFitsFixture(fixture, oddsRows[index])) {
      continue;
    }
    const score = scoreOddsFixtureMatch(fixture, oddsRows[index]);
    if (!best || score > best.score) {
      best = { index, odds: oddsRows[index], score };
    }
  }
  if (!best) {
    return null;
  }
  usedOddsIndexes.add(best.index);
  return best.odds;
}

function oddsByColumn(row, oddsData) {
  const item = oddsData?.[row.encodeEventId] || {};
  const odds = Array.isArray(item.odds) ? item.odds : [];
  const columns = String(row.cols || "1|X|2").split("|");
  const output = {};
  for (let index = 0; index < columns.length && index < odds.length; index += 1) {
    const column = columns[index];
    const target =
      column === "1" ? "home" : column.toUpperCase() === "X" ? "draw" : column === "2" ? "away" : "";
    if (!target) {
      continue;
    }
    const odd = odds[index] || {};
    output[`${target}_odds`] = numericOrBlank(odd.avgOdds);
    output[`max_${target}_odds`] = numericOrBlank(odd.maxOdds);
    output[`n_odds_${target}_win`] = numericOrBlank(odd.cntActive);
  }
  return output;
}

function oddsAvailable(record) {
  return Boolean(record.home_odds && record.draw_odds && record.away_odds);
}

function rawRecord(row, oddsData, pageUrl, scrapedAtUtc) {
  const odds = oddsByColumn(row, oddsData);
  return {
    source: "oddsportal",
    source_url: pageUrl,
    scraped_at_utc: scrapedAtUtc,
    oddsportal_event_id: row.id ?? "",
    oddsportal_encoded_event_id: row.encodeEventId ?? "",
    oddsportal_date_utc: dateFromTimestamp(row["date-start-base"] || row["date-start-timestamp"]),
    oddsportal_kickoff_utc: isoFromTimestamp(row["date-start-base"] || row["date-start-timestamp"]),
    oddsportal_home_team: row["home-name"] || "",
    oddsportal_away_team: row["away-name"] || "",
    home_team: modelTeamName(row["home-name"]),
    away_team: modelTeamName(row["away-name"]),
    tournament: row["tournament-name"] || "World Cup 2026",
    event_stage: row["event-stage-name"] || "",
    venue: row.venue || "",
    venue_town: row.venueTown || "",
    venue_country: row.venueCountry || "",
    detail_url: row.url ? new URL(row.url, "https://www.oddsportal.com").href : "",
    ...odds,
  };
}

async function scrapeOddsPortal(pageUrl) {
  const html = await fetchText(pageUrl);
  const compData = extractNextMatchesData(html);
  const userDataUrl = extractUserDataUrl(html, pageUrl);
  const userData = extractUserData(await fetchText(userDataUrl, pageUrl));
  const endpointUrl = new URL(compData.oddsRequest.url, pageUrl).href;
  const payload = decryptPayload(await fetchText(endpointUrl, pageUrl), userData.encriptedResponse !== false);
  const oddsData = payload.d?.oddsData || {};
  const scrapedAtUtc = new Date().toISOString();
  const rows = Array.isArray(compData.d?.rows) ? compData.d.rows : [];
  return rows.map((row) => rawRecord(row, oddsData, pageUrl, scrapedAtUtc));
}

function fallbackOddsRowsFromSchedule(schedulePath) {
  if (!schedulePath || !fs.existsSync(schedulePath)) {
    return [];
  }
  return parseCsv(fs.readFileSync(schedulePath, "utf8"))
    .filter((row) => oddsAvailable(row))
    .map((row) => {
      const oddsportalDate =
        row.oddsportal_date_utc || String(row.oddsportal_kickoff_utc || "").slice(0, 10) || row.date;
      return {
        source: "existing_odds_schedule",
        source_url: row.oddsportal_source_url || row.oddsportal_detail_url || "",
        scraped_at_utc: row.scraped_at_utc || "",
        oddsportal_event_id: row.oddsportal_event_id || "",
        oddsportal_encoded_event_id: row.oddsportal_encoded_event_id || "",
        oddsportal_date_utc: oddsportalDate,
        oddsportal_kickoff_utc: row.oddsportal_kickoff_utc || "",
        oddsportal_home_team: row.oddsportal_home_team || row.home_team,
        oddsportal_away_team: row.oddsportal_away_team || row.away_team,
        home_team: modelTeamName(row.oddsportal_home_team || row.home_team),
        away_team: modelTeamName(row.oddsportal_away_team || row.away_team),
        tournament: row.tournament || "World Cup 2026",
        event_stage: row.stage || "",
        venue: row.venue || "",
        detail_url: row.oddsportal_detail_url || "",
        home_odds: row.home_odds,
        draw_odds: row.draw_odds,
        away_odds: row.away_odds,
        max_home_odds: row.max_home_odds,
        max_draw_odds: row.max_draw_odds,
        max_away_odds: row.max_away_odds,
        n_odds_home_win: row.n_odds_home_win,
        n_odds_draw_win: row.n_odds_draw_win || row.n_odds_draw,
        n_odds_draw: row.n_odds_draw || row.n_odds_draw_win,
        n_odds_away_win: row.n_odds_away_win,
      };
    });
}

function oddsRowsFromRaw(rawPath) {
  if (!rawPath || !fs.existsSync(rawPath)) {
    return [];
  }
  return parseCsv(fs.readFileSync(rawPath, "utf8"))
    .filter((row) => oddsAvailable(row))
    .map((row) => ({
      source: row.source || "existing_odds_raw",
      source_url: row.source_url || "",
      scraped_at_utc: row.scraped_at_utc || "",
      oddsportal_event_id: row.oddsportal_event_id || "",
      oddsportal_encoded_event_id: row.oddsportal_encoded_event_id || "",
      oddsportal_date_utc: row.oddsportal_date_utc || String(row.oddsportal_kickoff_utc || "").slice(0, 10),
      oddsportal_kickoff_utc: row.oddsportal_kickoff_utc || "",
      oddsportal_home_team: row.oddsportal_home_team || row.home_team,
      oddsportal_away_team: row.oddsportal_away_team || row.away_team,
      home_team: modelTeamName(row.home_team || row.oddsportal_home_team),
      away_team: modelTeamName(row.away_team || row.oddsportal_away_team),
      tournament: row.tournament || "World Cup 2026",
      event_stage: row.event_stage || "",
      venue: row.venue || "",
      venue_town: row.venue_town || "",
      venue_country: row.venue_country || "",
      detail_url: row.detail_url || "",
      home_odds: row.home_odds,
      draw_odds: row.draw_odds,
      away_odds: row.away_odds,
      max_home_odds: row.max_home_odds,
      max_draw_odds: row.max_draw_odds,
      max_away_odds: row.max_away_odds,
      n_odds_home_win: row.n_odds_home_win,
      n_odds_draw_win: row.n_odds_draw_win || row.n_odds_draw,
      n_odds_draw: row.n_odds_draw || row.n_odds_draw_win,
      n_odds_away_win: row.n_odds_away_win,
    }));
}

function oddsRowKey(row) {
  const eventId = String(row.oddsportal_event_id || "").trim();
  if (eventId) {
    return `event:${eventId}`;
  }
  const encoded = String(row.oddsportal_encoded_event_id || "").trim();
  if (encoded) {
    return `encoded:${encoded}`;
  }
  return [
    "teams",
    row.oddsportal_date_utc || "",
    normalizeTeamKey(row.home_team || row.oddsportal_home_team),
    normalizeTeamKey(row.away_team || row.oddsportal_away_team),
  ].join("|");
}

function mergeOddsRows(existingRows, freshRows) {
  const merged = new Map();
  for (const row of existingRows) {
    if (oddsAvailable(row)) {
      merged.set(oddsRowKey(row), row);
    }
  }
  for (const row of freshRows) {
    if (oddsAvailable(row)) {
      merged.set(oddsRowKey(row), row);
    }
  }
  return [...merged.values()].sort((left, right) => {
    const dateCompare = String(left.oddsportal_date_utc || "").localeCompare(String(right.oddsportal_date_utc || ""));
    if (dateCompare !== 0) {
      return dateCompare;
    }
    return normalizeTeamKey(left.home_team).localeCompare(normalizeTeamKey(right.home_team));
  });
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let inQuotes = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];
    if (inQuotes) {
      if (char === '"' && next === '"') {
        value += '"';
        index += 1;
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
  if (!rows.length) {
    return [];
  }
  const headers = rows.shift();
  return rows
    .filter((items) => items.some((item) => item !== ""))
    .map((items) => Object.fromEntries(headers.map((header, index) => [header, items[index] ?? ""])));
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
    ...[
      ...new Set(rows.flatMap((row) => Object.keys(row))),
    ].filter((column) => !preferredColumns.includes(column)),
  ];
  const body = [
    columns.join(","),
    ...rows.map((row) => columns.map((column) => csvEscape(row[column])).join(",")),
  ].join("\n");
  fs.writeFileSync(filePath, `${body}\n`, "utf8");
}

function isTrue(value) {
  return String(value || "").toLowerCase() === "true" || String(value || "") === "1";
}

function boolString(value) {
  return value ? "True" : "False";
}

function splitPlaceholder(groupValue) {
  const text = String(groupValue || "").trim();
  const parts = text.split(/\s+vs\s+/i);
  return parts.length === 2 ? parts : ["", ""];
}

function normaliseFixtureSlot(slot, matchNumber, side) {
  const text = String(slot || "").trim();
  if (String(matchNumber) === "100" && side === "away" && text === "W100") {
    return "W96";
  }
  return text;
}

function fixtureSlotPair(row) {
  const [home, away] = splitPlaceholder(row.group);
  const homeSlot = normaliseFixtureSlot(home || row.home_team, row.match_number, "home");
  const awaySlot = normaliseFixtureSlot(away || row.away_team, row.match_number, "away");
  return `${homeSlot}|${awaySlot}`;
}

function cleanLocalTeam(value, groupValue, side) {
  const text = String(value || "").trim();
  if (text && text.toLowerCase() !== "nan") {
    return modelTeamName(text);
  }
  const [home, away] = splitPlaceholder(groupValue);
  return side === "home" ? home : away;
}

function hydrateFixturesWithEspn(fixtures, espnFixtures) {
  if (!espnFixtures.length) {
    return fixtures;
  }
  const espnByMatchNumber = new Map(espnFixtures.map((row) => [String(row.match_number), row]));
  const espnBySlotPair = new Map(
    espnFixtures
      .filter((row) => row.home_slot && row.away_slot)
      .map((row) => [`${row.home_slot}|${row.away_slot}`, row]),
  );
  return fixtures.map((fixture) => {
    const espn = espnByMatchNumber.get(String(fixture.match_number)) || espnBySlotPair.get(fixtureSlotPair(fixture));
    if (!espn) {
      return fixture;
    }
    const hydrated = { ...fixture };
    hydrated.espn_event_id = espn.espn_event_id;
    hydrated.espn_home_team = espn.espn_home_team;
    hydrated.espn_away_team = espn.espn_away_team;
    hydrated.espn_home_slot = espn.home_slot;
    hydrated.espn_away_slot = espn.away_slot;
    hydrated.espn_fixture_stage = espn.stage;
    hydrated.espn_kickoff_utc = espn.kickoff_utc;
    hydrated.espn_fixture_name = espn.espn_name;
    hydrated.fixture_hydrated_from = "espn";
    const espnHomePlaceholder = isTrue(espn.home_is_placeholder);
    const espnAwayPlaceholder = isTrue(espn.away_is_placeholder);
    if (!espnHomePlaceholder && !espnAwayPlaceholder) {
      hydrated.home_team = espn.home_team;
      hydrated.away_team = espn.away_team;
      hydrated.home_is_placeholder = "False";
      hydrated.away_is_placeholder = "False";
    } else if (espnHomePlaceholder && espnAwayPlaceholder) {
      if (espn.home_slot) {
        hydrated.home_team = espn.home_slot;
        hydrated.home_is_placeholder = "True";
      }
      if (espn.away_slot) {
        hydrated.away_team = espn.away_slot;
        hydrated.away_is_placeholder = "True";
      }
    } else {
      hydrated.fixture_hydrated_from = "espn_partial";
    }
    return hydrated;
  });
}

function buildSchedule(fixtures, oddsRows, espnFixtures = []) {
  const hydratedFixtures = hydrateFixturesWithEspn(fixtures, espnFixtures);
  const groupFixtures = fixtures.filter((row) => String(row.stage || "").toLowerCase().includes("group"));
  const scheduleRows = [];
  const reportRows = [];
  const usedOddsIndexes = new Set();
  for (const fixture of hydratedFixtures) {
    const odds = findOddsForFixture(fixture, oddsRows, usedOddsIndexes);
    const oddsHasValues = odds ? oddsAvailable(odds) : false;
    const [placeholderHome, placeholderAway] = splitPlaceholder(fixture.group);
    const localHome = cleanLocalTeam(fixture.home_team, fixture.group, "home");
    const localAway = cleanLocalTeam(fixture.away_team, fixture.group, "away");
    const homeTeam = odds ? odds.home_team : localHome;
    const awayTeam = odds ? odds.away_team : localAway;
    const homePlaceholder = isPlaceholderTeam(homeTeam);
    const awayPlaceholder = isPlaceholderTeam(awayTeam);
    const output = {
      date: fixture.date,
      match_number: fixture.match_number,
      tournament: fixture.tournament || "FIFA World Cup",
      stage: fixture.stage,
      group: fixture.group,
      home_team: homeTeam,
      away_team: awayTeam,
      city: fixture.city,
      country: fixture.country,
      venue: odds?.venue || fixture.venue,
      neutral: fixture.neutral,
      home_is_placeholder: odds
        ? "False"
        : boolString(isTrue(fixture.home_is_placeholder) || homePlaceholder || !localHome),
      away_is_placeholder: odds
        ? "False"
        : boolString(isTrue(fixture.away_is_placeholder) || awayPlaceholder || !localAway),
      odds_available: oddsHasValues ? "1" : "0",
      home_odds: odds?.home_odds ?? "",
      draw_odds: odds?.draw_odds ?? "",
      away_odds: odds?.away_odds ?? "",
      max_home_odds: odds?.max_home_odds ?? "",
      max_draw_odds: odds?.max_draw_odds ?? "",
      max_away_odds: odds?.max_away_odds ?? "",
      n_odds_home_win: odds?.n_odds_home_win ?? "",
      n_odds_draw: odds?.n_odds_draw_win ?? "",
      n_odds_away_win: odds?.n_odds_away_win ?? "",
      oddsportal_event_id: odds?.oddsportal_event_id ?? "",
      oddsportal_encoded_event_id: odds?.oddsportal_encoded_event_id ?? "",
      oddsportal_home_team: odds?.oddsportal_home_team ?? "",
      oddsportal_away_team: odds?.oddsportal_away_team ?? "",
      oddsportal_date_utc: odds?.oddsportal_date_utc ?? "",
      oddsportal_kickoff_utc: odds?.oddsportal_kickoff_utc ?? "",
      oddsportal_detail_url: odds?.detail_url ?? "",
      oddsportal_source_url: odds?.source_url ?? "",
      local_home_team: fixture.home_team,
      local_away_team: fixture.away_team,
      local_kickoff_at: fixture.kickoff_at,
      placeholder_home_label: odds ? "" : homePlaceholder ? homeTeam : placeholderHome,
      placeholder_away_label: odds ? "" : awayPlaceholder ? awayTeam : placeholderAway,
      espn_event_id: fixture.espn_event_id ?? "",
      espn_home_team: fixture.espn_home_team ?? "",
      espn_away_team: fixture.espn_away_team ?? "",
      espn_home_slot: fixture.espn_home_slot ?? "",
      espn_away_slot: fixture.espn_away_slot ?? "",
      espn_kickoff_utc: fixture.espn_kickoff_utc ?? "",
      merge_source: odds
        ? "oddsportal_fixture"
        : fixture.fixture_hydrated_from === "espn"
          ? "espn_fixture_no_odds"
          : fixture.fixture_hydrated_from === "espn_partial"
            ? "espn_partial_fixture_no_odds"
            : "local_fixture_skeleton_no_odds",
      scraped_at_utc: odds?.scraped_at_utc ?? new Date().toISOString(),
    };
    scheduleRows.push(output);
    reportRows.push({
      match_number: output.match_number,
      date: output.date,
      stage: output.stage,
      home_team: output.home_team,
      away_team: output.away_team,
      odds_available: output.odds_available,
      merge_source: output.merge_source,
      espn_event_id: output.espn_event_id,
    });
  }
  return { scheduleRows, reportRows, groupFixtures, matchedOdds: usedOddsIndexes.size };
}

function validateSchedule(scheduleRows, expectedGroupRows) {
  const groupRows = scheduleRows.filter((row) => String(row.stage || "").toLowerCase().includes("group"));
  if (groupRows.length !== expectedGroupRows) {
    throw new Error(`Expected ${expectedGroupRows} group fixture rows, got ${groupRows.length}.`);
  }
  const seen = new Map();
  for (const row of groupRows) {
    const key = [row.date, normalizeTeamKey(row.home_team), normalizeTeamKey(row.away_team)].join("|");
    seen.set(key, (seen.get(key) || 0) + 1);
  }
  const duplicates = [...seen.entries()].filter(([, count]) => count > 1);
  if (duplicates.length) {
    throw new Error(`Duplicate group fixtures after odds merge: ${duplicates.map(([key]) => key).join("; ")}`);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const cachedRawOdds = oddsRowsFromRaw(args.rawOutput);
  let freshOddsRows = [];
  let usedFallbackOdds = false;
  let scrapeFailed = false;
  if (!args.skipOddsScrape) {
    try {
      freshOddsRows = await scrapeOddsPortal(args.url);
    } catch (error) {
      const scheduleFallbackRows = fallbackOddsRowsFromSchedule(args.output);
      freshOddsRows = scheduleFallbackRows;
      if (!cachedRawOdds.length && !scheduleFallbackRows.length) {
        throw error;
      }
      usedFallbackOdds = true;
      scrapeFailed = true;
      console.warn(
        `WARNING: OddsPortal refresh failed; reused ${cachedRawOdds.length} cached raw odds rows and ${scheduleFallbackRows.length} existing schedule odds rows. ${error.message}`,
      );
    }
  }
  const oddsRows = mergeOddsRows(cachedRawOdds, freshOddsRows);
  const fixtures = parseCsv(fs.readFileSync(args.fixtures, "utf8"));
  let espnFixtures = [];
  if (!args.skipEspnFixtures) {
    try {
      espnFixtures = await fetchEspnFixtures(args.espnScoreboardUrl);
    } catch (error) {
      console.warn(`WARNING: ESPN fixture hydration failed; continuing with local skeleton. ${error.message}`);
    }
  }
  const { scheduleRows, reportRows, groupFixtures, matchedOdds } = buildSchedule(fixtures, oddsRows, espnFixtures);
  validateSchedule(scheduleRows, groupFixtures.length);
  if (args.espnFixturesOutput) {
    writeCsv(args.espnFixturesOutput, espnFixtures, [
      "source",
      "espn_event_id",
      "match_number",
      "date",
      "kickoff_utc",
      "stage",
      "home_team",
      "away_team",
      "home_slot",
      "away_slot",
      "home_is_placeholder",
      "away_is_placeholder",
      "espn_home_team",
      "espn_away_team",
      "espn_name",
    ]);
  }
  writeCsv(args.rawOutput, oddsRows, [
    "source",
    "source_url",
    "scraped_at_utc",
    "oddsportal_event_id",
    "oddsportal_encoded_event_id",
    "oddsportal_date_utc",
    "oddsportal_kickoff_utc",
    "home_team",
    "away_team",
    "oddsportal_home_team",
    "oddsportal_away_team",
    "home_odds",
    "draw_odds",
    "away_odds",
    "max_home_odds",
    "max_draw_odds",
    "max_away_odds",
    "n_odds_home_win",
    "n_odds_draw_win",
    "n_odds_draw",
    "n_odds_away_win",
    "tournament",
    "event_stage",
    "venue",
    "venue_town",
    "venue_country",
    "detail_url",
  ]);
  writeCsv(args.output, scheduleRows, [
    "date",
    "match_number",
    "tournament",
    "stage",
    "group",
    "home_team",
    "away_team",
    "city",
    "country",
    "venue",
    "neutral",
    "home_is_placeholder",
    "away_is_placeholder",
    "odds_available",
    "home_odds",
    "draw_odds",
    "away_odds",
    "max_home_odds",
    "max_draw_odds",
    "max_away_odds",
    "n_odds_home_win",
    "n_odds_draw",
    "n_odds_away_win",
    "oddsportal_event_id",
    "oddsportal_encoded_event_id",
    "oddsportal_date_utc",
    "oddsportal_kickoff_utc",
    "oddsportal_detail_url",
    "oddsportal_source_url",
    "local_home_team",
    "local_away_team",
    "local_kickoff_at",
    "placeholder_home_label",
    "placeholder_away_label",
    "espn_event_id",
    "espn_home_team",
    "espn_away_team",
    "espn_home_slot",
    "espn_away_slot",
    "espn_kickoff_utc",
    "merge_source",
  ]);
  writeCsv(args.report, reportRows, [
    "match_number",
    "date",
    "stage",
    "home_team",
    "away_team",
    "odds_available",
    "merge_source",
    "espn_event_id",
  ]);
  const withOdds = scheduleRows.filter((row) => row.odds_available === "1").length;
  const knockoutWithOdds = scheduleRows.filter(
    (row) => row.odds_available === "1" && !String(row.stage || "").toLowerCase().includes("group"),
  ).length;
  console.log(`OddsPortal fixtures scraped: ${scrapeFailed ? 0 : freshOddsRows.length}`);
  console.log(`Cached raw odds rows loaded: ${cachedRawOdds.length}`);
  if (usedFallbackOdds) {
    console.log(`Fallback odds rows reused: ${freshOddsRows.length}`);
  }
  console.log(`Merged odds rows available: ${oddsRows.length}`);
  console.log(`ESPN fixtures parsed: ${espnFixtures.length}`);
  console.log(`Local group fixture slots: ${groupFixtures.length}`);
  console.log(`Matched OddsPortal fixtures: ${matchedOdds}`);
  console.log(`Wrote ${scheduleRows.length} schedule rows to ${args.output}`);
  console.log(`Rows with 1X2 odds: ${withOdds}`);
  console.log(`Knockout rows with 1X2 odds: ${knockoutWithOdds}`);
  console.log(`Rows without odds: ${scheduleRows.length - withOdds}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
