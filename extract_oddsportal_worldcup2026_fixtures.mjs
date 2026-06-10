#!/usr/bin/env node

/*
Create a World Cup 2026 fixture schedule with current OddsPortal 1X2 odds.

OddsPortal currently exposes the group-stage fixtures and odds through the
"next matches" page. Knockout fixtures do not have teams/odds yet, so this
script appends them from the local fixture skeleton and leaves odds blank.
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

const DECRYPTION_KEY_TEXT = "J*8sQ!p$7aD_fR2yW@gHn*3bVp#sAdLd_k";
const DECRYPTION_SALT_TEXT = "5b9a8f2c3e6d1a4b7c8e9d0f1a2b3c4d";

const TEAM_ALIASES = new Map(
  Object.entries({
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    Curacao: "Curaçao",
    "D.R. Congo": "DR Congo",
    USA: "United States",
  }),
);

function parseArgs(argv) {
  const args = {
    url: DEFAULT_URL,
    fixtures: DEFAULT_FIXTURES,
    output: DEFAULT_OUTPUT,
    rawOutput: DEFAULT_RAW_OUTPUT,
    report: DEFAULT_REPORT,
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
  --raw-output PATH Raw OddsPortal group-stage fixture odds.
  --report PATH     Coverage report CSV.
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

function cleanLocalTeam(value, groupValue, side) {
  const text = String(value || "").trim();
  if (text && text.toLowerCase() !== "nan") {
    return modelTeamName(text);
  }
  const [home, away] = splitPlaceholder(groupValue);
  return side === "home" ? home : away;
}

function buildSchedule(fixtures, oddsRows) {
  const groupFixtures = fixtures.filter((row) => String(row.stage || "").toLowerCase().includes("group"));
  const scheduleRows = [];
  const reportRows = [];
  let oddsIndex = 0;
  for (const fixture of fixtures) {
    const isGroup = String(fixture.stage || "").toLowerCase().includes("group");
    const odds = isGroup ? oddsRows[oddsIndex++] : null;
    const oddsHasValues = odds ? oddsAvailable(odds) : false;
    const [placeholderHome, placeholderAway] = splitPlaceholder(fixture.group);
    const output = {
      date: fixture.date,
      match_number: fixture.match_number,
      tournament: fixture.tournament || "FIFA World Cup",
      stage: fixture.stage,
      group: fixture.group,
      home_team: odds ? odds.home_team : cleanLocalTeam(fixture.home_team, fixture.group, "home"),
      away_team: odds ? odds.away_team : cleanLocalTeam(fixture.away_team, fixture.group, "away"),
      city: fixture.city,
      country: fixture.country,
      venue: odds?.venue || fixture.venue,
      neutral: fixture.neutral,
      home_is_placeholder: odds
        ? "False"
        : boolString(isTrue(fixture.home_is_placeholder) || !cleanLocalTeam(fixture.home_team, fixture.group, "home")),
      away_is_placeholder: odds
        ? "False"
        : boolString(isTrue(fixture.away_is_placeholder) || !cleanLocalTeam(fixture.away_team, fixture.group, "away")),
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
      placeholder_home_label: odds ? "" : placeholderHome,
      placeholder_away_label: odds ? "" : placeholderAway,
      merge_source: odds ? "oddsportal_group_fixture" : "local_fixture_skeleton_no_odds",
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
    });
  }
  return { scheduleRows, reportRows, groupFixtures };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const oddsRows = await scrapeOddsPortal(args.url);
  const fixtures = parseCsv(fs.readFileSync(args.fixtures, "utf8"));
  const { scheduleRows, reportRows, groupFixtures } = buildSchedule(fixtures, oddsRows);
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
    "n_odds_away_win",
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
    "oddsportal_kickoff_utc",
    "oddsportal_detail_url",
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
  ]);
  const withOdds = scheduleRows.filter((row) => row.odds_available === "1").length;
  console.log(`OddsPortal group fixtures: ${oddsRows.length}`);
  console.log(`Local group fixture slots: ${groupFixtures.length}`);
  console.log(`Wrote ${scheduleRows.length} schedule rows to ${args.output}`);
  console.log(`Rows with 1X2 odds: ${withOdds}`);
  console.log(`Rows without odds: ${scheduleRows.length - withOdds}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
