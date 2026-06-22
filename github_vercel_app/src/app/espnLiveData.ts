export const SCOREBOARD_URL =
  "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?limit=200&dates=20260611-20260720";
export const STANDINGS_URL = "https://site.web.api.espn.com/apis/v2/sports/soccer/fifa.world/standings?season=2026";

export type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function recordValue(value: unknown): JsonRecord {
  return isRecord(value) ? value : {};
}

function stringValue(value: unknown, fallback = "") {
  return typeof value === "string" ? value : fallback;
}

function numberValue(value: unknown, fallback: number | null = null) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

export async function fetchEspnJson(url: string) {
  const response = await fetch(url, {
    cache: "no-store",
    headers: {
      Accept: "application/json",
      "User-Agent": "voetbal-prediction-dashboard/1.0",
    },
  });

  if (!response.ok) {
    throw new Error(`ESPN ${response.status} for ${url}`);
  }

  return response.json() as Promise<JsonRecord>;
}

function nested(record: JsonRecord, ...keys: string[]) {
  let current: unknown = record;
  for (const key of keys) {
    if (!isRecord(current)) return undefined;
    current = current[key];
  }
  return current;
}

function parseCompetitor(value: unknown) {
  const competitor = recordValue(value);
  const team = recordValue(competitor.team);

  return {
    homeAway: stringValue(competitor.homeAway),
    name: stringValue(team.displayName) || stringValue(team.shortDisplayName) || stringValue(team.name),
    abbreviation: stringValue(team.abbreviation),
    score: numberValue(competitor.score),
  };
}

export function parseEspnMatches(scoreboard: JsonRecord) {
  return arrayValue(scoreboard.events)
    .map((eventValue) => {
      const event = recordValue(eventValue);
      const competition = recordValue(arrayValue(event.competitions)[0]);
      const competitors = arrayValue(competition.competitors).map(parseCompetitor);
      const home = competitors.find((team) => team.homeAway === "home") || competitors[0];
      const away = competitors.find((team) => team.homeAway === "away") || competitors[1];
      const statusType = recordValue(nested(event, "status", "type"));

      if (!home || !away) return null;

      return {
        id: stringValue(event.id),
        date: stringValue(event.date),
        name: stringValue(event.name),
        shortName: stringValue(event.shortName),
        status: stringValue(statusType.description) || stringValue(statusType.shortDetail),
        statusDetail: stringValue(statusType.detail),
        completed: statusType.completed === true,
        inProgress: statusType.state === "in",
        homeTeam: home.name,
        awayTeam: away.name,
        homeAbbreviation: home.abbreviation,
        awayAbbreviation: away.abbreviation,
        homeScore: home.score,
        awayScore: away.score,
      };
    })
    .filter((match): match is NonNullable<typeof match> => match !== null)
    .sort((a, b) => Date.parse(b.date || "0") - Date.parse(a.date || "0"));
}

function statMap(entry: JsonRecord) {
  const map = new Map<string, { displayValue: string; value: number | null }>();
  for (const statValue of arrayValue(entry.stats)) {
    const stat = recordValue(statValue);
    const name = stringValue(stat.name);
    if (!name) continue;
    map.set(name, {
      displayValue: stringValue(stat.displayValue),
      value: numberValue(stat.value),
    });
  }
  return map;
}

function statNumber(stats: Map<string, { value: number | null }>, key: string, fallback = 0) {
  return stats.get(key)?.value ?? fallback;
}

export function parseEspnStandings(standings: JsonRecord) {
  return arrayValue(standings.children)
    .map((groupValue) => {
      const group = recordValue(groupValue);
      const entries = arrayValue(nested(group, "standings", "entries"));
      const teams = entries.map((entryValue, index) => {
        const entry = recordValue(entryValue);
        const team = recordValue(entry.team);
        const stats = statMap(entry);

        return {
          rank: statNumber(stats, "rank", index + 1),
          team: stringValue(team.displayName) || stringValue(team.shortDisplayName) || stringValue(team.name),
          abbreviation: stringValue(team.abbreviation),
          played: statNumber(stats, "gamesPlayed"),
          wins: statNumber(stats, "wins"),
          draws: statNumber(stats, "ties"),
          losses: statNumber(stats, "losses"),
          goalsFor: statNumber(stats, "pointsFor"),
          goalsAgainst: statNumber(stats, "pointsAgainst"),
          goalDifference: statNumber(stats, "pointDifferential"),
          points: statNumber(stats, "points"),
          advanced: statNumber(stats, "advanced"),
        };
      });

      return {
        group: stringValue(group.name) || stringValue(group.abbreviation),
        teams,
      };
    })
    .filter((group) => group.group && group.teams.length);
}
