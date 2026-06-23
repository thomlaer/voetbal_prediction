import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import { fetchEspnJson, parseEspnMatches, SCOREBOARD_URL } from "../../espnLiveData";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type Prediction = {
  match_number: string | number;
  date: string;
  stage: string;
  group?: string;
  home_team: string;
  away_team: string;
  score: string;
  filled_score?: string;
  filled_predicted_winner?: string;
  pre_match_score?: string;
  pre_match_predicted_winner?: string;
  predicted_winner?: string;
  actual_available?: boolean;
  actual_score?: string;
  actual_winner?: string;
  actual_source?: string;
  prediction_exact?: boolean;
  prediction_outcome_correct?: boolean;
};

type DashboardData = {
  predictions: Prediction[];
};

type EspnMatch = ReturnType<typeof parseEspnMatches>[number];

const TEAM_ALIASES: Record<string, string> = {
  "bosnia herzegovina": "bosnia and herzegovina",
  "bosnia and herzegovina": "bosnia and herzegovina",
  czechia: "czech republic",
  "czech republic": "czech republic",
  "cote divoire": "ivory coast",
  "cote d ivoire": "ivory coast",
  "côte d ivoire": "ivory coast",
  curacao: "curacao",
  "curaçao": "curacao",
  "korea republic": "south korea",
  "south korea": "south korea",
  turkiye: "turkey",
  "türkiye": "turkey",
  usa: "united states",
  "u s a": "united states",
  "united states": "united states",
};

function normalizeTeam(value: string) {
  const normalized = value
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");

  return TEAM_ALIASES[normalized] || normalized;
}

function dateNumber(value: string) {
  const match = value.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!match) return null;
  return Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])) / 86400000;
}

function dateDistanceDays(left: string, right: string) {
  const leftNumber = dateNumber(left);
  const rightNumber = dateNumber(right);
  if (leftNumber === null || rightNumber === null) return Number.POSITIVE_INFINITY;
  return Math.abs(leftNumber - rightNumber);
}

function scoreOutcome(score: string) {
  const match = String(score || "").match(/^(\d+)-(\d+)$/);
  if (!match) return "";
  const homeGoals = Number(match[1]);
  const awayGoals = Number(match[2]);
  if (homeGoals > awayGoals) return "home";
  if (awayGoals > homeGoals) return "away";
  return "draw";
}

function winnerFromScore(row: Prediction, score: string) {
  const outcome = scoreOutcome(score);
  if (outcome === "home") return row.home_team;
  if (outcome === "away") return row.away_team;
  if (outcome === "draw") return "Draw";
  return "";
}

function predictedScore(row: Prediction) {
  return row.filled_score || row.pre_match_score || row.score || "";
}

function predictedWinner(row: Prediction) {
  return (
    winnerFromScore(row, predictedScore(row)) ||
    row.filled_predicted_winner ||
    row.pre_match_predicted_winner ||
    row.predicted_winner ||
    ""
  );
}

function actualScoreFromEspn(match: EspnMatch) {
  if (match.homeScore === null || match.awayScore === null) return "";
  return `${match.homeScore}-${match.awayScore}`;
}

function findEspnMatch(row: Prediction, matches: EspnMatch[]) {
  const rowHome = normalizeTeam(row.home_team);
  const rowAway = normalizeTeam(row.away_team);

  return matches
    .filter((match) => match.completed && actualScoreFromEspn(match))
    .map((match) => ({
      match,
      dateDistance: dateDistanceDays(row.date, match.date.slice(0, 10)),
      homeMatches: normalizeTeam(match.homeTeam) === rowHome,
      awayMatches: normalizeTeam(match.awayTeam) === rowAway,
    }))
    .filter((candidate) => candidate.dateDistance <= 1 && candidate.homeMatches && candidate.awayMatches)
    .sort((a, b) => a.dateDistance - b.dateDistance)[0]?.match;
}

function resultFlags(predictionScore: string, actualScore: string) {
  const exact = predictionScore === actualScore;
  const outcomeCorrect = Boolean(actualScore) && scoreOutcome(predictionScore) === scoreOutcome(actualScore);

  return {
    prediction_exact: exact,
    prediction_outcome_correct: outcomeCorrect,
  };
}

function resultClass(exact: boolean, outcomeCorrect: boolean, hasActual: boolean) {
  if (exact) return "green";
  if (outcomeCorrect) return "orange";
  if (hasActual) return "red";
  return "";
}

function resultLabel(exact: boolean, outcomeCorrect: boolean, hasActual: boolean) {
  if (exact) return "Exact";
  if (outcomeCorrect) return "Winnaar goed";
  if (hasActual) return "Mis";
  return "-";
}

async function loadDashboard() {
  const file = path.join(process.cwd(), "public", "data", "dashboard.json");
  return JSON.parse(await readFile(file, "utf8")) as DashboardData;
}

export async function GET() {
  try {
    const [dashboard, scoreboard] = await Promise.all([loadDashboard(), fetchEspnJson(SCOREBOARD_URL)]);
    const espnMatches = parseEspnMatches(scoreboard);

    const rows = dashboard.predictions
      .map((row) => {
        const espnMatch = findEspnMatch(row, espnMatches);
        const espnActualScore = espnMatch ? actualScoreFromEspn(espnMatch) : "";
        const storedActualScore = String(row.actual_score || "");
        const actualScore = espnActualScore || storedActualScore;
        const hasActual = Boolean(actualScore);
        const predictionScore = predictedScore(row);
        const flags = resultFlags(predictionScore, actualScore);
        const source = espnActualScore ? "ESPN live" : row.actual_source || "dashboard";
        const backupMismatch = Boolean(espnActualScore && storedActualScore && espnActualScore !== storedActualScore);

        if (!hasActual) return null;

        return {
          match_number: String(row.match_number),
          date: row.date,
          stage: row.stage,
          group: row.group || "",
          home_team: row.home_team,
          away_team: row.away_team,
          predicted_score: predictionScore,
          predicted_winner: predictedWinner(row),
          actual_score: actualScore,
          actual_winner: winnerFromScore(row, actualScore),
          actual_source: source,
          backup_actual_score: storedActualScore,
          backup_mismatch: backupMismatch,
          prediction_exact: flags.prediction_exact,
          prediction_outcome_correct: flags.prediction_outcome_correct,
          result_class: resultClass(flags.prediction_exact, flags.prediction_outcome_correct, hasActual),
          result_label: resultLabel(flags.prediction_exact, flags.prediction_outcome_correct, hasActual),
        };
      })
      .filter((row): row is NonNullable<typeof row> => row !== null)
      .sort((a, b) => Number(a.match_number) - Number(b.match_number));

    return NextResponse.json({
      ok: true,
      fetchedAt: new Date().toISOString(),
      source: SCOREBOARD_URL,
      rows,
      summary: {
        played: rows.length,
        exact: rows.filter((row) => row.prediction_exact).length,
        outcome: rows.filter((row) => row.prediction_outcome_correct).length,
        backupMismatches: rows.filter((row) => row.backup_mismatch).length,
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        message: error instanceof Error ? error.message : "Live gespeelde wedstrijden ophalen is mislukt.",
      },
      { status: 502 },
    );
  }
}
