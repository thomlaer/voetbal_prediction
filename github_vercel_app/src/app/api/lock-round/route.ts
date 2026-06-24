import crypto from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DEFAULT_REPO = "thomlaer/voetbal_prediction";
const DEFAULT_WORKFLOW = "rebuild-predictions.yml";
const DEFAULT_REF = "main";
const LOCK_PATH = "data/extracted/scorito_locked_scores.csv";

type LockRequest = {
  action?: unknown;
  code?: unknown;
  stage?: unknown;
  updateSoccerbase?: unknown;
  deployToVercel?: unknown;
};

type Prediction = {
  match_number?: string | number;
  date?: string;
  stage?: string;
  group?: string;
  home_team?: string;
  away_team?: string;
  score?: string;
  filled_score?: string;
  predicted_winner?: string;
  filled_predicted_winner?: string;
  actual_available?: boolean;
  actual_score?: string;
};

type LockRow = {
  match_number: string;
  date: string;
  stage: string;
  home_team: string;
  away_team: string;
  score: string;
  predicted_winner: string;
  note: string;
};

function safeEquals(left: string, right: string) {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);
  return leftBuffer.length === rightBuffer.length && crypto.timingSafeEqual(leftBuffer, rightBuffer);
}

function textValue(value: unknown, fallback = "") {
  return typeof value === "string" ? value.trim() : fallback;
}

function boolValue(value: unknown, fallback: boolean) {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(normalized)) return true;
    if (["0", "false", "no", "off"].includes(normalized)) return false;
  }
  return fallback;
}

function parseScore(value: unknown) {
  return /^(\d+)-(\d+)$/.test(String(value || "").trim());
}

function isPlaceholder(value: unknown) {
  const text = String(value || "").trim().toLowerCase();
  return !text || text === "nan" || text.startsWith("winner ") || text.startsWith("group ") || /^w\d+|ru\d+|l\d+$/.test(text);
}

function csvEscape(value: unknown) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function parseCsvLine(line: string) {
  const cells: string[] = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (quoted) {
      if (char === '"' && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        current += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      cells.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  cells.push(current);
  return cells;
}

function parseCsv(text: string) {
  const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter((line) => line.trim());
  if (!lines.length) return [] as Record<string, string>[];
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] || ""]));
  });
}

function writeCsv(rows: Record<string, string>[]) {
  const headers = ["match_number", "date", "stage", "home_team", "away_team", "score", "predicted_winner", "note"];
  return `${headers.join(",")}\n${rows
    .map((row) => headers.map((header) => csvEscape(row[header] || "")).join(","))
    .join("\n")}\n`;
}

async function githubRequest(repository: string, token: string, url: string, init: RequestInit = {}) {
  return fetch(`https://api.github.com/repos/${repository}${url}`, {
    ...init,
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      "X-GitHub-Api-Version": "2022-11-28",
      ...(init.headers || {}),
    },
  });
}

async function loadRemoteLocks(repository: string, token: string, ref: string) {
  const response = await githubRequest(
    repository,
    token,
    `/contents/${LOCK_PATH}?ref=${encodeURIComponent(ref)}`,
    { method: "GET" },
  );
  if (response.status === 404) {
    return { sha: "", rows: [] as Record<string, string>[] };
  }
  if (!response.ok) {
    throw new Error((await response.text()).slice(0, 500));
  }
  const payload = (await response.json()) as { sha?: string; content?: string; encoding?: string };
  const content = payload.content ? Buffer.from(payload.content.replace(/\n/g, ""), "base64").toString("utf8") : "";
  return { sha: payload.sha || "", rows: parseCsv(content) };
}

async function dispatchRebuild(
  repository: string,
  workflow: string,
  ref: string,
  token: string,
  updateSoccerbase: boolean,
  deployToVercel: boolean,
) {
  const response = await githubRequest(repository, token, `/actions/workflows/${workflow}/dispatches`, {
    method: "POST",
    body: JSON.stringify({
      ref,
      inputs: {
        previous_label: "",
        simulations: "10000",
        model_profile: "best_default",
        use_lineups: false,
        update_soccerbase: updateSoccerbase,
        deploy_to_vercel: deployToVercel,
      },
    }),
  });
  if (!response.ok) {
    throw new Error((await response.text()).slice(0, 500));
  }
}

export async function POST(request: Request) {
  const expectedCode = process.env.REBUILD_ACTION_CODE || "";
  const token = process.env.GITHUB_ACTIONS_TOKEN || "";
  const repository = process.env.GITHUB_ACTIONS_REPO || DEFAULT_REPO;
  const workflow = process.env.GITHUB_ACTIONS_WORKFLOW || DEFAULT_WORKFLOW;
  const ref = process.env.GITHUB_ACTIONS_REF || DEFAULT_REF;

  if (!expectedCode || !token) {
    return NextResponse.json(
      { ok: false, message: "Server mist REBUILD_ACTION_CODE of GITHUB_ACTIONS_TOKEN." },
      { status: 503 },
    );
  }

  let body: LockRequest;
  try {
    body = (await request.json()) as LockRequest;
  } catch {
    return NextResponse.json({ ok: false, message: "Ongeldige request." }, { status: 400 });
  }

  const submittedCode = textValue(body.code);
  if (!submittedCode || !safeEquals(submittedCode, expectedCode)) {
    return NextResponse.json({ ok: false, message: "Code klopt niet." }, { status: 401 });
  }

  const stage = textValue(body.stage);
  if (!stage) {
    return NextResponse.json({ ok: false, message: "Kies eerst een ronde." }, { status: 400 });
  }
  const action = textValue(body.action, "lock").toLowerCase();
  if (!["lock", "unlock"].includes(action)) {
    return NextResponse.json({ ok: false, message: "Ongeldige ronde-actie." }, { status: 400 });
  }

  const dashboardPath = path.join(process.cwd(), "public", "data", "dashboard.json");
  const dashboard = JSON.parse(await readFile(dashboardPath, "utf8")) as { predictions?: Prediction[] };
  const stageRows = (dashboard.predictions || []).filter((row) => row.stage === stage);
  if (!stageRows.length) {
    return NextResponse.json({ ok: false, message: "Deze ronde staat niet in de huidige dashboarddata." }, { status: 404 });
  }

  try {
    const remote = await loadRemoteLocks(repository, token, ref);
    let merged: Record<string, string>[];
    let successMessage: string;
    let changedCount = 0;

    if (action === "unlock") {
      const before = remote.rows.length;
      merged = remote.rows.filter((row) => row.stage !== stage);
      const removed = before - merged.length;
      if (!removed) {
        return NextResponse.json(
          { ok: false, message: `Er stonden geen vastgezette scores voor ${stage}.` },
          { status: 400 },
        );
      }
      changedCount = removed;
      successMessage = `${removed} vastgezette scores verwijderd voor ${stage}. GitHub rebuild is gestart.`;
    } else {
      const locks = stageRows
        .filter((row) => !row.actual_available && !row.actual_score)
        .map((row) => {
          const score = String(row.filled_score || row.score || "").trim();
          const home = String(row.home_team || "").trim();
          const away = String(row.away_team || "").trim();
          if (!parseScore(score) || isPlaceholder(home) || isPlaceholder(away)) return null;
          return {
            match_number: String(row.match_number || "").trim(),
            date: String(row.date || "").slice(0, 10),
            stage,
            home_team: home,
            away_team: away,
            score,
            predicted_winner: String(row.filled_predicted_winner || row.predicted_winner || "").trim(),
            note: `locked_from_dashboard_${new Date().toISOString()}`,
          };
        })
        .filter((row): row is LockRow => Boolean(row?.match_number));

      if (!locks.length) {
        return NextResponse.json(
          { ok: false, message: "Geen bruikbare open wedstrijden gevonden om vast te zetten." },
          { status: 400 },
        );
      }
      const lockNumbers = new Set(locks.map((row) => row.match_number));
      merged = remote.rows.filter((row) => !lockNumbers.has(row.match_number));
      merged.push(...locks);
      changedCount = locks.length;
      successMessage = `${locks.length} scores vastgezet voor ${stage}. GitHub rebuild is gestart.`;
    }
    merged.sort((a, b) => Number(a.match_number || 0) - Number(b.match_number || 0));

    const response = await githubRequest(repository, token, `/contents/${LOCK_PATH}`, {
      method: "PUT",
      body: JSON.stringify({
        message: `${action === "unlock" ? "Unlock" : "Lock"} ${stage} Scorito scores`,
        content: Buffer.from(writeCsv(merged), "utf8").toString("base64"),
        branch: ref,
        ...(remote.sha ? { sha: remote.sha } : {}),
      }),
    });

    if (!response.ok) {
      return NextResponse.json(
        { ok: false, message: "Lock-bestand kon niet naar GitHub geschreven worden.", detail: (await response.text()).slice(0, 500) },
        { status: 502 },
      );
    }

    await dispatchRebuild(
      repository,
      workflow,
      ref,
      token,
      boolValue(body.updateSoccerbase, false),
      boolValue(body.deployToVercel, true),
    );

    return NextResponse.json({
      ok: true,
      message: successMessage,
      actionsUrl: `https://github.com/${repository}/actions/workflows/${workflow}`,
      locked: action === "lock" ? changedCount : 0,
      unlocked: action === "unlock" ? changedCount : 0,
      action,
    });
  } catch (error) {
    return NextResponse.json(
      { ok: false, message: "Ronde-actie is mislukt.", detail: error instanceof Error ? error.message : String(error) },
      { status: 502 },
    );
  }
}
