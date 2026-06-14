import crypto from "node:crypto";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DEFAULT_REPO = "thomlaer/voetbal_prediction";
const DEFAULT_WORKFLOW = "rebuild-predictions.yml";
const DEFAULT_REF = "main";
const ACTIONS_URL = `https://github.com/${DEFAULT_REPO}/actions/workflows/${DEFAULT_WORKFLOW}`;

type RebuildRequest = {
  code?: unknown;
  previousLabel?: unknown;
  simulations?: unknown;
  modelProfile?: unknown;
  useLineups?: unknown;
  updateSoccerbase?: unknown;
  deployToVercel?: unknown;
};

function safeEquals(left: string, right: string) {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);
  return leftBuffer.length === rightBuffer.length && crypto.timingSafeEqual(leftBuffer, rightBuffer);
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

function textValue(value: unknown, fallback = "") {
  return typeof value === "string" ? value.trim() : fallback;
}

function simulationValue(value: unknown) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "10000";
  return String(Math.min(50000, Math.max(1000, Math.round(parsed))));
}

export async function POST(request: Request) {
  const expectedCode = process.env.REBUILD_ACTION_CODE || "";
  const token = process.env.GITHUB_ACTIONS_TOKEN || "";
  const repository = process.env.GITHUB_ACTIONS_REPO || DEFAULT_REPO;
  const workflow = process.env.GITHUB_ACTIONS_WORKFLOW || DEFAULT_WORKFLOW;
  const ref = process.env.GITHUB_ACTIONS_REF || DEFAULT_REF;

  if (!expectedCode || !token) {
    return NextResponse.json(
      {
        ok: false,
        message: "Server mist REBUILD_ACTION_CODE of GITHUB_ACTIONS_TOKEN.",
      },
      { status: 503 },
    );
  }

  let body: RebuildRequest;
  try {
    body = (await request.json()) as RebuildRequest;
  } catch {
    return NextResponse.json({ ok: false, message: "Ongeldige request." }, { status: 400 });
  }

  const submittedCode = textValue(body.code);
  if (!submittedCode || !safeEquals(submittedCode, expectedCode)) {
    return NextResponse.json({ ok: false, message: "Code klopt niet." }, { status: 401 });
  }

  const modelProfile = textValue(body.modelProfile, "best_default");
  const safeModelProfile = ["best_default", "full_with_lineups"].includes(modelProfile) ? modelProfile : "best_default";
  const previousLabel = textValue(body.previousLabel);
  const safePreviousLabel = /^[A-Za-z0-9_-]{0,64}$/.test(previousLabel) ? previousLabel : "";

  const response = await fetch(
    `https://api.github.com/repos/${repository}/actions/workflows/${workflow}/dispatches`,
    {
      method: "POST",
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
      },
      body: JSON.stringify({
        ref,
        inputs: {
          previous_label: safePreviousLabel,
          simulations: simulationValue(body.simulations),
          model_profile: safeModelProfile,
          use_lineups: boolValue(body.useLineups, false),
          update_soccerbase: boolValue(body.updateSoccerbase, true),
          deploy_to_vercel: boolValue(body.deployToVercel, true),
        },
      }),
    },
  );

  if (!response.ok) {
    const detail = await response.text();
    return NextResponse.json(
      {
        ok: false,
        message: "GitHub workflow kon niet gestart worden.",
        detail: detail.slice(0, 500),
      },
      { status: 502 },
    );
  }

  return NextResponse.json({
    ok: true,
    message: "Workflow gestart. Wacht tot GitHub Actions klaar is en refresh daarna de site.",
    actionsUrl: `https://github.com/${repository}/actions/workflows/${workflow}`,
    fallbackActionsUrl: ACTIONS_URL,
  });
}
