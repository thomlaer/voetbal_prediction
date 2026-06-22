import { NextResponse } from "next/server";
import {
  fetchEspnJson,
  parseEspnMatches,
  parseEspnStandings,
  SCOREBOARD_URL,
  STANDINGS_URL,
} from "../../espnLiveData";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  try {
    const [scoreboard, standings] = await Promise.all([fetchEspnJson(SCOREBOARD_URL), fetchEspnJson(STANDINGS_URL)]);

    return NextResponse.json({
      ok: true,
      fetchedAt: new Date().toISOString(),
      sources: {
        scoreboard: SCOREBOARD_URL,
        standings: STANDINGS_URL,
      },
      matches: parseEspnMatches(scoreboard),
      standings: parseEspnStandings(standings),
    });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        message: error instanceof Error ? error.message : "ESPN live data ophalen is mislukt.",
      },
      { status: 502 },
    );
  }
}
