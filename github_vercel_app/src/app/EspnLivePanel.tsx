"use client";

import { useEffect, useMemo, useState } from "react";

type EspnMatch = {
  id: string;
  date: string;
  status: string;
  statusDetail: string;
  completed: boolean;
  inProgress: boolean;
  homeTeam: string;
  awayTeam: string;
  homeScore: number | null;
  awayScore: number | null;
};

type EspnStandingTeam = {
  rank: number;
  team: string;
  played: number;
  wins: number;
  draws: number;
  losses: number;
  goalsFor: number;
  goalsAgainst: number;
  goalDifference: number;
  points: number;
  advanced: number;
};

type EspnStandingGroup = {
  group: string;
  teams: EspnStandingTeam[];
};

type EspnLiveData = {
  ok: boolean;
  fetchedAt: string;
  matches: EspnMatch[];
  standings: EspnStandingGroup[];
  message?: string;
};

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value || "-";
  return date.toLocaleString("nl-NL", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function score(match: EspnMatch) {
  if (match.homeScore === null || match.awayScore === null) return "-";
  return `${match.homeScore}-${match.awayScore}`;
}

export function EspnLivePanel() {
  const [data, setData] = useState<EspnLiveData | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [message, setMessage] = useState("ESPN direct uitlezen zonder modelrun.");

  async function loadLiveData() {
    setStatus("loading");
    setMessage("ESPN live data ophalen...");

    try {
      const response = await fetch("/api/espn-live", { cache: "no-store" });
      const payload = (await response.json()) as EspnLiveData;

      if (!response.ok || !payload.ok) {
        setStatus("error");
        setMessage(payload.message || "ESPN live data ophalen is mislukt.");
        return;
      }

      setData(payload);
      setStatus("success");
      setMessage("Live check bijgewerkt.");
    } catch {
      setStatus("error");
      setMessage("Geen verbinding met de ESPN live route.");
    }
  }

  useEffect(() => {
    void loadLiveData();
  }, []);

  const completedMatches = useMemo(
    () => (data?.matches || []).filter((match) => match.completed).slice(0, 10),
    [data],
  );
  const liveMatches = useMemo(
    () => (data?.matches || []).filter((match) => match.inProgress).slice(0, 6),
    [data],
  );

  return (
    <div className="live-panel">
      <div className="live-panel-head">
        <div>
          <strong>ESPN Live Check</strong>
          <div className="metric-note">
            Uitslagen en poulestanden zoals ESPN ze nu toont. Dit overschrijft geen vaste Scorito-score.
          </div>
        </div>
        <button className="secondary-button" disabled={status === "loading"} onClick={loadLiveData} type="button">
          {status === "loading" ? "Verversen..." : "Ververs ESPN"}
        </button>
      </div>

      <div className={`status-line ${status === "error" ? "red" : status === "success" ? "green" : ""}`}>
        {message}
        {data?.fetchedAt ? ` Laatst gelezen: ${formatDate(data.fetchedAt)}.` : ""}
      </div>

      <div className="live-grid">
        <div className="live-card">
          <div className="team-card-top">
            <strong>Laatste uitslagen</strong>
            <span className="pill">{completedMatches.length}</span>
          </div>
          <div className="live-list">
            {completedMatches.length ? (
              completedMatches.map((match) => (
                <div className="live-match" key={match.id || `${match.homeTeam}-${match.awayTeam}-${match.date}`}>
                  <div className="live-match-main">
                    <strong>
                      {match.homeTeam} - {match.awayTeam}
                    </strong>
                    <span className="metric-note">
                      {formatDate(match.date)} - {match.status || match.statusDetail || "ESPN"}
                    </span>
                  </div>
                  <span className="score actual-score">{score(match)}</span>
                </div>
              ))
            ) : (
              <div className="metric-note">Nog geen afgeronde ESPN-wedstrijden gevonden.</div>
            )}
          </div>
        </div>

        <div className="live-card">
          <div className="team-card-top">
            <strong>Nu live</strong>
            <span className="pill">{liveMatches.length}</span>
          </div>
          <div className="live-list">
            {liveMatches.length ? (
              liveMatches.map((match) => (
                <div className="live-match" key={match.id || `${match.homeTeam}-${match.awayTeam}-${match.date}`}>
                  <div className="live-match-main">
                    <strong>
                      {match.homeTeam} - {match.awayTeam}
                    </strong>
                    <span className="metric-note">{match.statusDetail || match.status || formatDate(match.date)}</span>
                  </div>
                  <span className="score">{score(match)}</span>
                </div>
              ))
            ) : (
              <div className="metric-note">Geen wedstrijd live volgens ESPN.</div>
            )}
          </div>
        </div>
      </div>

      <div className="live-standings-grid">
        {(data?.standings || []).map((group) => (
          <div className="live-group" key={group.group}>
            <div className="team-card-top">
              <strong>{group.group}</strong>
              <span className="pill">ESPN</span>
            </div>
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Team</th>
                  <th>W-D-L</th>
                  <th>GD</th>
                  <th>Pts</th>
                </tr>
              </thead>
              <tbody>
                {group.teams.map((team) => (
                  <tr key={`${group.group}-${team.team}`}>
                    <td className="mono">{team.rank}</td>
                    <td>
                      {team.advanced ? <strong>{team.team}</strong> : team.team}
                      {team.advanced ? <div className="metric-note">door volgens ESPN</div> : null}
                    </td>
                    <td className="mono">
                      {team.wins}-{team.draws}-{team.losses}
                    </td>
                    <td className="mono">{team.goalDifference > 0 ? `+${team.goalDifference}` : team.goalDifference}</td>
                    <td className="mono">{team.points}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  );
}
