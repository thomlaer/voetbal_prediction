"use client";

import { useEffect, useMemo, useState } from "react";

export type PlayedMatchRow = {
  match_number: string;
  date: string;
  stage: string;
  group?: string;
  home_team: string;
  away_team: string;
  predicted_score: string;
  predicted_winner: string;
  actual_score: string;
  actual_winner: string;
  actual_source: string;
  backup_actual_score?: string;
  backup_mismatch?: boolean;
  prediction_exact: boolean;
  prediction_outcome_correct: boolean;
  result_class: string;
  result_label: string;
};

type LivePlayedResponse = {
  ok?: boolean;
  fetchedAt?: string;
  rows?: PlayedMatchRow[];
  summary?: {
    played: number;
    exact: number;
    outcome: number;
    backupMismatches: number;
  };
  message?: string;
};

type PlayedMatchesPanelProps = {
  initialRows: PlayedMatchRow[];
};

function formatDate(value?: string) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("nl-NL", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function matchupLabel(row: PlayedMatchRow) {
  const homeIsWinner = row.predicted_winner === row.home_team;
  const awayIsWinner = row.predicted_winner === row.away_team;

  return (
    <>
      {homeIsWinner ? <strong>{row.home_team}</strong> : row.home_team} -{" "}
      {awayIsWinner ? <strong>{row.away_team}</strong> : row.away_team}
    </>
  );
}

export function PlayedMatchesPanel({ initialRows }: PlayedMatchesPanelProps) {
  const [rows, setRows] = useState(initialRows);
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [message, setMessage] = useState("ESPN live scores worden op de achtergrond gecontroleerd.");
  const [fetchedAt, setFetchedAt] = useState("");

  async function refreshPlayedMatches() {
    setStatus("loading");
    setMessage("Gespeelde wedstrijden via ESPN ophalen...");

    try {
      const response = await fetch("/api/live-played", { cache: "no-store" });
      const payload = (await response.json()) as LivePlayedResponse;

      if (!response.ok || !payload.ok || !payload.rows) {
        setStatus("error");
        setMessage(payload.message || "Live gespeelde wedstrijden ophalen is mislukt.");
        return;
      }

      setRows(payload.rows);
      setFetchedAt(payload.fetchedAt || "");
      setStatus("success");
      setMessage("ESPN live scores meegeteld.");
    } catch {
      setStatus("error");
      setMessage("Geen verbinding met de live gespeelde wedstrijden-route.");
    }
  }

  useEffect(() => {
    void refreshPlayedMatches();
  }, []);

  const summary = useMemo(
    () => ({
      exact: rows.filter((row) => row.prediction_exact).length,
      outcome: rows.filter((row) => row.prediction_outcome_correct).length,
      mismatches: rows.filter((row) => row.backup_mismatch).length,
    }),
    [rows],
  );

  return (
    <details className="section-details" open>
      <summary className="section-header collapse-summary">
        <div>
          <h2 className="section-title">Gespeelde Wedstrijden</h2>
          <p className="section-subtitle">
            {rows.length} gespeeld - {summary.exact} exact - {summary.outcome} winnaar goed.
          </p>
        </div>
        <span className="collapse-actions">
          <span className={`pill ${summary.mismatches ? "red" : "green"}`}>
            {summary.mismatches ? `${summary.mismatches} check` : "ESPN"}
          </span>
          <span className="collapse-caret" aria-hidden="true" />
        </span>
      </summary>

      <div className={`status-line ${status === "error" ? "red" : status === "success" ? "green" : ""}`}>
        {message}
        {fetchedAt ? ` Laatst gelezen: ${formatDate(fetchedAt)}.` : ""}
        <button className="link-button" disabled={status === "loading"} onClick={refreshPlayedMatches} type="button">
          {status === "loading" ? "Verversen..." : "Ververs"}
        </button>
      </div>

      <div className="table-shell">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Wedstrijd</th>
              <th>Voorspelling</th>
              <th>Uitslag</th>
              <th>Resultaat</th>
            </tr>
          </thead>
          <tbody>
            {rows.length ? (
              rows.map((row) => (
                <tr key={`played-${row.stage}-${row.match_number}`}>
                  <td className="mono">{row.match_number}</td>
                  <td>
                    {matchupLabel(row)}
                    <div className="metric-note">
                      {row.stage}
                      {row.group ? ` - Poule ${row.group}` : ""} - {row.date}
                    </div>
                  </td>
                  <td>
                    <span className="score">{row.predicted_score}</span>
                    <div className="metric-note">{row.predicted_winner}</div>
                  </td>
                  <td>
                    <span className="score actual-score">{row.actual_score || "-"}</span>
                    <div className="metric-note">
                      {row.actual_winner || "-"} - {row.actual_source || "-"}
                      {row.backup_mismatch ? ` - backup ${row.backup_actual_score}` : ""}
                    </div>
                  </td>
                  <td>
                    <span className={`pill ${row.result_class}`}>{row.result_label}</span>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={5}>Nog geen WK-wedstrijden met live of opgeslagen uitslag.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </details>
  );
}
