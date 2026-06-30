"use client";

import { FormEvent, useState } from "react";

type Status = {
  kind: "idle" | "loading" | "success" | "error";
  message: string;
  actionsUrl?: string;
};

export type LockStage = {
  stage: string;
  label: string;
  rows: number;
  lockedRows: number;
  playedRows: number;
  fixtureConfirmed: boolean;
};

export type EditableMatch = {
  matchNumber: string;
  label: string;
  score: string;
};

type RebuildControlProps = {
  lockStages?: LockStage[];
  defaultLockStage?: string;
  editableMatches?: EditableMatch[];
};

function scoreParts(score: string) {
  const match = String(score || "").match(/^(\d+)-(\d+)$/);
  return match ? [match[1], match[2]] : ["", ""];
}

function validGoalInput(value: string) {
  if (!/^\d{1,2}$/.test(value)) return false;
  const goals = Number(value);
  return Number.isInteger(goals) && goals >= 0 && goals <= 20;
}

export function RebuildControl({
  lockStages = [],
  defaultLockStage = "",
  editableMatches = [],
}: RebuildControlProps) {
  const initialEditableMatch = editableMatches[0];
  const [initialHomeScore, initialAwayScore] = scoreParts(initialEditableMatch?.score || "");
  const [code, setCode] = useState("");
  const [lockCode, setLockCode] = useState("");
  const [scoreCode, setScoreCode] = useState("");
  const [lockStage, setLockStage] = useState(defaultLockStage || lockStages[0]?.stage || "");
  const [scoreMatchNumber, setScoreMatchNumber] = useState(initialEditableMatch?.matchNumber || "");
  const [homeScore, setHomeScore] = useState(initialHomeScore);
  const [awayScore, setAwayScore] = useState(initialAwayScore);
  const [updateSoccerbase, setUpdateSoccerbase] = useState(true);
  const [deployToVercel, setDeployToVercel] = useState(true);
  const [status, setStatus] = useState<Status>({
    kind: "idle",
    message: "Start alleen een update na een ronde of als uitslagen/odds echt veranderd zijn.",
  });
  const [lockStatus, setLockStatus] = useState<Status>({
    kind: "idle",
    message: "Zet een ronde pas vast nadat je die scores echt hebt ingevuld.",
  });
  const [unlockStatus, setUnlockStatus] = useState<Status>({
    kind: "idle",
    message: "Alleen gebruiken als je per ongeluk een ronde hebt vastgezet.",
  });
  const [scoreStatus, setScoreStatus] = useState<Status>({
    kind: "idle",
    message: editableMatches.length
      ? "Wijzigt één ingevulde score, zet die wedstrijd vast en start daarna een rebuild."
      : "Er zijn nu geen bevestigde, nog te spelen wedstrijden om aan te passen.",
  });
  const selectedLockStage = lockStages.find((stage) => stage.stage === lockStage);
  const canLockSelectedStage = Boolean(selectedLockStage?.fixtureConfirmed);
  const hasValidManualScore = validGoalInput(homeScore) && validGoalInput(awayScore);

  function selectScoreMatch(matchNumber: string) {
    setScoreMatchNumber(matchNumber);
    const selected = editableMatches.find((match) => match.matchNumber === matchNumber);
    const [nextHomeScore, nextAwayScore] = scoreParts(selected?.score || "");
    setHomeScore(nextHomeScore);
    setAwayScore(nextAwayScore);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setStatus({ kind: "loading", message: "Workflow starten..." });

    try {
      const response = await fetch("/api/rebuild", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code,
          simulations: 10000,
          modelProfile: "best_default",
          useLineups: false,
          updateSoccerbase,
          deployToVercel,
        }),
      });
      const data = (await response.json()) as {
        ok?: boolean;
        message?: string;
        actionsUrl?: string;
        detail?: string;
      };

      if (!response.ok || !data.ok) {
        setStatus({
          kind: "error",
          message: data.message || "Update starten is mislukt.",
        });
        return;
      }

      setStatus({
        kind: "success",
        message: data.message || "Workflow gestart.",
        actionsUrl: data.actionsUrl,
      });
      setCode("");
    } catch {
      setStatus({ kind: "error", message: "Geen verbinding met de update-route." });
    }
  }

  async function lockRound(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLockStatus({ kind: "loading", message: "Ronde vastzetten en rebuild starten..." });

    try {
      const response = await fetch("/api/lock-round", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code: lockCode,
          stage: lockStage,
          updateSoccerbase: false,
          deployToVercel: true,
        }),
      });
      const data = (await response.json()) as {
        ok?: boolean;
        message?: string;
        actionsUrl?: string;
      };

      if (!response.ok || !data.ok) {
        setLockStatus({
          kind: "error",
          message: data.message || "Ronde vastzetten is mislukt.",
        });
        return;
      }

      setLockStatus({
        kind: "success",
        message: data.message || "Ronde vastgezet.",
        actionsUrl: data.actionsUrl,
      });
      setLockCode("");
    } catch {
      setLockStatus({ kind: "error", message: "Geen verbinding met de lock-route." });
    }
  }

  async function unlockRound(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setUnlockStatus({ kind: "loading", message: "Ronde ontgrendelen en rebuild starten..." });

    try {
      const response = await fetch("/api/lock-round", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "unlock",
          code: lockCode,
          stage: lockStage,
          updateSoccerbase: false,
          deployToVercel: true,
        }),
      });
      const data = (await response.json()) as {
        ok?: boolean;
        message?: string;
        actionsUrl?: string;
      };

      if (!response.ok || !data.ok) {
        setUnlockStatus({
          kind: "error",
          message: data.message || "Ronde ontgrendelen is mislukt.",
        });
        return;
      }

      setUnlockStatus({
        kind: "success",
        message: data.message || "Ronde ontgrendeld.",
        actionsUrl: data.actionsUrl,
      });
      setLockCode("");
    } catch {
      setUnlockStatus({ kind: "error", message: "Geen verbinding met de lock-route." });
    }
  }

  async function updateScore(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setScoreStatus({ kind: "loading", message: "Score opslaan en rebuild starten..." });

    try {
      const response = await fetch("/api/lock-round", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "update-score",
          code: scoreCode,
          matchNumber: scoreMatchNumber,
          homeScore,
          awayScore,
          updateSoccerbase: false,
          deployToVercel: true,
        }),
      });
      const data = (await response.json()) as {
        ok?: boolean;
        message?: string;
        actionsUrl?: string;
      };

      if (!response.ok || !data.ok) {
        setScoreStatus({ kind: "error", message: data.message || "Score wijzigen is mislukt." });
        return;
      }

      setScoreStatus({
        kind: "success",
        message: data.message || "Score gewijzigd.",
        actionsUrl: data.actionsUrl,
      });
      setScoreCode("");
    } catch {
      setScoreStatus({ kind: "error", message: "Geen verbinding met de score-route." });
    }
  }

  return (
    <div className="control-stack">
      <form className="control-panel" onSubmit={submit}>
        <div className="control-grid">
          <label className="control-field">
            <span>Code</span>
            <input
              autoComplete="one-time-code"
              inputMode="numeric"
              onChange={(event) => setCode(event.target.value)}
              placeholder="Update-code"
              type="password"
              value={code}
            />
          </label>

          <label className="check-row">
            <input
              checked={updateSoccerbase}
              onChange={(event) => setUpdateSoccerbase(event.target.checked)}
              type="checkbox"
            />
            Soccerbase verversen
          </label>

          <label className="check-row">
            <input
              checked={deployToVercel}
              onChange={(event) => setDeployToVercel(event.target.checked)}
              type="checkbox"
            />
            Daarna live zetten
          </label>

          <button className="primary-button" disabled={status.kind === "loading" || !code.trim()} type="submit">
            {status.kind === "loading" ? "Bezig..." : "Run update"}
          </button>
        </div>

        <div className={`status-line ${status.kind === "error" ? "red" : status.kind === "success" ? "green" : ""}`}>
          {status.message}
          {status.actionsUrl ? (
            <>
              {" "}
              <a href={status.actionsUrl} rel="noreferrer" target="_blank">
                Bekijk GitHub Actions
              </a>
            </>
          ) : null}
        </div>
      </form>

      <form className="control-panel" onSubmit={lockRound}>
        <div className="control-grid lock-grid">
          <label className="control-field">
            <span>Ronde vastzetten</span>
            <select onChange={(event) => setLockStage(event.target.value)} value={lockStage}>
              {lockStages.map((stage) => (
                <option key={stage.stage} value={stage.stage}>
                  {stage.label} ({stage.rows - stage.playedRows} nog te spelen, {stage.lockedRows} vast
                  {stage.fixtureConfirmed ? "" : ", voorlopig"})
                </option>
              ))}
            </select>
          </label>

          <label className="control-field">
            <span>Code</span>
            <input
              autoComplete="one-time-code"
              inputMode="numeric"
              onChange={(event) => setLockCode(event.target.value)}
              placeholder="Update-code"
              type="password"
              value={lockCode}
            />
          </label>

          <button
            className="primary-button"
            disabled={lockStatus.kind === "loading" || !lockCode.trim() || !lockStage || !canLockSelectedStage}
            type="submit"
          >
            {lockStatus.kind === "loading" ? "Bezig..." : "Zet ronde vast"}
          </button>
        </div>

        <div
          className={`status-line ${lockStatus.kind === "error" ? "red" : lockStatus.kind === "success" ? "green" : ""}`}
        >
          {lockStatus.message}
          {lockStatus.actionsUrl ? (
            <>
              {" "}
              <a href={lockStatus.actionsUrl} rel="noreferrer" target="_blank">
                Bekijk GitHub Actions
              </a>
            </>
          ) : null}
        </div>
      </form>

      <form className="control-panel subtle-control-panel" onSubmit={unlockRound}>
        <div className="control-grid unlock-grid">
          <div className="control-copy">
            <strong>Ronde ontgrendelen</strong>
            <span>Gebruikt dezelfde code en ronde hierboven. Alleen nodig als je per ongeluk hebt vastgezet.</span>
          </div>

          <button
            className="secondary-button danger-button"
            disabled={unlockStatus.kind === "loading" || !lockCode.trim() || !lockStage}
            type="submit"
          >
            {unlockStatus.kind === "loading" ? "Bezig..." : "Ontgrendel ronde"}
          </button>
        </div>

        <div
          className={`status-line ${
            unlockStatus.kind === "error" ? "red" : unlockStatus.kind === "success" ? "green" : ""
          }`}
        >
          {unlockStatus.message}
          {unlockStatus.actionsUrl ? (
            <>
              {" "}
              <a href={unlockStatus.actionsUrl} rel="noreferrer" target="_blank">
                Bekijk GitHub Actions
              </a>
            </>
          ) : null}
        </div>
      </form>

      <form className="control-panel" data-testid="manual-score-form" onSubmit={updateScore}>
        <div className="control-grid manual-score-grid">
          <label className="control-field">
            <span>Handmatige score wijzigen</span>
            <select
              disabled={!editableMatches.length}
              onChange={(event) => selectScoreMatch(event.target.value)}
              value={scoreMatchNumber}
            >
              {editableMatches.map((match) => (
                <option key={match.matchNumber} value={match.matchNumber}>
                  {match.label}
                </option>
              ))}
            </select>
          </label>

          <div className="score-inputs" aria-label="Nieuwe score">
            <label className="control-field">
              <span>Thuis</span>
              <input
                inputMode="numeric"
                max="20"
                min="0"
                onChange={(event) => setHomeScore(event.target.value)}
                type="number"
                value={homeScore}
              />
            </label>
            <span aria-hidden="true">-</span>
            <label className="control-field">
              <span>Uit</span>
              <input
                inputMode="numeric"
                max="20"
                min="0"
                onChange={(event) => setAwayScore(event.target.value)}
                type="number"
                value={awayScore}
              />
            </label>
          </div>

          <label className="control-field">
            <span>Code</span>
            <input
              autoComplete="one-time-code"
              inputMode="numeric"
              onChange={(event) => setScoreCode(event.target.value)}
              placeholder="Update-code"
              type="password"
              value={scoreCode}
            />
          </label>

          <button
            className="primary-button"
            disabled={
              scoreStatus.kind === "loading" ||
              !scoreCode.trim() ||
              !scoreMatchNumber ||
              !hasValidManualScore
            }
            type="submit"
          >
            {scoreStatus.kind === "loading" ? "Bezig..." : "Score opslaan"}
          </button>
        </div>

        <div
          className={`status-line ${scoreStatus.kind === "error" ? "red" : scoreStatus.kind === "success" ? "green" : ""}`}
        >
          {scoreStatus.message}
          {scoreStatus.actionsUrl ? (
            <>
              {" "}
              <a href={scoreStatus.actionsUrl} rel="noreferrer" target="_blank">
                Bekijk GitHub Actions
              </a>
            </>
          ) : null}
        </div>
      </form>
    </div>
  );
}
