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
};

type RebuildControlProps = {
  lockStages?: LockStage[];
  defaultLockStage?: string;
};

export function RebuildControl({ lockStages = [], defaultLockStage = "" }: RebuildControlProps) {
  const [code, setCode] = useState("");
  const [lockCode, setLockCode] = useState("");
  const [lockStage, setLockStage] = useState(defaultLockStage || lockStages[0]?.stage || "");
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
                  {stage.label} ({stage.rows - stage.playedRows} nog te spelen, {stage.lockedRows} vast)
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
            disabled={lockStatus.kind === "loading" || !lockCode.trim() || !lockStage}
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
    </div>
  );
}
