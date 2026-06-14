"use client";

import { FormEvent, useState } from "react";

type Status = {
  kind: "idle" | "loading" | "success" | "error";
  message: string;
  actionsUrl?: string;
};

export function RebuildControl() {
  const [code, setCode] = useState("");
  const [updateSoccerbase, setUpdateSoccerbase] = useState(true);
  const [deployToVercel, setDeployToVercel] = useState(true);
  const [status, setStatus] = useState<Status>({
    kind: "idle",
    message: "Start alleen een update na een ronde of als uitslagen/odds echt veranderd zijn.",
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

  return (
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
  );
}
