import { readFile } from "node:fs/promises";
import path from "node:path";

type Prediction = {
  match_number: string;
  date: string;
  stage: string;
  group: string;
  home_team: string;
  away_team: string;
  score: string;
  predicted_winner: string;
  confidence: string;
  safe_score: string;
  upside_score: string;
  model_favourite_prob: number | null;
  prob_home_win: number | null;
  prob_draw: number | null;
  prob_away_win: number | null;
  actual_available?: boolean;
  actual_home_score?: number | "";
  actual_away_score?: number | "";
  actual_score?: string;
  actual_outcome?: string;
  actual_winner?: string;
  pre_match_score?: string;
  pre_match_predicted_winner?: string;
  pre_match_confidence?: string;
  pre_match_model_favourite_prob?: number | "";
  pre_match_source?: string;
  prediction_exact?: boolean | "";
  prediction_outcome_correct?: boolean | "";
};

type Change = {
  match_number: string;
  stage_old?: string;
  stage: string;
  home_team_old: string;
  away_team_old: string;
  score_old: string;
  predicted_winner_old: string;
  home_team_new: string;
  away_team_new: string;
  score_new: string;
  predicted_winner_new: string;
};

type Champion = {
  rank: string;
  team: string;
  champion_prob: number;
  advance_final_prob: number;
  advance_sf_prob: number;
};

type GroupStanding = {
  group: string;
  team: string;
  points: string;
  gd: string;
  rank: string;
  qualified_by_pick: string;
};

type TopScorer = {
  rank: string;
  player: string;
  team: string;
  position: string;
  expected_goals: number;
  expected_scorito_points: number;
};

type SourceStatus = {
  name: string;
  status: string;
  rows?: number;
  last_modified?: string;
  note?: string;
};

type DashboardData = {
  metadata: {
    generated_at: string;
    source_run: string;
    model_accuracy?: number;
    exact_score_accuracy?: number;
    row_count?: number;
    latest_match_date?: string;
    features?: number;
    lineup_features_enabled?: boolean;
    stat_features_enabled?: boolean;
    card_features_enabled?: boolean;
  };
  downloads: {
    compact_excel?: string;
    probabilities_excel?: string;
    full_excel?: string;
    compact_csv?: string;
  };
  predictions: Prediction[];
  changes: Change[];
  champions: Champion[];
  group_standings: GroupStanding[];
  top_scorers: TopScorer[];
  group_top_scorers: TopScorer[];
  sources: SourceStatus[];
};

async function loadDashboard(): Promise<DashboardData | null> {
  try {
    const file = path.join(process.cwd(), "public", "data", "dashboard.json");
    return JSON.parse(await readFile(file, "utf8")) as DashboardData;
  } catch {
    return null;
  }
}

function pct(value?: number | null, digits = 1) {
  if (value === undefined || value === null || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(digits)}%`;
}

function confidenceClass(confidence: string) {
  const value = confidence.toLowerCase();
  if (value.includes("high")) return "green";
  if (value.includes("medium")) return "orange";
  return "";
}

function isPlayed(row: Prediction) {
  return row.actual_available === true || Boolean(row.actual_score);
}

function resultClass(row: Prediction) {
  if (row.prediction_exact === true) return "green";
  if (row.prediction_outcome_correct === true) return "orange";
  if (isPlayed(row)) return "red";
  return "";
}

function resultLabel(row: Prediction) {
  if (row.prediction_exact === true) return "Exact";
  if (row.prediction_outcome_correct === true) return "Toto goed";
  if (isPlayed(row)) return "Mis";
  return "-";
}

const STAGE_ORDER = [
  "Group Stage",
  "Round of 32",
  "Round of 16",
  "Quarterfinals",
  "Semifinals",
  "Third Place Playoff",
  "Final",
];

const STAGE_LABELS: Record<string, string> = {
  "Group Stage": "Poulefase",
  "Round of 32": "1/16 finale",
  "Round of 16": "Achtste finale",
  Quarterfinals: "Kwartfinale",
  Semifinals: "Halve finale",
  "Third Place Playoff": "Troostfinale",
  Final: "Finale",
};

function stageLabel(stage: string) {
  return STAGE_LABELS[stage] || stage || "-";
}

function dateKey(value?: string) {
  const match = String(value || "").match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!match) return null;
  return Number(`${match[1]}${match[2]}${match[3]}`);
}

function stageStartKeys(rows: Prediction[]) {
  const starts = new Map<string, number>();
  for (const row of rows) {
    const key = dateKey(row.date);
    if (key === null) continue;
    const current = starts.get(row.stage);
    if (current === undefined || key < current) {
      starts.set(row.stage, key);
    }
  }
  return starts;
}

function isStageLocked(stage: string, starts: Map<string, number>, snapshotKey: number | null) {
  const start = starts.get(stage);
  return snapshotKey !== null && start !== undefined && snapshotKey >= start;
}

function firstOpenStage(rows: Prediction[], starts: Map<string, number>, snapshotKey: number | null) {
  const stageSet = new Set(rows.map((row) => row.stage));
  return STAGE_ORDER.find((stage) => stageSet.has(stage) && !isStageLocked(stage, starts, snapshotKey)) || "";
}

function grouped<T extends { group: string }>(items: T[]) {
  return items.reduce<Record<string, T[]>>((acc, item) => {
    const key = item.group || "-";
    acc[key] ??= [];
    acc[key].push(item);
    return acc;
  }, {});
}

export default async function Home() {
  const data = await loadDashboard();

  if (!data) {
    return (
      <main className="content">
        <h1>Voetbal Prediction</h1>
        <p>Er staat nog geen dashboarddata klaar. Run eerst:</p>
        <pre>npm run publish:data</pre>
      </main>
    );
  }

  const playedMatches = data.predictions.filter(isPlayed);
  const upcomingPredictions = data.predictions.filter((row) => !isPlayed(row));
  const starts = stageStartKeys(data.predictions);
  const snapshotKey = dateKey(data.metadata.generated_at);
  const lockedUpcoming = upcomingPredictions.filter((row) => isStageLocked(row.stage, starts, snapshotKey));
  const fillablePredictions = upcomingPredictions.filter((row) => !isStageLocked(row.stage, starts, snapshotKey));
  const currentFillStage = firstOpenStage(fillablePredictions, starts, snapshotKey);
  const fillAdviceRows = currentFillStage
    ? fillablePredictions.filter((row) => row.stage === currentFillStage)
    : [];
  const knockouts = upcomingPredictions.filter((row) => row.stage !== "Group Stage");
  const groups = grouped(data.group_standings);
  const actionableChanges = data.changes.filter((row) =>
    !isStageLocked(row.stage || row.stage_old || "", starts, snapshotKey)
  );
  const scoreChanges = actionableChanges.filter((row) => row.score_old !== row.score_new).length;

  return (
    <main className="page">
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="#invullen">
            <strong>Voetbal Prediction</strong>
            <span>Scorito dashboard voor mobiel</span>
          </a>
          <nav className="nav" aria-label="Dashboard">
            <a href="#wijzigingen">Wijzigingen</a>
            <a href="#gespeeld">Gespeeld</a>
            <a href="#invullen">Invullen</a>
            <a href="#groepen">Groepen</a>
            <a href="#knockout">Knockout</a>
            <a href="#kampioen">Kampioen</a>
            <a href="#model">Model</a>
          </nav>
          {data.downloads.probabilities_excel ? (
            <a className="download-link" href={data.downloads.probabilities_excel}>
              Excel
            </a>
          ) : null}
        </div>
      </header>

      <div className="content">
        <section className="summary-grid" aria-label="Samenvatting">
          <div className="metric">
            <div className="metric-label">Laatste versie</div>
            <div className="metric-value">{data.metadata.generated_at}</div>
            <div className="metric-note">{data.metadata.source_run}</div>
          </div>
          <div className="metric">
            <div className="metric-label">Scorewijzigingen</div>
            <div className="metric-value">{actionableChanges.length}</div>
            <div className="metric-note">
              {scoreChanges} dezelfde wedstrijden - {playedMatches.length} gespeeld
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">Model</div>
            <div className="metric-value">{pct(data.metadata.model_accuracy)}</div>
            <div className="metric-note">holdout W/D/L accuracy</div>
          </div>
          <div className="metric">
            <div className="metric-label">Exact score</div>
            <div className="metric-value">{pct(data.metadata.exact_score_accuracy)}</div>
            <div className="metric-note">historische testset</div>
          </div>
        </section>

        <section id="wijzigingen" className="section">
          <div className="section-header">
            <div>
              <h2 className="section-title">Laatste Scorewijzigingen</h2>
              <p className="section-subtitle">Alleen dezelfde wedstrijden waarvan de invulscore veranderde.</p>
            </div>
          </div>
          <div className="stack">
            {actionableChanges.length ? (
              actionableChanges.map((row) => {
                const newMatch = `${row.home_team_new} - ${row.away_team_new}`;

                return (
                  <div
                    className="change-card"
                    key={`${row.match_number}-${row.home_team_new}-${row.away_team_new}`}
                  >
                    <div className="change-meta">
                      <span className="pill">#{row.match_number}</span>
                      <span className="pill">{row.stage || row.stage_old || "-"}</span>
                    </div>

                    <div className="change-match">
                      <strong>{newMatch}</strong>
                    </div>

                    <div className="change-details">
                      <div className="change-detail">
                        <span className="detail-label">Score</span>
                        <span className="old-score">{row.score_old}</span>
                        <span className="change-arrow">naar</span>
                        <span className="score">{row.score_new}</span>
                      </div>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="change-card">Geen scorewijzigingen voor open invulrondes.</div>
            )}
          </div>
        </section>

        <section id="gespeeld" className="section">
          <div className="section-header">
            <div>
              <h2 className="section-title">Gespeelde Wedstrijden</h2>
              <p className="section-subtitle">Voorspelling naast de echte uitslag zodra results.csv is bijgewerkt.</p>
            </div>
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
                {playedMatches.length ? (
                  playedMatches.map((row) => (
                    <tr key={`played-${row.stage}-${row.match_number}`}>
                      <td className="mono">{row.match_number}</td>
                      <td>
                        <strong>{row.home_team}</strong> - {row.away_team}
                        <div className="metric-note">
                          {row.stage}
                          {row.group ? ` - Poule ${row.group}` : ""} - {row.date}
                        </div>
                      </td>
                      <td>
                        <span className="score">{row.pre_match_score || row.score}</span>
                        <div className="metric-note">{row.pre_match_predicted_winner || row.predicted_winner}</div>
                      </td>
                      <td>
                        <span className="score actual-score">{row.actual_score || "-"}</span>
                        <div className="metric-note">{row.actual_winner || "-"}</div>
                      </td>
                      <td>
                        <span className={`pill ${resultClass(row)}`}>{resultLabel(row)}</span>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5}>Nog geen WK-wedstrijden met uitslag in de dashboarddata.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section id="invullen" className="section">
          <div className="section-header">
            <div>
              <h2 className="section-title">Scorito Invullen</h2>
              <p className="section-subtitle">
                {currentFillStage
                  ? `${stageLabel(currentFillStage)} - alleen de ronde die nog niet vergrendeld is.`
                  : "Geen open invulronde in deze dashboarddata."}
              </p>
            </div>
          </div>
          {lockedUpcoming.length ? (
            <div className="inline-note">
              {lockedUpcoming.length} toekomstige wedstrijden zijn al vergrendeld voor invullen en blijven buiten deze lijst.
            </div>
          ) : null}
          <div className="table-shell">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Wedstrijd</th>
                  <th>Score</th>
                  <th>Winnaar</th>
                  <th>Kans</th>
                  <th>Conf.</th>
                </tr>
              </thead>
              <tbody>
                {fillAdviceRows.length ? (
                  fillAdviceRows.map((row) => (
                    <tr key={`${row.stage}-${row.match_number}`}>
                      <td className="mono">{row.match_number}</td>
                      <td>
                        <strong>{row.home_team}</strong> - {row.away_team}
                        <div className="metric-note">
                          {row.group ? `Poule ${row.group}` : stageLabel(row.stage)} - {row.date}
                        </div>
                      </td>
                      <td>
                        <span className="score">{row.score}</span>
                      </td>
                      <td>{row.predicted_winner}</td>
                      <td className="mono">{pct(row.model_favourite_prob)}</td>
                      <td>
                        <span className={`pill ${confidenceClass(row.confidence)}`}>
                          {row.confidence || "-"}
                        </span>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={6}>Geen open wedstrijden voor de volgende invulronde.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section id="groepen" className="section">
          <div className="section-header">
            <div>
              <h2 className="section-title">Groepsstanden</h2>
              <p className="section-subtitle">Verwachte eindstand op basis van de invulscores.</p>
            </div>
          </div>
          <div className="cards-grid">
            {Object.entries(groups).map(([group, rows]) => (
              <div className="team-card" key={group}>
                <div className="team-card-top">
                  <strong>Poule {group}</strong>
                  <span className="pill">{rows.length} teams</span>
                </div>
                <table>
                  <tbody>
                    {rows
                      .sort((a, b) => Number(a.rank) - Number(b.rank))
                      .map((row) => (
                        <tr key={`${group}-${row.team}`}>
                          <td className="mono">{row.rank}</td>
                          <td>{row.team}</td>
                          <td className="mono">{row.points}p</td>
                          <td className="mono">GD {row.gd}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            ))}
          </div>
        </section>

        <section id="knockout" className="section">
          <div className="section-header">
            <div>
              <h2 className="section-title">Knockout</h2>
              <p className="section-subtitle">Vanaf R32 opnieuw invullen zodra Scorito dat vraagt.</p>
            </div>
          </div>
          <div className="table-shell">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Ronde</th>
                  <th>Wedstrijd</th>
                  <th>Score</th>
                  <th>Winnaar</th>
                </tr>
              </thead>
              <tbody>
                {knockouts.map((row) => (
                  <tr key={`${row.stage}-${row.match_number}`}>
                    <td className="mono">{row.match_number}</td>
                    <td>{row.stage}</td>
                    <td>
                      <strong>{row.home_team}</strong> - {row.away_team}
                    </td>
                    <td>
                      <span className="score">{row.score}</span>
                    </td>
                    <td>{row.predicted_winner}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section id="kampioen" className="section">
          <div className="section-header">
            <div>
              <h2 className="section-title">Kampioenskansen</h2>
              <p className="section-subtitle">Monte Carlo-output uit de laatste run.</p>
            </div>
          </div>
          <div className="cards-grid">
            {data.champions.slice(0, 12).map((row) => (
              <div className="team-card" key={row.team}>
                <div className="team-card-top">
                  <span className="pill">#{row.rank}</span>
                  <span className="prob">{pct(row.champion_prob)}</span>
                </div>
                <div className="team-name">{row.team}</div>
                <div className="metric-note">
                  Finale {pct(row.advance_final_prob)} - Halve finale {pct(row.advance_sf_prob)}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section id="topscorers" className="section">
          <div className="section-header">
            <div>
              <h2 className="section-title">Topscorers</h2>
              <p className="section-subtitle">Scorito-punten wegen positie zwaarder mee.</p>
            </div>
          </div>
          <div className="table-shell">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Speler</th>
                  <th>Team</th>
                  <th>Pos.</th>
                  <th>xG</th>
                  <th>Scorito</th>
                </tr>
              </thead>
              <tbody>
                {data.top_scorers.slice(0, 12).map((row) => (
                  <tr key={`${row.rank}-${row.player}`}>
                    <td className="mono">{row.rank}</td>
                    <td>
                      <strong>{row.player}</strong>
                    </td>
                    <td>{row.team}</td>
                    <td>{row.position}</td>
                    <td className="mono">{row.expected_goals?.toFixed(2)}</td>
                    <td className="mono">{row.expected_scorito_points?.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section id="model" className="section">
          <div className="section-header">
            <div>
              <h2 className="section-title">Modelinformatie</h2>
              <p className="section-subtitle">Compacte controle zonder ruwe bronlijst.</p>
            </div>
          </div>
          <div className="source-grid">
            <div className="source-card">
              <strong>Data</strong>
              <div className="metric-note">
                {data.metadata.row_count?.toLocaleString("nl-NL") ?? "-"} uitslagen - laatste match{" "}
                {data.metadata.latest_match_date || "-"}
              </div>
            </div>
            <div className="source-card">
              <strong>Features</strong>
              <div className="metric-note">{data.metadata.features ?? "-"} modelvariabelen</div>
            </div>
            <div className="source-card">
              <strong>Soccerbase</strong>
              <div className="metric-note">
                stats {data.metadata.stat_features_enabled ? "aan" : "uit"} - cards{" "}
                {data.metadata.card_features_enabled ? "aan" : "uit"} - lineups{" "}
                {data.metadata.lineup_features_enabled ? "aan" : "uit"}
              </div>
            </div>
            <div className="source-card">
              <strong>Downloads</strong>
              <div className="metric-note">
                <a href={data.downloads.probabilities_excel || "#"}>kansen Excel</a> -{" "}
                <a href={data.downloads.full_excel || "#"}>volledige Excel</a> -{" "}
                <a href={data.downloads.compact_csv || "#"}>CSV</a>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
