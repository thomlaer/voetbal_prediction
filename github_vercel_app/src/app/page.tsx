import { readFile } from "node:fs/promises";
import path from "node:path";
import { EspnLivePanel } from "./EspnLivePanel";
import { PlayedMatchesPanel, type PlayedMatchRow } from "./PlayedMatchesPanel";
import { RebuildControl } from "./RebuildControl";

type Prediction = {
  match_number: string;
  date: string;
  stage: string;
  group: string;
  home_team: string;
  away_team: string;
  score: string;
  model_score?: string;
  model_predicted_winner?: string;
  filled_score?: string;
  filled_predicted_winner?: string;
  new_model_score?: string;
  new_model_predicted_winner?: string;
  round_locked?: boolean;
  round_status?: string;
  score_source?: string;
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
  actual_source?: string;
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
  advance_qf_prob?: number;
  advance_r16_prob?: number;
};

type GroupStanding = {
  group: string;
  team: string;
  points: string;
  gd: string;
  rank: string;
  qualified_by_pick: string;
  group_matches_total?: number | string;
  group_matches_actual?: number | string;
  group_matches_online_verified?: number | string;
  group_complete?: boolean | string;
  group_online_verified?: boolean | string;
  standing_source?: string;
  rank_confirmed?: boolean | string;
  qualified_confirmed?: boolean | string;
  advance_r16_prob?: number | "";
  advance_qf_prob?: number | "";
  champion_prob?: number | "";
};

type TopScorer = {
  rank: string;
  player: string;
  team: string;
  position: string;
  expected_goals: number;
  expected_scorito_points: number;
};

type RoundTopScorer = {
  round_rank: string;
  stage: string;
  stage_label: string;
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
  round_top_scorers?: RoundTopScorer[];
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

function pct(value?: number | string | null, digits = 1) {
  if (value === undefined || value === null || value === "") return "-";
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return "-";
  return `${(numeric * 100).toFixed(digits)}%`;
}

function truthy(value?: boolean | string) {
  return value === true || value === "true";
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
  if (row.prediction_outcome_correct === true) return "Winnaar goed";
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

const TOPSCORER_STAGE_ORDER = [
  "Group Stage",
  "Round of 32",
  "Round of 16",
  "Quarterfinals",
  "Semifinals",
  "Final/Third",
];

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

function stageRows(rows: Prediction[]) {
  return STAGE_ORDER.map((stage) => ({
    stage,
    rows: rows.filter((row) => row.stage === stage),
  })).filter((section) => section.rows.length);
}

function isRoundManuallyLocked(rows: Prediction[]) {
  return rows.length > 0 && rows.every((row) => row.round_locked || isPlayed(row));
}

function isStageClosed(stage: string, rows: Prediction[], starts: Map<string, number>, snapshotKey: number | null) {
  return isStageLocked(stage, starts, snapshotKey) || isRoundManuallyLocked(rows);
}

function firstOpenStage(sections: { stage: string; rows: Prediction[] }[], starts: Map<string, number>, snapshotKey: number | null) {
  return sections.find((section) => !isStageClosed(section.stage, section.rows, starts, snapshotKey))?.stage || "";
}

function stageShouldOpen(stage: string, currentFillStage: string) {
  if (!currentFillStage) return false;
  const stageIndex = STAGE_ORDER.indexOf(stage);
  const currentIndex = STAGE_ORDER.indexOf(currentFillStage);
  if (stageIndex === -1 || currentIndex === -1) return true;
  return stageIndex >= currentIndex;
}

function stageStatus(stage: string, rows: Prediction[], starts: Map<string, number>, snapshotKey: number | null, currentFillStage: string) {
  if (rows.length && rows.every(isPlayed)) return { label: "Gespeeld", className: "green" };
  if (isStageClosed(stage, rows, starts, snapshotKey)) return { label: "Vergrendeld", className: "orange" };
  if (stage === currentFillStage) return { label: "Nu invullen", className: "green" };
  return { label: "Later", className: "" };
}

function displayScore(row: Prediction) {
  return row.filled_score || row.pre_match_score || row.score;
}

function winnerFromScore(row: Prediction) {
  const match = String(displayScore(row) || "").match(/^(\d+)-(\d+)$/);
  if (!match) return "";
  const homeGoals = Number(match[1]);
  const awayGoals = Number(match[2]);
  if (homeGoals > awayGoals) return row.home_team;
  if (awayGoals > homeGoals) return row.away_team;
  return "Draw";
}

function displayWinner(row: Prediction) {
  return (
    winnerFromScore(row) ||
    row.filled_predicted_winner ||
    row.pre_match_predicted_winner ||
    row.predicted_winner ||
    row.model_predicted_winner
  );
}

function matchupLabel(row: Prediction) {
  const winner = displayWinner(row);
  const homeIsWinner = winner === row.home_team;
  const awayIsWinner = winner === row.away_team;

  return (
    <>
      {homeIsWinner ? <strong>{row.home_team}</strong> : row.home_team} -{" "}
      {awayIsWinner ? <strong>{row.away_team}</strong> : row.away_team}
    </>
  );
}

function hasOutcomeProbabilities(row: Prediction) {
  return [row.prob_home_win, row.prob_draw, row.prob_away_win].some(
    (value) => value !== undefined && value !== null,
  );
}

function probabilityLine(row: Prediction) {
  return `Thuis ${pct(row.prob_home_win)} · Gelijk ${pct(row.prob_draw)} · Uit ${pct(row.prob_away_win)}`;
}

function grouped<T extends { group: string }>(items: T[]) {
  return items.reduce<Record<string, T[]>>((acc, item) => {
    const key = item.group || "-";
    acc[key] ??= [];
    acc[key].push(item);
    return acc;
  }, {});
}

function scorerRoundSections(rows: RoundTopScorer[]) {
  return TOPSCORER_STAGE_ORDER.map((stage) => ({
    stage,
    label: rows.find((row) => row.stage === stage)?.stage_label || stageLabel(stage),
    rows: rows.filter((row) => row.stage === stage),
  })).filter((section) => section.rows.length);
}

function groupStatus(rows: GroupStanding[]) {
  const total = Number(rows[0]?.group_matches_total || 0);
  const actual = Number(rows[0]?.group_matches_actual || 0);
  const online = Number(rows[0]?.group_matches_online_verified || 0);
  const complete = rows.some((row) => truthy(row.group_complete));
  const onlineVerified = rows.some((row) => truthy(row.group_online_verified));
  const suffix = total > 0 ? ` · ${online}/${total} online` : "";
  if (onlineVerified) {
    return { label: "Online bevestigd", note: `${actual}/${total} echte uitslagen${suffix}`, className: "green" };
  }
  if (complete) {
    return { label: "Bevestigd", note: `${actual}/${total} echte uitslagen${suffix}`, className: "green" };
  }
  if (actual > 0 && total > 0) {
    return { label: "Live check", note: `${actual}/${total} echte uitslagen${suffix}`, className: "orange" };
  }
  return { label: "Projectie", note: `nog geen volledige echte poule${suffix}`, className: "" };
}

function playedPanelRow(row: Prediction): PlayedMatchRow {
  const predictedScore = row.filled_score || row.pre_match_score || row.score || "";
  const actualScore = row.actual_score || "";

  return {
    match_number: String(row.match_number),
    date: row.date,
    stage: row.stage,
    group: row.group,
    home_team: row.home_team,
    away_team: row.away_team,
    predicted_score: predictedScore,
    predicted_winner:
      row.filled_predicted_winner ||
      row.pre_match_predicted_winner ||
      winnerFromScore(row) ||
      row.predicted_winner ||
      "",
    actual_score: actualScore,
    actual_winner: row.actual_winner || "",
    actual_source: row.actual_source || "dashboard",
    backup_actual_score: actualScore,
    backup_mismatch: false,
    prediction_exact: row.prediction_exact === true,
    prediction_outcome_correct: row.prediction_outcome_correct === true,
    result_class: resultClass(row),
    result_label: resultLabel(row),
  };
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
  const playedPanelRows = playedMatches.map(playedPanelRow);
  const starts = stageStartKeys(data.predictions);
  const snapshotKey = dateKey(data.metadata.generated_at);
  const rounds = stageRows(data.predictions);
  const stageMap = new Map(rounds.map((section) => [section.stage, section.rows]));
  const currentFillStage = firstOpenStage(rounds, starts, snapshotKey);
  const groups = grouped(data.group_standings);
  const lockStages = rounds.map(({ stage, rows }) => ({
    stage,
    label: stageLabel(stage),
    rows: rows.length,
    lockedRows: rows.filter((row) => row.round_locked).length,
    playedRows: rows.filter(isPlayed).length,
  }));
  const topScorerRounds = scorerRoundSections(data.round_top_scorers || []);
  const actionableChanges = data.changes.filter((row) => {
    const stage = row.stage || row.stage_old || "";
    return !isStageClosed(stage, stageMap.get(stage) || [], starts, snapshotKey);
  });
  const scoreChanges = actionableChanges.filter((row) => row.score_old !== row.score_new).length;

  return (
    <main className="page">
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="#rondes">
            <strong>Voetbal Prediction</strong>
            <span>Scorito dashboard voor mobiel</span>
          </a>
          <nav className="nav" aria-label="Dashboard">
            <a href="#wijzigingen">Wijzigingen</a>
            <a href="#gespeeld">Gespeeld</a>
            <a href="#rondes">Rondes</a>
            <a href="#groepen">Groepen</a>
            <a href="#kampioen">Kampioen</a>
            <a href="#update">Beheer</a>
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
          <PlayedMatchesPanel initialRows={playedPanelRows} />
        </section>

        <section id="rondes" className="section">
          <div className="section-header">
            <div>
              <h2 className="section-title">Rondes</h2>
              <p className="section-subtitle">
                Gesloten rondes tonen de ingevulde score. Geel is alleen de nieuwe modelscore als die later afwijkt.
              </p>
            </div>
          </div>
          <div className="round-stack">
            {rounds.map(({ stage, rows }) => {
              const status = stageStatus(stage, rows, starts, snapshotKey, currentFillStage);
              const changedRows = rows.filter((row) => row.new_model_score).length;

              return (
                <details className="round-card" key={stage} open={stageShouldOpen(stage, currentFillStage)}>
                  <summary className="round-header collapse-summary">
                    <div>
                      <h3>{stageLabel(stage)}</h3>
                      <div className="metric-note">
                        {rows.length} wedstrijden
                        {changedRows ? ` - ${changedRows} nieuwe modelscore${changedRows === 1 ? "" : "s"}` : ""}
                      </div>
                    </div>
                    <span className="collapse-actions">
                      <span className={`pill ${status.className}`}>{status.label}</span>
                      <span className="collapse-caret" aria-hidden="true" />
                    </span>
                  </summary>
                  <div className="table-shell compact-table">
                    <table>
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Wedstrijd</th>
                          <th>Score</th>
                          <th>Winnaar</th>
                          <th>Nieuw</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((row) => (
                          <tr key={`round-${stage}-${row.match_number}`}>
                            <td className="mono">{row.match_number}</td>
                            <td>
                              {matchupLabel(row)}
                              <div className="metric-note">
                                {row.group ? `Poule ${row.group}` : stageLabel(row.stage)} - {row.date}
                                {isPlayed(row) ? ` - uitslag ${row.actual_score}` : ""}
                              </div>
                              {!isPlayed(row) && hasOutcomeProbabilities(row) ? (
                                <div className="prob-note">{probabilityLine(row)}</div>
                              ) : null}
                            </td>
                            <td>
                              <span className="score">{displayScore(row)}</span>
                            </td>
                            <td>{displayWinner(row)}</td>
                            <td>
                              {row.new_model_score ? (
                                <span className="new-score" title="Nieuwe score uit de laatste modelrun">
                                  {row.new_model_score}
                                </span>
                              ) : (
                                <span className="metric-note">-</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
              );
            })}
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
            {Object.entries(groups).map(([group, rows]) => {
              const status = groupStatus(rows);

              return (
                <div className={`team-card ${status.className === "green" ? "confirmed-card" : ""}`} key={group}>
                  <div className="team-card-top">
                    <strong>Poule {group}</strong>
                    <span className={`pill ${status.className}`}>{status.label}</span>
                  </div>
                  <div className="metric-note">{status.note}</div>
                  <table>
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Team</th>
                        <th>Pts</th>
                        <th>GD</th>
                        <th>Door</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows
                        .sort((a, b) => Number(a.rank) - Number(b.rank))
                        .map((row) => {
                          const confirmedQualified = truthy(row.qualified_confirmed);

                          return (
                            <tr
                              className={confirmedQualified ? "confirmed-qualified-row" : ""}
                              key={`${group}-${row.team}`}
                            >
                              <td className="mono">{row.rank}</td>
                              <td>
                                {confirmedQualified ? <strong>{row.team}</strong> : row.team}
                                {truthy(row.rank_confirmed) ? <div className="metric-note">plek vast</div> : null}
                              </td>
                              <td className="mono">{row.points}p</td>
                              <td className="mono">GD {row.gd}</td>
                              <td className="mono">
                                {confirmedQualified ? <span className="pill green">zeker</span> : pct(row.advance_r16_prob)}
                              </td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              );
            })}
          </div>
          <EspnLivePanel />
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
              <p className="section-subtitle">Totaal bovenaan. Daarna per ronde, gewogen op xG en Scorito-punten.</p>
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
          {topScorerRounds.length ? (
            <div className="round-scorer-grid">
              {topScorerRounds.map((section) => (
                <div className="team-card" key={section.stage}>
                  <div className="team-card-top">
                    <strong>{section.label}</strong>
                    <span className="pill">top {section.rows.length}</span>
                  </div>
                  <table>
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Speler</th>
                        <th>Team</th>
                        <th>xG</th>
                        <th>Pts</th>
                      </tr>
                    </thead>
                    <tbody>
                      {section.rows.slice(0, 6).map((row) => (
                        <tr key={`${section.stage}-${row.round_rank}-${row.player}`}>
                          <td className="mono">{row.round_rank}</td>
                          <td>
                            <strong>{row.player}</strong>
                            <div className="metric-note">{row.position || "-"}</div>
                          </td>
                          <td>{row.team}</td>
                          <td className="mono">{Number(row.expected_goals || 0).toFixed(2)}</td>
                          <td className="mono">{Number(row.expected_scorito_points || 0).toFixed(1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
          ) : null}
        </section>

        <section id="update" className="section admin-section">
          <details className="section-details admin-details">
            <summary className="section-header collapse-summary">
              <div>
                <h2 className="section-title">Beheer</h2>
                <p className="section-subtitle">Update starten of een invulronde vastzetten.</p>
              </div>
              <span className="collapse-actions">
                <span className="pill">code nodig</span>
                <span className="collapse-caret" aria-hidden="true" />
              </span>
            </summary>
            <RebuildControl defaultLockStage={currentFillStage || lockStages[0]?.stage || ""} lockStages={lockStages} />
          </details>
        </section>

      </div>
    </main>
  );
}
