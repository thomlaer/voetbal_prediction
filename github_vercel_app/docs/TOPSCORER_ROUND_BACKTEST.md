# Topscorer round xG backtest

## Why this change

The dashboard used the rounded pool score as a fallback when the final knockout
fixture xG was attached during publishing. That made a `1-2` prediction count
as exactly two expected team goals, while a `0-1` prediction counted as exactly
one. Player rankings could therefore disagree with the xG shown beside the same
match.

The publisher now recalculates every round-specific player projection from the
final published fixture xG. Completed matches use their actual goals. A rounded
score is used only when neither actual goals nor xG is available.

Live-data sanity check after recalculation:

- Kylian Mbappé: Round-of-16 rank 10 -> 3; expected goals 0.38 -> 0.66.
- Romelu Lukaku: Round-of-16 rank 4 -> 7; expected goals 0.60 -> 0.44.

## Method

Run:

```powershell
python backtest_topscorer_rounds.py `
  --data-root C:\path\to\voetbal_prediction `
  --top-n 10
```

The backtest covers the five rounds in the 2018 and 2022 32-team format. It:

- uses the historical XGBoost fixture xG generated for each tournament;
- uses pre-tournament squads, ratings, caps and international goals;
- uses tournament goals and starts only from rounds already completed;
- compares `fixture_xg` with the previous `rounded_score` fallback;
- evaluates the displayed top 10 against actual non-own goals in that round.

The output contains `summary.csv`, `picks.csv` and `actual_leaders.csv`.

## 2018 results

| Round | Actual team goals | Top-10 goals: xG | Top-10 goals: score | Points: xG | Points: score | Team-goal MAE: xG | Team-goal MAE: score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Group stage | 122 | 9 | 9 | 72 | 72 | 1.347 | 2.219 |
| Round of 16 | 24 | 5 | 3 | 120 | 72 | 0.728 | 1.125 |
| Quarterfinals | 11 | 1 | 1 | 32 | 32 | 0.794 | 0.875 |
| Semifinals | 4 | 2 | 0 | 80 | 0 | 0.820 | 1.000 |
| Final/third place | 8 | 5 | 2 | 240 | 96 | 1.315 | 2.000 |
| **Total top-10 result** |  | **22** | **15** | **544** | **272** |  |  |

Actual leading scorers by round:

- Group: Harry Kane 5; Cristiano Ronaldo 4; Romelu Lukaku 4; Denis Cheryshev 3; Diego Costa 3.
- Round of 16: Edinson Cavani 2; Kylian Mbappé 2; Benjamin Pavard, Gabriel Mercado and Jan Vertonghen scored 1.
- Quarterfinals: Domagoj Vida, Harry Maguire, Mário Fernandes, Raphaël Varane and Dele Alli scored 1.
- Semifinals: Kieran Trippier, Samuel Umtiti, Ivan Perišić and Mario Mandžukić scored 1.
- Final/third place: Thomas Meunier, Paul Pogba, Antoine Griezmann, Eden Hazard and Ivan Perišić scored 1.

## 2022 results

| Round | Actual team goals | Top-10 goals: xG | Top-10 goals: score | Points: xG | Points: score | Team-goal MAE: xG | Team-goal MAE: score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Group stage | 120 | 7 | 9 | 56 | 72 | 1.515 | 2.281 |
| Round of 16 | 28 | 5 | 6 | 120 | 144 | 0.935 | 1.313 |
| Quarterfinals | 10 | 3 | 3 | 96 | 96 | 0.638 | 1.000 |
| Semifinals | 5 | 3 | 3 | 120 | 120 | 1.008 | 0.750 |
| Final/third place | 9 | 5 | 3 | 240 | 144 | 1.126 | 1.500 |
| **Total top-10 result** |  | **23** | **24** | **632** | **576** |  |  |

Actual leading scorers by round:

- Group: Cody Gakpo, Enner Valencia, Kylian Mbappé, Marcus Rashford and Álvaro Morata, all 3.
- Round of 16: Gonçalo Ramos 3; Kylian Mbappé 2; the remaining leaders scored 1.
- Quarterfinals: Wout Weghorst 2; the remaining leaders scored 1.
- Semifinals: Julián Álvarez 2; Théo Hernandez, Lionel Messi and Randal Kolo Muani scored 1.
- Final/third place: Kylian Mbappé 3; Lionel Messi 2; the remaining leaders scored 1.

## Assessment

Fixture xG has lower team-goal MAE in 9 of 10 tested rounds. The only exception
is the 2022 semifinals. It materially improves 2018 top-10 selections. In 2022
the rounded-score baseline catches one more goal in total, but fixture xG earns
more weighted points because it performs better in the final/third-place round.

This supports the xG fix, but two tournaments are not enough to claim that every
individual round will improve. The main benefit is that rankings are continuous,
less threshold-sensitive and consistent with the match xG shown to the user.
