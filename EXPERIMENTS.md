# Experimenten

## Split

De huidige evaluatie gebruikt een tijdsplit, geen random 80/20 split:

- trainingsdata: wedstrijden vanaf `1993-01-01` t/m `2022-12-31`
- testdata: wedstrijden vanaf `2023-01-01` t/m `2026-03-31`

Dit is bewust gedaan om leakage te vermijden. Een random 80/20 split zou oude en nieuwe wedstrijden door elkaar gooien en daardoor te optimistisch kunnen zijn.

## Baseline zonder player features

Command:

```powershell
.\.venv\Scripts\python.exe train_xgboost_worldcup.py --skip-download --no-player-features --output-dir outputs_baseline --model-dir models_baseline
```

Resultaat:

- features: 66
- test rows: 3.452
- accuracy: `0.6098`
- balanced accuracy: `0.5152`
- log loss: `0.8572`
- brier score: `0.5039`

## Met Transfermarkt player-form en lineup-signalen

Command:

```powershell
.\.venv\Scripts\python.exe train_xgboost_worldcup.py --skip-download
```

Nieuwe featurelaag:

- laatste 730 dagen voor de wedstrijd
- recency half-life: 365 dagen
- player minutes, goals, assists, cards
- lineup starts en named substitutes uit `game_lineups.csv`
- competition quality weights, o.a. Champions League en top-5 competities zwaarder
- positiefeatures voor attack, midfield, defender en goalkeeper

Resultaat:

- features: 193
- player-form beschikbaar voor 11.410 van 49.256 totale wedstrijden
- test rows: 3.452
- accuracy: `0.6098`
- balanced accuracy: `0.5152`
- log loss: `0.8554`
- brier score: `0.5028`

Conclusie:

De player-form laag verbetert de kansinschatting licht, vooral `log_loss` en `brier_score`. Pure accuracy blijft gelijk. Dit is logisch: de extra spelerinformatie verschuift vooral waarschijnlijkheden, niet altijd de hoogste klasse.

## Vergelijking met handmatige voetbalpoule

Bronbestand:

`handmatige_voetbalpoule_voorspellingen_analyse.xlsx`

Voor een eerlijke vergelijking zijn drie losse pre-tournament runs gebruikt:

- UEFA Euro 2021: trainen tot `2021-06-10`, voorspellen vanaf `2021-06-11`
- FIFA World Cup 2022: trainen tot `2022-11-19`, voorspellen vanaf `2022-11-20`
- UEFA Euro 2024: trainen tot `2024-06-13`, voorspellen vanaf `2024-06-14`

Het model is dus niet achteraf op het toernooi zelf getraind.

Modelresultaat op dezelfde 166 wedstrijden:

- correct outcomes: 97 van 166
- outcome accuracy: `58.43%`

Per toernooi:

- UEFA Euro 2021: 32 van 51, `62.75%`
- FIFA World Cup 2022: 37 van 64, `57.81%`
- UEFA Euro 2024: 28 van 51, `54.90%`

Handmatige voorspellingen overall:

- Enzo: `54.17%`, 48 voorspellingen
- Ruben: `52.68%`, 112 voorspellingen
- Berend: `52.41%`, 166 voorspellingen
- Thomas: `51.55%`, 161 voorspellingen
- Joep: `51.53%`, 163 voorspellingen
- Meijer: `50.92%`, 163 voorspellingen
- Gijs: `50.00%`, 166 voorspellingen
- Didier: `50.00%`, 32 voorspellingen
- Bram: `49.02%`, 51 voorspellingen
- Siep: `46.58%`, 161 voorspellingen
- Daan: `45.40%`, 163 voorspellingen

Conclusie:

Op deze 166 wedstrijden is het model beter dan de beste handmatige overall score op juiste uitkomst. Het model voorspelt geen exacte scores, dus exacte-scorepercentages zijn hier niet direct vergelijkbaar.

## Datadekking huidige lokale model

Lokale data op dit moment:

- `data/results.csv`: 49.256 wedstrijden met score, laatste complete uitslag `2026-03-31`
- `data/results.csv`: bevat ook WK 2026 fixtures met lege scores; die worden genegeerd tijdens training
- `fifa_ranking-2026-04-01.csv`: laatste rankingdatum `2026-04-01`
- Transfermarkt `games.csv`, `appearances.csv` en `game_lineups.csv`: data t/m `2026-05-07`
- Transfermarkt `player_valuations.csv`: data t/m `2026-02-27`

De finale modelfile in `models/worldcup_xgboost_model.joblib` is getraind op alle complete wedstrijduitslagen vanaf `1993-01-01` t/m `2026-03-31`.

## Euro 2024 Door Huidig Model

Als Euro 2024 achteraf door het huidige finale model wordt gehaald:

- matches: 51
- correct outcomes: 28
- outcome accuracy: `54.90%`

Dit is niet hetzelfde als een eerlijke pre-tournament voorspelling, omdat het finale model inmiddels ook latere data kent. In deze specifieke check kwam de outcome-score wel gelijk uit aan de pre-tournament Euro 2024 run.

Euro 2024 handmatige vergelijking:

- Ruben: 32 van 51, `62.75%`
- Model: 28 van 51, `54.90%`
- Gijs: 27 van 51, `52.94%`
- Thomas: 24 van 49, `48.98%`

Conclusie:

Op alleen Euro 2024 zou het model op juiste uitkomst niet hebben gewonnen; Ruben was beter. Het model zou wel boven Thomas zijn geëindigd op outcome-percentage. Exacte scorepunten zijn nog niet vergelijkbaar, omdat het model nog geen scoreline-model heeft.

## Scoreline Model

Het script voorspelt nu naast 1X2 ook scores:

- `expected_home_goals`
- `expected_away_goals`
- `predicted_score`: meest waarschijnlijke score uit de goalmodellen
- `pool_predicted_score`: score die past bij de 1X2-keuze van de classifier

Voor een voetbalpoule is `pool_predicted_score` het meest logisch, omdat die score nooit de hoofdvoorspelling tegenspreekt.

Algemene tijdsplit `2023-01-01`:

- 1X2 accuracy: `60.98%`
- exact score via puur scoremodel: `13.88%`
- exact score via poule-score: `13.47%`
- poule-score outcome accuracy: `60.98%`
- total goals MAE: `1.4004`

Euro 2024 pre-tournament run, trainen tot `2024-06-13`:

- 1X2/outcome: 28 van 51, `54.90%`
- exacte scores met poule-score: 5 van 51, `9.80%`
- punten bij 1 punt outcome + 1 extra exact: 33
- punten bij 1 punt outcome + 2 extra exact: 38

Euro 2024 met huidig model waarin 2024 al in training zit:

- 1X2/outcome: 28 van 51, `54.90%`
- exacte scores met poule-score: 7 van 51, `13.73%`
- punten bij 1 punt outcome + 1 extra exact: 35
- punten bij 1 punt outcome + 2 extra exact: 42

Vergelijking Euro 2024:

- Ruben: 32 outcomes, 13 exact
- huidig model: 28 outcomes, 7 exact
- Thomas: 24 outcomes, 4 exact

Conclusie:

Scorevoorspelling voegt bruikbare poule-output toe, maar exacte scores blijven moeilijk. Ook met 2024 in training had het model Euro 2024 niet gewonnen van Ruben; het zou wel boven Thomas zijn geëindigd onder gangbare puntentellingen.

## Per Toernooi Met Scorepunten

Met `pool_predicted_score`, dus scorelijn die past bij de 1X2-voorspelling.

Puntdefinities:

- `p1e1`: 1 punt voor juiste uitkomst + 1 extra punt voor exacte score
- `p1e2`: 1 punt voor juiste uitkomst + 2 extra punten voor exacte score

UEFA Euro 2021:

- model: 32 outcomes, 9 exact, `p1e1=41`, `p1e2=50`
- beste mens op `p1e2`: Berend, 27 outcomes, 9 exact, `p1e2=45`
- Thomas: 29 outcomes, 6 exact, `p1e2=41`
- conclusie: model had Euro 2021 gewonnen onder beide puntdefinities

FIFA World Cup 2022:

- model: 37 outcomes, 6 exact, `p1e1=43`, `p1e2=49`
- beste mens op `p1e2`: Daan, 29 outcomes, 7 exact, `p1e2=43`
- Thomas: 30 outcomes, 3 exact, `p1e2=36`
- conclusie: model had WK 2022 gewonnen onder beide puntdefinities

UEFA Euro 2024:

- model pre-tournament: 28 outcomes, 5 exact, `p1e1=33`, `p1e2=38`
- Ruben: 32 outcomes, 13 exact, `p1e1=45`, `p1e2=58`
- Thomas: 24 outcomes, 4 exact, `p1e1=28`, `p1e2=32`
- conclusie: model had Euro 2024 niet gewonnen, maar was wel beter dan Thomas

## Euro 2024 Update Na Eerste Pouleronde

Vraag: wordt de voorspelling voor de tweede poulewedstrijd beter als het model de eerste pouleronde al kent?

Vergelijking op de 12 tweede-poulewedstrijdmatches van Euro 2024:

- pre-tournament model, trainen t/m `2024-06-13`: 6 outcomes goed, 1 exact
- model na eerste pouleronde, trainen t/m `2024-06-18`: 6 outcomes goed, 2 exact
- outcome predictions gewijzigd: 0 van 12
- score predictions gewijzigd: 1 van 12

Conclusie:

Alleen het toevoegen van de eerste pouleronde veranderde weinig. De team-Elo/ranking en historische signalen domineren nog. Het hielp exact-score licht, doordat België-Roemenië naar `2-0` verschoof.

## Feature V2

Toegevoegd in `train_xgboost_worldcup.py`:

- live tournament-state per team: punten, doelsaldo, goals, matchnummer, opening/second/third match, pressure-indicatoren
- contextfeatures: host-country, same-confederation-as-host, home/away/host confederation
- Fjelstul WK-managerfeatures waar beschikbaar: managerervaring, manager-punten-per-match, manager-country match, manager-streak/change
- recency sample weights als optie via `--sample-half-life-days`

Belangrijk: sample weights zijn toegevoegd, maar staan standaard uit (`--sample-half-life-days 0`), omdat ze in de huidige test juist slechter presteerden.

Recency sample weights betekenen hier: een recente wedstrijd krijgt meer gewicht in `XGBoost.fit()` dan een oude wedstrijd. De implementatie gebruikt exponentiele decay met half-life in dagen en wordt toegepast op zowel het 1X2-model als de twee scoremodellen. Voorbeeld: `--sample-half-life-days 3650` geeft een wedstrijd van 10 jaar oud ongeveer de helft van het basisgewicht, met ondergrens `--sample-min-weight`.

Algemene tijdsplit vanaf `2023-01-01`:

Baseline met player-form maar zonder V2:

- features: 193
- accuracy: `60.98%`
- log loss: `0.8554`
- brier: `0.5028`
- pool exact score: `13.47%`

V2 zonder sample weights, huidige default:

- features: 269
- manager side-matches: 1.928
- accuracy: `61.33%`
- log loss: `0.8531`
- brier: `0.5015`
- pool exact score: `13.70%`

V2 met sample half-life 10 jaar:

- accuracy: `60.89%`
- log loss: `0.8576`
- brier: `0.5043`

V2 met sample half-life 20 jaar:

- accuracy: `60.86%`
- log loss: `0.8566`
- brier: `0.5036`

Conclusie:

Tournament/context/managerfeatures helpen licht. Recency sample weights zijn beschikbaar, maar bij deze dataset niet standaard verstandig; de recency-informatie zit waarschijnlijk al genoeg in Elo, recente teamvorm en player-form features.

Feature importance V2 top-signalen bevat onder andere:

- `elo_expected_home`
- `elo_diff`
- `fifa_rank_diff`
- `player_recency_quality_goal_contrib_2y_diff`
- `either_team_is_host_country`
- `home_team_is_host_country`
- `tournament_goal_diff_diff`
- `away_tournament_goal_diff_per_match`
- confederation-features zoals `away_confederation_CAF` en `away_confederation_UEFA`

## Feature V2 Per Toernooi

UEFA Euro 2021:

- model: 32 outcomes, 9 exact, `p1e1=41`, `p1e2=50`
- beste mens op `p1e2`: Berend, 27 outcomes, 9 exact, `p1e2=45`
- Thomas: 29 outcomes, 6 exact, `p1e2=41`
- conclusie: model had gewonnen

FIFA World Cup 2022:

- model: 35 outcomes, 5 exact, `p1e1=40`, `p1e2=45`
- beste mens op `p1e2`: Daan, 29 outcomes, 7 exact, `p1e2=43`
- Thomas: 30 outcomes, 3 exact, `p1e2=36`
- conclusie: model had gewonnen

UEFA Euro 2024:

- model: 28 outcomes, 5 exact, `p1e1=33`, `p1e2=38`
- Ruben: 32 outcomes, 13 exact, `p1e1=45`, `p1e2=58`
- Thomas: 24 outcomes, 4 exact, `p1e1=28`, `p1e2=32`
- conclusie: model had niet gewonnen, maar wel boven Thomas geëindigd

## Feature V2 Update Na Eerste Pouleronde Euro 2024

Tweede pouleronde, 12 wedstrijden:

- pre-tournament V2: 6 outcomes goed, 1 exact
- V2 na eerste pouleronde: 6 outcomes goed, 2 exact
- outcome predictions gewijzigd: 0 van 12
- score predictions gewijzigd: 2 van 12

Conclusie:

Zelfs met tournament-state veranderde een pouleronde weinig aan de 1X2-keuzes. De scorelijnen reageerden iets meer: Duitsland-Hongarije verschoof naar exact `2-0`.

## Fjelstul Match Audit

Command:

`python analyze_fjelstul_matching.py`

Resultaat:

- martj42 complete mannen-WK wedstrijden: 964
- Fjelstul mannen-WK wedstrijden: 964
- Fjelstul vrouwen-WK wedstrijden: 284, uitgesloten voor dit model
- mannen-WK exact gekoppeld met dezelfde home/away-volgorde: 825
- mannen-WK gekoppeld met omgekeerde home/away-volgorde: 139
- niet-gekoppelde mannen-WK wedstrijden: 0
- vrouwen-WK regels die per ongeluk een mannenwedstrijd matchen: 0
- managerfeatures na filter: 964 WK-wedstrijden, 1.928 team-sides

Fix:

Fjelstul managerfeatures worden nu expliciet gefilterd op `FIFA Men's World Cup`. Daarnaast zijn historische naamverschillen opgelost: onder andere `West Germany` -> `Germany`, `Soviet Union` -> `Russia`, `Zaire` -> `DR Congo`, `East Germany` -> `German DR`, en datumafhankelijk `Yugoslavia` na 1992 -> `Serbia`.

## Feature V3 XFKZ Extract

Command:

`python train_xgboost_worldcup.py --skip-download --use-xfkz-features --output-dir outputs_feature_v3_xfkz --model-dir models_feature_v3_xfkz`

Extract:

- `xfkz_country_market_injury_snapshots.csv`: 61.080 land-datumregels, ongeveer 8,2 MB.
- Historische marktwaardes en blessures zijn per wedstrijddatum met `merge_asof` gekoppeld.
- Current squad/national-player exports zijn apart bewaard, maar niet in de historische training gebruikt vanwege leakage-risico.

Resultaat:

- features: 323
- xfkz snapshot rows matched: 18.934
- accuracy: `61.21%`
- log loss: `0.8529`
- brier: `0.5013`
- pool exact score: `13.56%`

Conclusie:

XFKZ helpt de kansverdeling iets (`log_loss` en `brier` beter), maar maakte de harde 1X2-accuracy en exacte poulescore niet beter dan V2. Daarom blijft V2 de standaard voor poulepunten, en staat XFKZ optioneel aan met `--use-xfkz-features`.

## Feature V3 XFKZ Met Player Performance

Command:

`python train_xgboost_worldcup.py --skip-download --use-xfkz-features --output-dir outputs_feature_v3_xfkz_perf --model-dir models_feature_v3_xfkz_perf`

Extract:

- `xfkz_country_market_injury_snapshots.csv`: 61.404 land-datumregels inclusief current-season snapshot van 12 mei 2026.
- Extra `perf_*` kolommen uit `player_performances.csv`, tijdveilig gekoppeld aan het laatste volledig beschikbare seizoen.
- Voorbeeld: `24/25` is beschikbaar vanaf 1 juli 2025.
- Voor WK 2026 is een current-season override toegevoegd met `--current-season-as-of 2026-05-12`, waardoor future fixtures na die datum de actuele `25/26` performance kunnen gebruiken zonder de historische backtest te veranderen.

Resultaat:

- features: 356
- xfkz snapshot rows matched: 18.934
- accuracy: `61.18%`
- log loss: `0.8529`
- brier: `0.5013`
- pool exact score: `13.88%`

Conclusie:

De player-performance features worden duidelijk gebruikt door XGBoost (`xfkz_perf_player_count_diff`, `xfkz_perf_top23_minutes_diff`, apps/cards/goals). Ze verbeteren de exacte poulescore licht tegenover de eerdere XFKZ-run, maar V2 zonder XFKZ blijft nog net beter op harde 1X2-accuracy.

## Feature V3 Geo

Command:

`python train_xgboost_worldcup.py --skip-download --use-geo-features --output-dir outputs_feature_v2_geo --model-dir models_feature_v2_geo`

Extract:

- `worldcities_city_locations.csv`: 45.787 city/country locaties.
- `worldcities_country_representatives.csv`: 244 land-representatieve locaties.
- Dekking op `results.csv`: venue geo voor 44.750 van 49.256 complete wedstrijden (`90,85%`).

Resultaat:

- features: 289
- geo rows matched: 44.750
- accuracy: `60.92%`
- log loss: `0.8525`
- brier: `0.5015`
- pool exact score: `13.64%`

Conclusie:

Geo/travel-context is nuttig om het WK-script klaar te maken, maar de eerste test verlaagt de harde 1X2-accuracy. Daarom staat geo optioneel aan met `--use-geo-features`, niet standaard.

## Feature Inventory

Command:

`python analyze_feature_inventory.py`

Resultaat:

- standaard V2 features: 341
- feature-unie inclusief optionele lagen: 859
- output: `outputs/feature_inventory_union.csv`

Belangrijkste groepen:

- Soccerbase rolling match-stats: 132 optionele features
- Soccerbase lineups/opstellingsvorm: 138 optionele features
- team history: 92 standaard features
- Soccerbase rolling cards: 90 optionele features
- Transfermarkt player-form: 59 standaard features
- live tournament-state: 47 standaard features
- toernooi-selecties: 42 standaard features, 60 inclusief optionele squad-market-values
- goalscorer-form: 30 standaard features
- Fjelstul manager: 18 standaard features
- FIFA ranking: 8 standaard features
- externe Kaggle-Elo: 6 optionele features
- odds: 27 optionele features zodra een odds-CSV wordt meegegeven
- geo/travel: 20 optionele features
- XFKZ market/injury/performance: 87 optionele features

## World Cup 2026 Fixtures

Command:

`python prepare_worldcup2026_fixtures.py`

Resultaat:

- alle WK 2026 fixtures: 104
- fixtures met bekende teams: 54
- output: `data/extracted/worldcup2026_future_fixtures.csv`
- ready output: `data/extracted/worldcup2026_future_fixtures_known_teams.csv`

## Beat The Bookie Closing Odds

Command:

`python extract_beat_the_bookie_odds.py`

Extract:

- bron: `data/kagglehub/datasets/austro/beat-the-bookie-worldwide-football-dataset/versions/2/closing_odds.csv.gz`
- output: `data/extracted/beat_the_bookie_closing_1x2.csv`
- rows: 479.440 closing-odds wedstrijden
- periode: 2005-01-01 t/m 2015-06-30
- gematchte internationale wedstrijden: 5.540
- gematchte FIFA World Cup wedstrijden: 170

Modeltest:

`python train_xgboost_worldcup.py --skip-download --odds-csv data/extracted/beat_the_bookie_closing_1x2.csv --output-dir outputs_feature_v2_odds_btb --model-dir models_feature_v2_odds_btb`

Resultaat:

- features: 289
- odds rows matched: 5.540
- test odds rows vanaf 2023: 0
- accuracy: `61.10%`
- log loss: `0.8536`
- brier: `0.5018`
- pool exact score: `13.53%`

Conclusie:

De odds-features werken en worden gebruikt door XGBoost, maar deze Kaggle-set stopt te vroeg voor de huidige testperiode. Daarom verbetert hij de 2023-2026 holdout niet. Voor WK 2026 is nog een actuele odds-CSV nodig.

## Goalscorer Form, Tournament Squads En OddsPortal Odds

Nieuwe standaardfeatures:

- `goalscorers.csv`: per land doelpuntenmakersvorm in de laatste 730 dagen voor de wedstrijd, inclusief recency-score, unieke scorers, top-scorer afhankelijkheid en penalty-share. Goals uit dezelfde wedstrijddatum tellen niet mee.
- `wikipedia_tournament_squads.csv`: toernooi-selectiefeatures voor WK/EK/Copa/Africa Cup, o.a. selectie-leeftijd, caps, goals en positieverdeling. Dit zijn selecties, geen match-lineups.
- `--use-squad-market-values`: optionele koppeling van squad-spelers aan Transfermarkt-waarde op naam + geboortedatum, tijdveilig met waardes van voor de toernooistart.
- `--sample-half-life-days`: recency sample weights zijn beschikbaar, maar blijven standaard uit omdat de eerste holdout-test lager scoorde.
- `--odds-csv data/extracted/oddsportal_international_closing_1x2.csv`: actuele OddsPortal closing 1X2 odds met coverage t/m 2026.

Zonder odds:

`python train_xgboost_worldcup.py --skip-download --output-dir outputs_feature_v2_goalscorers_squads_no_sample_weights --model-dir models_feature_v2_goalscorers_squads_no_sample_weights`

Resultaat:

- features: 341
- goalscorer rows matched: 38.720
- squad rows matched: 1.406
- accuracy: `61.41%`
- log loss: `0.8533`
- brier: `0.5015`
- pool exact score: `13.56%`

Met OddsPortal closing odds:

`python train_xgboost_worldcup.py --skip-download --odds-csv data/extracted/oddsportal_international_closing_1x2.csv --output-dir outputs_feature_v2_oddsportal_plus_goalscorers_squads_no_sample_weights --model-dir models_feature_v2_oddsportal_plus_goalscorers_squads_no_sample_weights`

Resultaat:

- features: 368
- odds rows matched: 11.415
- test odds rows vanaf 2023: 2.076
- accuracy: `61.21%`
- log loss: `0.8488`
- brier: `0.4989`
- pool exact score: `13.53%`

Recency sample weights test:

- command extra: `--sample-half-life-days 3650`
- zonder odds: accuracy `60.95%`, log loss `0.8576`
- met OddsPortal odds: accuracy `60.95%`, log loss `0.8530`

Squad market-value test:

- command extra: `--use-squad-market-values`
- features: 359
- accuracy: `61.24%`
- log loss: `0.8533`

Conclusie:

De goalscorer- en squadfeatures helpen licht op harde 1X2-accuracy. OddsPortal closing odds verbeteren vooral de kanskalibratie (`log_loss` en `brier`) en staan hoog in feature-importance, maar geven in deze split niet de hoogste top-1 accuracy. Recency sample weights en squad-market-values zijn toegevoegd, maar niet standaard aan omdat deze tests lager scoorden.

## Soccerbase Live Lineups En FIFA Ratings

Nieuwe optionele featurelaag:

- bron: `data/extracted/soccerbase_lineups_used.csv`
- match-koppeling: datum + unordered team-pair + teamnaam
- dekking: 4.668 historische wedstrijden gematcht, waarvan 771 in de 2023-2026 holdout
- ratingsdekking in lineupbestand: ongeveer 42% van player rows
- de featurelaag gebruikt uitsluitend starters (`is_starter=1`); gebruikte wissels worden genegeerd omdat die niet eerlijk pre-match bekend zijn
- `--use-sofifa-yearly-ratings` vervangt de 2025-snapshot door een tijdveilige SoFIFA/EAFC-ratingreeks 2015-2026 uit `data/extracted/sofifa_yearly_player_ratings.csv`

Starters + FIFA/SoFIFA ratings:

`python train_xgboost_worldcup.py --skip-download --use-soccerbase-lineup-features --use-soccerbase-ratings --output-dir outputs_feature_v2_soccerbase_starters_ratings --model-dir models_feature_v2_soccerbase_starters_ratings`

Resultaat:

- features: 401
- Soccerbase lineup rows matched: 4.668
- accuracy: `61.24%`
- log loss: `0.8541`
- pool exact score: `13.70%`

Starters zonder FIFA/SoFIFA ratings:

`python train_xgboost_worldcup.py --skip-download --use-soccerbase-lineup-features --output-dir outputs_feature_v2_soccerbase_starters_no_ratings --model-dir models_feature_v2_soccerbase_starters_no_ratings`

Resultaat:

- features: 362
- Soccerbase lineup rows matched: 4.668
- accuracy: `61.12%`
- log loss: `0.8539`
- pool exact score: `13.70%`

Starters + tijdveilige SoFIFA/EAFC jaarratings:

`python train_xgboost_worldcup.py --skip-download --use-soccerbase-lineup-features --use-sofifa-yearly-ratings --output-dir outputs_feature_v2_soccerbase_starters_yearly_ratings --model-dir models_feature_v2_soccerbase_starters_yearly_ratings`

Resultaat:

- features: 461
- Soccerbase lineup rows matched: 4.668
- accuracy: `61.24%`
- log loss: `0.8553`
- pool exact score: `13.73%`

Conclusie:

De lineup-features worden wel gebruikt (`lineup_overall_avg_diff` staat in feature importance als ratings aan staan), maar verbeteren de 1X2-score in deze holdout niet ten opzichte van de huidige standaardrun zonder lineups (`61.41%`). Zonder ratings zakt accuracy naar `61.12%`; met snapshot-ratings en met tijdveilige jaarratings komt accuracy op `61.24%`. Voor live WK 2026 kan confirmed starter info nog steeds nuttig zijn, maar in de historische holdout geeft het nu vooral ruis.

## Paper Voetbal Features, Rolling Stats En SoFIFA

Bron: `paper voetbal.pdf` (`Prediction of football match results with Machine Learning`, Procedia Computer Science 204, 2022).

Belangrijkste aanpak in het paper:

- Premier League, 1.900 wedstrijden over 5 seizoenen.
- Train/test split per seizoen: 4 seizoenen train, 1 seizoen test.
- Rolling pre-match averages van vorige wedstrijden: goals, shots, shots on target, corners, fouls, yellow cards, red cards.
- Odds per match.
- Referee.
- SoFIFA team/player quality: overall, attack, midfield, defence en speler-skills.
- Feature selectie met Boruta/variabele combinaties.
- Beste resultaat: Random Forest met 15 geselecteerde variabelen, accuracy `65.26%`.

Lokale toepasbaarheid:

- Onze bijgewerkte `soccerbase_match_stats.csv` bevat nu bruikbare stats voor een deel van de wedstrijden: possession, shots on/off target en corners.
- Directe match stats van dezelfde wedstrijd zijn leakage; daarom gebruikt `--use-soccerbase-stat-features` alleen rolling gemiddelden uit eerdere wedstrijden.
- De paper-feature `referee` is nog niet toegevoegd, omdat die in deze internationale data niet stabiel beschikbaar is.
- Fouls, yellow cards en red cards zijn nog niet toegevoegd als rolling features; `soccerbase_cards_events.csv` is er wel, maar is nog niet omgezet naar eerlijke pre-match teamgemiddelden.
- SoFIFA/FIFA ratings zijn tijdveiliger gemaakt met `extract_sofifa_yearly_ratings.py` en `--use-sofifa-yearly-ratings`.

Rolling Soccerbase match-stats zonder lineups:

`python train_xgboost_worldcup.py --skip-download --use-soccerbase-stat-features --output-dir outputs_feature_v2_soccerbase_stats --model-dir models_feature_v2_soccerbase_stats`

Resultaat:

- features: 473
- Soccerbase rolling stat rows matched: 10.823
- accuracy: `61.56%`
- log loss: `0.8547`
- exact score: `14.11%`
- pool exact score: `13.96%`

Na toevoegen van AFC Asian Cup aan de squad-competitiemapping:

`python train_xgboost_worldcup.py --skip-download --use-soccerbase-stat-features --output-dir outputs_feature_v2_soccerbase_stats_afc_squads --model-dir models_feature_v2_soccerbase_stats_afc_squads`

Resultaat:

- squad feature rows matched: 1.636 in plaats van 1.406
- accuracy: `61.56%`
- log loss: `0.8548`
- pool exact score: `14.02%`

Dit helpt vooral de dekking; de 1X2-accuracy blijft afgerond gelijk.

Externe Kaggle-Elo zonder lineups:

`python train_xgboost_worldcup.py --skip-download --use-external-elo-features --output-dir outputs_feature_v2_external_elo --model-dir models_feature_v2_external_elo`

Resultaat:

- external Elo rows matched: 41.146
- features: 347
- accuracy: `61.27%`
- log loss: `0.8525`
- pool exact score: `13.64%`

Externe Kaggle-Elo + rolling Soccerbase stats, zonder lineups:

`python train_xgboost_worldcup.py --skip-download --use-external-elo-features --use-soccerbase-stat-features --output-dir outputs_feature_v2_external_elo_soccerbase_stats --model-dir models_feature_v2_external_elo_soccerbase_stats`

Resultaat:

- external Elo rows matched: 41.146
- Soccerbase rolling stat rows matched: 10.823
- features: 479
- accuracy: `61.53%`
- log loss: `0.8530`
- exact score: `14.25%`

Selectie-marktwaarde + rolling Soccerbase stats, zonder lineups:

`python train_xgboost_worldcup.py --skip-download --use-soccerbase-stat-features --use-squad-market-values --output-dir outputs_feature_v2_soccerbase_stats_squad_market --model-dir models_feature_v2_soccerbase_stats_squad_market`

Resultaat:

- features: 491
- accuracy: `61.56%`
- log loss: `0.8547`
- exact score: `14.19%`

Recency-gewogen Soccerbase stats, zonder lineups:

`python train_xgboost_worldcup.py --skip-download --use-soccerbase-stat-features --use-soccerbase-stat-recency-features --output-dir outputs_feature_v2_soccerbase_stats_recency --model-dir models_feature_v2_soccerbase_stats_recency`

Resultaat:

- features: 605
- accuracy: `61.53%`
- log loss: `0.8546`
- exact score: `14.17%`

De recency-gewogen statvariant staat daarom optioneel aan met `--use-soccerbase-stat-recency-features`, niet standaard binnen de stats-laag.

Rolling Soccerbase stats + rolling cards, zonder lineups:

`python train_xgboost_worldcup.py --skip-download --use-soccerbase-stat-features --use-soccerbase-card-features --output-dir outputs_feature_v2_soccerbase_stats_cards --model-dir models_feature_v2_soccerbase_stats_cards`

Resultaat:

- features: 563
- Soccerbase rolling card rows matched: 14.850
- accuracy: `61.21%`
- log loss: `0.8555`
- exact score: `14.17%`
- score outcome: `59.13%`

Rolling Soccerbase stats + starters zonder ratings:

`python train_xgboost_worldcup.py --skip-download --use-soccerbase-stat-features --use-soccerbase-lineup-features --output-dir outputs_feature_v2_soccerbase_stats_lineups_no_ratings --model-dir models_feature_v2_soccerbase_stats_lineups_no_ratings`

Resultaat:

- features: 494
- Soccerbase lineup rows matched: 4.668
- Soccerbase rolling stat rows matched: 10.823
- accuracy: `61.44%`
- log loss: `0.8547`
- exact score: `14.11%`

Rolling Soccerbase stats + opstellingsvorm zonder ratings:

`python train_xgboost_worldcup.py --skip-download --use-soccerbase-stat-features --use-soccerbase-lineup-features --output-dir outputs_feature_v2_soccerbase_stats_lineup_shapes --model-dir models_feature_v2_soccerbase_stats_lineup_shapes`

Resultaat:

- features: 512
- Soccerbase lineup rows matched: 4.683
- Soccerbase rolling stat rows matched: 10.847
- accuracy: `61.15%`
- log loss: `0.8549`
- exact score: `14.22%`
- score outcome: `59.10%`

Rolling Soccerbase stats + rolling cards + opstellingsvorm zonder ratings:

`python train_xgboost_worldcup.py --skip-download --use-soccerbase-stat-features --use-soccerbase-card-features --use-soccerbase-lineup-features --output-dir outputs_feature_v2_soccerbase_stats_cards_lineup_shapes --model-dir models_feature_v2_soccerbase_stats_cards_lineup_shapes`

Resultaat:

- features: 602
- Soccerbase lineup rows matched: 4.683
- Soccerbase rolling card rows matched: 14.901
- accuracy: `61.10%`
- log loss: `0.8569`
- exact score: `14.25%`
- score outcome: `59.41%`

Rolling Soccerbase stats + starters + tijdveilige SoFIFA/EAFC jaarratings:

`python train_xgboost_worldcup.py --skip-download --use-soccerbase-stat-features --use-soccerbase-lineup-features --use-sofifa-yearly-ratings --output-dir outputs_feature_v2_soccerbase_stats_lineups_yearly_ratings --model-dir models_feature_v2_soccerbase_stats_lineups_yearly_ratings`

Resultaat:

- features: 593
- Soccerbase lineup rows matched: 4.668
- Soccerbase rolling stat rows matched: 10.823
- accuracy: `61.30%`
- log loss: `0.8562`
- exact score: `14.02%`

Conclusie:

De paper-aanpak bevestigt vooral dat rolling pre-match stats nuttig zijn. In deze internationale dataset is de beste nieuwe test `--use-soccerbase-stat-features` zonder lineups/cards: `61.56%` accuracy tegen `61.41%` voor de eerdere standaard. Opstellingsvorm en rolling cards worden wel door XGBoost gebruikt en verbeteren sommige scoremodel-metrics licht, maar verlagen de harde 1X2-accuracy in deze holdout. Daarom blijven ze optioneel voor analyse en live WK-scenario's, niet standaard voor de poule-run.

## Toernooi-Backtests Met Soccerbase Rolling Stats

Eerlijke pre-tournament runs, telkens trainen tot net voor het toernooi en daarna alleen het toernooi filteren.

UEFA Euro 2021:

- command: `python train_xgboost_worldcup.py --skip-download --test-from 2021-06-11 --use-soccerbase-stat-features --output-dir outputs_eval_euro2021_stats --model-dir models_eval_euro2021_stats`
- outcomes: 31 van 51, `60.78%`
- pool exact: 6 van 51, `11.76%`
- `p1e1=37`, `p1e2=43`

FIFA World Cup 2022:

- command: `python train_xgboost_worldcup.py --skip-download --test-from 2022-11-20 --use-soccerbase-stat-features --output-dir outputs_eval_wc2022_stats --model-dir models_eval_wc2022_stats`
- outcomes: 33 van 64, `51.56%`
- pool exact: 5 van 64, `7.81%`
- `p1e1=38`, `p1e2=43`

UEFA Euro 2024:

- command: `python train_xgboost_worldcup.py --skip-download --test-from 2024-06-14 --use-soccerbase-stat-features --output-dir outputs_eval_euro2024_stats --model-dir models_eval_euro2024_stats`
- outcomes: 28 van 51, `54.90%`
- pool exact: 6 van 51, `11.76%`
- `p1e1=34`, `p1e2=40`

Conclusie:

De rolling stat-laag verbetert de algemene 2023+ holdout licht, maar niet deze drie poule-toernooien samen. De oudere toernooi-backtests zonder deze laag waren beter op Euro 2021 en WK 2022. Daarom moet de feature optioneel blijven totdat we hem op meer toernooisplits hebben gevalideerd.
