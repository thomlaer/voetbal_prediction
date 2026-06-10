# Lokale datasets

Alle bestanden hieronder staan binnen deze projectmap. De huidige aanpak gebruikt geen live API's voor training.

## Internationale uitslagen

Pad: `data/`

Bron: https://github.com/martj42/international_results

Belangrijkste bestanden:
- `results.csv`: 49.256 internationale mannenwedstrijden, 1872-11-30 t/m 2026-03-31.
- `shootouts.csv`: penalty-shootouts.
- `goalscorers.csv`: doelpuntenmakers.
- `former_names.csv`: historische teamnamen.

Gebruik:
- Basis voor het XGBoost 1X2-model.
- Vorm, Elo, goals, home/away/neutral, toernooi en FIFA-ranking features.
- `goalscorers.csv` wordt nu ook als pre-match scorer-vorm gebruikt: alleen goals van voor de wedstrijddatum tellen mee.

## Ratings

FIFA-ranking:
- Hoofdbestand: `fifa_ranking-2026-04-01.csv`.
- Dekking: 70.194 rankingregels, 1992-12-31 t/m 2026-04-01.
- Standaard aan via `--rankings`.
- De Kaggle-set `lucasyukioimafuko/fifa-mens-world-ranking` staat ook lokaal, maar loopt slechts t/m 2024 en is daarom minder geschikt dan het huidige 2026-bestand.

Elo:
- Interne Elo wordt standaard uit `results.csv` opgebouwd en is een van de sterkste features.
- Externe Kaggle-Elo staat in `data/kagglehub/datasets/saifalnimri/international-football-elo-ratings/versions/1/eloratings.csv`.
- Optioneel via `--use-external-elo-features`.
- De externe Elo wordt strikt tijdveilig gebruikt: alleen ratings van vóór de wedstrijddatum tellen mee, omdat ratings op dezelfde datum het wedstrijdresultaat kunnen bevatten.
- Dekking: 41.146 historische wedstrijdrijen met beide teams gematcht.

## Transfermarkt

Pad: `data/kagglehub/datasets/davidcariboo/player-scores/versions/655/`

Bron: https://www.kaggle.com/datasets/davidcariboo/player-scores

Belangrijkste bestanden:
- `games.csv`: 88.271 wedstrijden.
- `appearances.csv`: 1.877.839 speler-wedstrijdregels met goals, assists, kaarten en minuten.
- `game_lineups.csv`: 3.149.964 lineupregels met starters en wissels.
- `player_valuations.csv`: 507.815 historische marktwaardes.
- `players.csv`: 47.637 spelers.
- `national_teams.csv`: 118 nationale teams met o.a. marktwaarde en FIFA-ranking.

Let op:
- De dataset is sterk voor clubvorm, marktwaarde en spelerfeatures.
- Voor nationale teamwedstrijden is de directe dekking beperkter: 670 national-team games, 128 met lineupregels, 52 met appearance-regels.

## Fjelstul World Cup Database

Pad: `data/fjelstul_worldcup/data-csv/`

Bron: https://github.com/jfjelstul/worldcup

Belangrijkste bestanden:
- `matches.csv`: 1.248 WK-wedstrijden.
- `squads.csv`: 13.843 WK-selectieregels.
- `player_appearances.csv`: 27.432 player appearances vanaf WK 1970, inclusief `starter` en `substitute`.
- `substitutions.csv`: 10.222 wissels vanaf WK 1970.
- `goals.csv`: 3.637 goals.
- `bookings.csv`: 3.178 kaarten vanaf WK 1970.

Gebruik:
- Beste lokale CSV-bron voor historische WK-opstellingen.
- Kan gekoppeld worden aan Transfermarkt/Sofifa spelerkwaliteit via spelernaam, land en jaar.
- Mannen-WK-selecties: 10.973 spelers/selectieregels over 22 toernooien, 1930 t/m 2022.
- Mannen-WK-player appearances: 20.618 speler-wedstrijdregels over 764 wedstrijden, 1970 t/m 2022.
- Voor WK 2022 bevat `player_appearances.csv` 1.995 speler-wedstrijdregels, waarvan 1.408 starters en 587 invallers.
- Match-audit met `data/results.csv`: 964 van 964 mannen-WK-wedstrijden koppelen. Fjelstul bevat ook 284 vrouwen-WK-wedstrijden; die worden expliciet uitgesloten voor dit mannenmodel.
- Van de 964 mannen-WK-wedstrijden hebben 825 dezelfde home/away-volgorde en 139 een omgekeerde home/away-volgorde. Dat is geen scorefout, maar verschil in bronconventie.
- Venue-context: alle 964 mannen-WK-wedstrijden hebben stadium, city en country; coördinaten, tijdzone en klimaat staan niet in Fjelstul en vragen een aparte geo/klimaat-CSV.

## Wikipedia Tournament Squads

Pad: `data/extracted/wikipedia_tournament_squads.csv`

Bron: Wikipedia tournament squad pages.

Dekking:
- 18.950 spelers/selectieregels.
- Toernooien: FIFA World Cup, UEFA Euro, Copa America, Africa Cup of Nations, AFC Asian Cup en CONCACAF Gold Cup wanneer die in de CSV staat.
- Jaren: 1996 t/m 2025.

Gebruik:
- Standaard aan in `train_xgboost_worldcup.py`.
- Maakt per land/toernooi selectiefeatures: speler_count, gemiddelde leeftijd, caps, goals, top11 caps, top5 goals en positieverdeling.
- Dit zijn toernooiselecties, geen confirmed match-lineups.
- Bij vertraagde toernooien kan het model een selectie van `jaar - 1` gebruiken, bijvoorbeeld Euro 2020 in kalenderjaar 2021.
- AFC Asian Cup wordt nu ook gemapt; daardoor stijgt de squad-featuredekking van 1.406 naar 1.636 historische wedstrijdrijen.
- CONCACAF Gold Cup/Gold Cup wordt ook gemapt zodra die selecties in `wikipedia_tournament_squads.csv` staan.
- Marktwaarde-koppeling is beschikbaar via `--use-squad-market-values`. Die matcht spelers op naam + geboortedatum en gebruikt alleen Transfermarkt-waardes van voor de toernooistart. De eerste holdout-test was lager, daarom staat dit niet standaard aan.

## Soccerbase Lineups

Pad:
- `data/extracted/soccerbase_lineups.csv`: starters, gebruikte wissels en ongebruikte bankspelers.
- `data/extracted/soccerbase_lineups_used.csv`: starters en gebruikte wissels.
- `data/extracted/soccerbase_match_stats.csv`: post-match stats.
- `data/extracted/soccerbase_cards_events.csv`: kaarten/events.

Gebruik:
- Optioneel in `train_xgboost_worldcup.py` via `--use-soccerbase-lineup-features`.
- Koppeling gebeurt op datum + unordered team pair + teamnaam; daardoor werkt het ook als een bron home/away andersom noteert.
- De featurelaag gebruikt uitsluitend starters (`is_starter=1`). Gebruikte wissels en ongebruikte bankspelers worden genegeerd, omdat die niet eerlijk pre-match bekend zijn.
- Opstellingsvorm wordt als geaggregeerde features gebruikt: aantallen GK/DF/MF/FW, verdedigend/middenveld/aanvallend aandeel, DF/FW-ratio en home-away verschillen. Het model ziet niet elke speler los als losse kolom.
- `--use-soccerbase-ratings` gebruikt de meegeleverde FIFA/SoFIFA player ratings als geaggregeerde teamfeatures. Let op: de huidige extract bevat een `sofifa_2025_snapshot_not_historical` notitie, dus dit is niet volledig tijdveilig voor historische backtests.
- `--use-sofifa-yearly-ratings` gebruikt in plaats daarvan `data/extracted/sofifa_yearly_player_ratings.csv`, met per speler de nieuwste SoFIFA/EAFC-rating die op de wedstrijddatum al bekend kon zijn.
- `--use-soccerbase-stat-features` gebruikt `soccerbase_match_stats.csv` als rolling pre-match laag. Directe stats van dezelfde wedstrijd worden niet gebruikt; per land worden alleen eerdere wedstrijden samengevat.
- De rolling stat-laag bevat possession, shots on target, shots off target, total shots en corners, telkens voor/tegen, als recent venster en als all-time-tot-dan.
- `--use-soccerbase-card-features` gebruikt `soccerbase_cards_events.csv` als rolling pre-match laag. Kaarten van de wedstrijd zelf worden niet gebruikt; alleen eerdere wedstrijden tellen mee.
- De rolling card-laag bevat yellow/red/card-points/players-carded per match, over recent 730 dagen, laatste 5 kaartwedstrijden en all-time-tot-dan.

## SoFIFA Yearly Player Ratings

Pad:
- Ruwe Kaggle-downloads staan onder `data/kagglehub/datasets/...`.
- Compacte extract: `data/extracted/sofifa_yearly_player_ratings.csv`.
- Extract-script: `extract_sofifa_yearly_ratings.py`.

Bronnen:
- FIFA 15 t/m FIFA 22: `stefanoleone992/fifa-22-complete-player-dataset`, alleen `players_15.csv` t/m `players_22.csv`.
- FIFA 23: `stefanoleone992/fifa-23-complete-player-dataset`, `male_players (legacy).csv`.
- FIFA 24: `jmacd745/sofifa-data-set`, `all_fifa_players.csv`.
- FIFA 25: `aniss7/fifa-player-data-from-sofifa-2025-06-03`, `player-data-full-2025-june.csv`.
- EAFC 26: `flynn28/eafc26-player-database`, `EAFC26-Men.csv`.

Gebruik:
- De extract bevat 355.743 speler-jaarregels voor 2015 t/m 2026.
- Belangrijke kolommen: `sofifa_id`, `available_from`, `overall`, `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic`, plus enkele skillvelden.
- In training worden deze ratings alleen via echte Soccerbase starters gebruikt, niet als losse nationale selectie-snapshot.
- Voor historische wedstrijden gebruikt `merge_asof` alleen ratings met `available_from <= match_date`, zodat toekomstige ratings niet teruglekken.

## XFKZ Football Datasets

Pad ruwe data: `data/kagglehub/datasets/xfkzujqjvx97n/football-datasets/versions/2/`

Bron: https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets

Geextraheerde bestanden:
- `data/extracted/xfkz_country_market_injury_snapshots.csv`: compacte landen-snapshots per wedstrijddatum met historische marktwaarde, blessure-indicatoren en laatste volledig beschikbare seizoensperformance.
- `data/extracted/xfkz_current_national_players.csv`: huidige/recent bekende nationale spelers, alleen geschikt voor future/current analyse.
- `data/extracted/xfkz_current_squad_strength.csv`: huidige selectie-sterkte per land, alleen geschikt voor future/current analyse.

Gebruik:
- `xfkz_country_market_injury_snapshots.csv` is tijdveilig gemaakt: per wedstrijd worden alleen marktwaardes en blessures gebruikt die op of voor die wedstrijddatum bekend waren.
- `player_performances.csv` wordt per land samengevat uit het laatste volledig beschikbare clubseizoen. Bijvoorbeeld seizoen `24/25` is pas beschikbaar vanaf 1 juli 2025, zodat historische training geen eindcijfers van een nog lopend seizoen ziet.
- Voor WK 2026 is daarnaast een current-season snapshot toegevoegd met `extract_xfkz_features.py --current-season-as-of 2026-05-12`; daardoor kunnen future fixtures na 12 mei 2026 de actuele `25/26` performance gebruiken zonder de historische backtest te veranderen.
- De current national-player bestanden worden niet gebruikt voor historische training, omdat all-time caps/goals en actuele selectie-info anders toekomstinformatie kunnen lekken.
- In `train_xgboost_worldcup.py` staat deze laag optioneel aan via `--use-xfkz-features`.

## Odds

Status:
- Het model ondersteunt een lokale odds-CSV via `--odds-csv`.
- Beat The Bookie staat lokaal in `data/kagglehub/datasets/austro/beat-the-bookie-worldwide-football-dataset/versions/2/`.
- Geextraheerd naar `data/extracted/beat_the_bookie_closing_1x2.csv`.
- OddsPortal extracts staan lokaal in `data/extracted/oddsportal_international_closing_1x2.csv` en `data/extracted/oddsportal_worldcup_closing_1x2.csv`.
- Verwachte kolommen: datum, home team, away team en decimal 1X2 odds. Ondersteunde odds-namen zijn o.a. `home_odds`, `draw_odds`, `away_odds` of FootyStats-achtige `odds_ft_home_team_win`, `odds_ft_draw`, `odds_ft_away_team_win`.

Gebruik:
- Alleen closing odds gebruiken als de voorspelling vlak voor de wedstrijd wordt gemaakt.
- Voor een pre-tournament poule zijn closing odds voor latere wedstrijden nog niet bekend; die mogen dan niet in de historische pre-tournament vergelijking.
- Beat The Bookie `closing_odds.csv.gz` loopt van 2005-01-01 t/m 2015-06-30. De odds-series files lopen door t/m 2016-11-20, maar zijn geen actuele WK 2026 odds.
- De extract matcht 5.540 internationale wedstrijden in `results.csv`, waarvan 170 FIFA World Cup wedstrijden.
- OddsPortal international closing odds matcht 11.415 wedstrijden in `results.csv`, waarvan 2.076 in de 2023-2026 holdout.

## Geo / Travel Context

Pad brondata:
- `simplemaps_worldcities_basicv1.901/worldcities.csv`
- fallback: `simplemaps_worldcities_basicv1.901.zip`

Geextraheerde bestanden:
- `data/extracted/worldcities_city_locations.csv`: 45.787 city/country locaties.
- `data/extracted/worldcities_country_representatives.csv`: 244 land-representatieve locaties, meestal hoofdstad of grootste administratieve stad.

Gebruik:
- Optioneel in `train_xgboost_worldcup.py` via `--use-geo-features`.
- Maakt venue lat/lng, venue-populatie, ruwe thuisland-naar-venue reisafstand en ruwe longitude-timezone shift.
- Geen historische weerdata. Klimaat/weer wordt bewust niet gegokt zonder betrouwbare bron.
- Eerste holdout-test: betere `log_loss`, maar lagere 1X2-accuracy; daarom niet standaard aan.

## FIFA World Cup 2026 Match Data

Pad ruwe data: `data/kagglehub/datasets/areezvisram12/fifa-world-cup-2026-match-data-unofficial/versions/3/`

Bron: https://www.kaggle.com/datasets/areezvisram12/fifa-world-cup-2026-match-data-unofficial

Belangrijkste bestanden:
- `matches.csv`: 104 WK 2026 fixtures met kickoff, city, stage en team-id's.
- `teams.csv`: teams, groepen en placeholder-indicatoren.
- `host_cities.csv`: host city, land, venue, region cluster en airport code.

Geextraheerde bestanden:
- `data/extracted/worldcup2026_future_fixtures.csv`: alle 104 fixtures.
- `data/extracted/worldcup2026_future_fixtures_known_teams.csv`: 54 fixtures zonder placeholder-teams.

Gebruik:
- Gemaakt met `prepare_worldcup2026_fixtures.py`.
- Kan direct naar `train_xgboost_worldcup.py --future-fixtures`.
- Knockoutwedstrijden en play-off placeholders blijven uit de ready-file totdat teams bekend zijn.

## Mogelijke volgende featurelaag

1. Gebruik Fjelstul WK-opstellingen als testbed:
   - historische starters koppelen aan spelerkwaliteit
   - meten of echte lineups betere voorspellingen geven dan alleen team-Elo/ranking
2. Gebruik pas later live data voor WK 2026:
   - confirmed lineups vlak voor de wedstrijd
   - blessures/schorsingen
   - live score updates
