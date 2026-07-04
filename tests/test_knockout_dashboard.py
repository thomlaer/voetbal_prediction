from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from openpyxl import load_workbook

import make_scorito_worldcup_picks as scorito
from github_vercel_app.tools import publish_latest_outputs as publisher
from predict_worldcup2026_montecarlo import apply_actual_results


class KnockoutDashboardTests(unittest.TestCase):
    def test_shootout_winner_is_used_as_advancing_team(self) -> None:
        fixtures = pd.DataFrame(
            [
                {
                    "date": "2026-06-29",
                    "home_team": "Netherlands",
                    "away_team": "Morocco",
                    "expected_home_goals": 1.0,
                    "expected_away_goals": 1.0,
                    "sim_prob_home_win": 0.4,
                    "sim_prob_draw": 0.3,
                    "sim_prob_away_win": 0.3,
                    "sim_predicted_outcome": "home_win",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                [
                    {
                        "date": "2026-06-29",
                        "home_team": "Netherlands",
                        "away_team": "Morocco",
                        "home_score": 1,
                        "away_score": 1,
                        "tournament": "FIFA World Cup",
                    }
                ]
            ).to_csv(root / "results.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "date": "2026-06-29",
                        "home_team": "Netherlands",
                        "away_team": "Morocco",
                        "winner": "Morocco",
                    }
                ]
            ).to_csv(root / "shootouts.csv", index=False)

            actual = apply_actual_results(
                fixtures,
                root / "results.csv",
                root / "missing_espn.csv",
                root / "shootouts.csv",
            )

        self.assertEqual(actual.loc[0, "actual_score"], "1-1")
        self.assertEqual(actual.loc[0, "actual_advancing_team"], "Morocco")

    def test_dashboard_keeps_shootout_winner_when_espn_has_date_alias(self) -> None:
        prediction = {
            "match_number": 75,
            "date": "2026-06-29",
            "home_team": "Germany",
            "away_team": "Paraguay",
            "score": "2-1",
            "predicted_winner": "Germany",
        }
        shared_actual = {
            "actual_available": True,
            "actual_home_score": 1,
            "actual_away_score": 1,
            "actual_score": "1-1",
            "actual_outcome": "draw",
            "actual_winner": "Draw",
            "actual_source": "espn",
            "actual_online_verified": True,
        }
        espn_actuals = {
            ("2026-06-29", "germany", "paraguay"): shared_actual,
            ("2026-06-28", "germany", "paraguay"): shared_actual,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                [
                    {
                        "date": "2026-06-29",
                        "home_team": "Germany",
                        "away_team": "Paraguay",
                        "winner": "Paraguay",
                    }
                ]
            ).to_csv(root / "shootouts.csv", index=False)

            with patch.object(publisher, "load_espn_actuals", return_value=espn_actuals):
                result = publisher.attach_actual_results(
                    [prediction],
                    root / "results.csv",
                    root / "missing_stats.csv",
                    {},
                    {"by_number": {}, "by_fixture": {}},
                    {"by_number": {}, "by_fixture": {}},
                )

        self.assertEqual(result[0]["actual_advancing_team"], "Paraguay")

    def test_first_round_of_16_links_use_actual_r32_advancers(self) -> None:
        source_rows = [
            (73, "South Africa", "Canada", "0-1", "Canada"),
            (74, "Brazil", "Japan", "2-1", "Brazil"),
            (75, "Germany", "Paraguay", "1-1", "Paraguay"),
            (76, "Netherlands", "Morocco", "1-1", "Morocco"),
            (77, "Ivory Coast", "Norway", "1-2", ""),
            (78, "France", "Sweden", "2-1", ""),
        ]
        rows = [
            {
                "match_number": match_number,
                "home_team": home,
                "away_team": away,
                "score": score,
                "actual_available": bool(advancing),
                "actual_score": "1-1" if advancing else "",
                "actual_advancing_team": advancing,
                "fixture_confirmed": True,
            }
            for match_number, home, away, score, advancing in source_rows
        ]
        rows.extend(
            [
                {"match_number": 89, "home_team": "Canada", "away_team": "Germany", "score": "1-0"},
                {"match_number": 90, "home_team": "Brazil", "away_team": "Norway", "score": "2-1"},
                {"match_number": 91, "home_team": "Netherlands", "away_team": "France", "score": "1-2"},
            ]
        )

        repaired = publisher.repair_knockout_bracket(rows)
        by_match = {row["match_number"]: row for row in repaired}

        self.assertEqual((by_match[89]["home_team"], by_match[89]["away_team"]), ("Canada", "Morocco"))
        self.assertEqual((by_match[90]["home_team"], by_match[90]["away_team"]), ("Paraguay", "Norway"))
        self.assertEqual((by_match[91]["home_team"], by_match[91]["away_team"]), ("Brazil", "France"))

    def test_entry_sheet_keeps_match_expected_goals(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "match_number": 89,
                    "home_team": "Canada",
                    "away_team": "Morocco",
                    "score": "1-1",
                    "expected_home_goals": 1.23,
                    "expected_away_goals": 1.08,
                }
            ]
        )

        clean = scorito.clean_entry_sheet(frame)

        self.assertIn("expected_home_goals", clean.columns)
        self.assertIn("expected_away_goals", clean.columns)

    def test_probability_excel_includes_match_xg(self) -> None:
        with tempfile.TemporaryDirectory(dir=publisher.APP_ROOT / "public") as directory:
            destination = Path(directory) / "probabilities.xlsx"
            publisher.write_prediction_excel(
                [
                    {
                        "match_number": 89,
                        "date": "2026-07-04",
                        "stage": "Round of 16",
                        "home_team": "Canada",
                        "away_team": "Morocco",
                        "score": "1-1",
                        "prob_home_win": 0.31,
                        "prob_draw": 0.34,
                        "prob_away_win": 0.35,
                        "expected_home_goals": 1.23,
                        "expected_away_goals": 1.08,
                    }
                ],
                destination,
                include_probabilities=True,
            )
            workbook = load_workbook(destination, read_only=True)
            sheet = workbook.active
            headers = [cell.value for cell in sheet[2]]
            values = [cell.value for cell in sheet[3]]
            workbook.close()

        self.assertEqual(headers[12:14], ["xG thuis", "xG uit"])
        self.assertEqual(values[12:14], [1.23, 1.08])

    def test_round_topscorers_are_recomputed_from_published_xg(self) -> None:
        predictions = [
            {
                "stage": "Round of 16",
                "home_team": "Paraguay",
                "away_team": "France",
                "score": "0-1",
                "expected_home_goals": 0.74,
                "expected_away_goals": 1.74,
                "actual_available": False,
            },
            {
                "stage": "Round of 16",
                "home_team": "United States",
                "away_team": "Belgium",
                "score": "1-2",
                "expected_home_goals": 1.02,
                "expected_away_goals": 1.46,
                "actual_available": False,
            },
        ]
        stale_rows = [
            {
                "stage": "Round of 16",
                "stage_label": "Achtste finale",
                "team": "France",
                "player": "Kylian Mbappe",
                "goal_share": 0.38,
                "points_per_goal": 24,
                "star_scorer_power": 0.86,
                "expected_goals": 0.38,
                "expected_scorito_points": 9.12,
                "recommended_stage_topscorer_score": 8.55,
                "stage_order": 2,
            },
            {
                "stage": "Round of 16",
                "stage_label": "Achtste finale",
                "team": "Belgium",
                "player": "Romelu Lukaku",
                "goal_share": 0.30,
                "points_per_goal": 24,
                "star_scorer_power": 0.97,
                "expected_goals": 0.60,
                "expected_scorito_points": 14.40,
                "recommended_stage_topscorer_score": 14.21,
                "stage_order": 2,
            },
        ]

        recomputed = publisher.recompute_stage_top_scorers(stale_rows, predictions)
        ranked = publisher.normalize_stage_top_scorers(recomputed, predictions)
        by_player = {row["player"]: row for row in ranked}

        self.assertAlmostEqual(by_player["Kylian Mbappe"]["expected_goals"], 1.74 * 0.38)
        self.assertAlmostEqual(by_player["Romelu Lukaku"]["expected_goals"], 1.46 * 0.30)
        self.assertEqual(by_player["Kylian Mbappe"]["round_rank"], 1)
        self.assertEqual(by_player["Romelu Lukaku"]["round_rank"], 2)

    def test_round_topscorers_use_actual_goals_after_match_is_played(self) -> None:
        prediction = {
            "stage": "Group Stage",
            "home_team": "France",
            "away_team": "Belgium",
            "score": "1-1",
            "expected_home_goals": 1.2,
            "expected_away_goals": 1.1,
            "actual_available": True,
            "actual_home_score": 3,
            "actual_away_score": 0,
            "actual_score": "3-0",
        }
        scorer_row = {
            "stage": "Group Stage",
            "team": "France",
            "player": "Kylian Mbappe",
            "goal_share": 0.5,
            "points_per_goal": 8,
            "star_scorer_power": 1.0,
        }

        result = publisher.recompute_stage_top_scorers([scorer_row], [prediction])[0]

        self.assertEqual(result["expected_goals"], 1.5)
        self.assertEqual(result["expected_scorito_points"], 12.0)


if __name__ == "__main__":
    unittest.main()
