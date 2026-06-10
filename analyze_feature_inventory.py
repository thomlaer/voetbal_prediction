"""
Create an inventory of model features from saved model payloads.

This is useful because the default model and optional feature-layer experiments
do not have the exact same feature columns.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATHS = {
    "standard_v2": Path("models_feature_v2_goalscorers_squads_no_sample_weights/worldcup_xgboost_model.joblib"),
    "with_oddsportal": Path("models_feature_v2_oddsportal_plus_goalscorers_squads_no_sample_weights/worldcup_xgboost_model.joblib"),
    "with_squad_market": Path("models_feature_v2_goalscorers_squads_market/worldcup_xgboost_model.joblib"),
    "with_soccerbase_starters_no_ratings": Path("models_feature_v2_soccerbase_starters_no_ratings/worldcup_xgboost_model.joblib"),
    "with_soccerbase_starters": Path("models_feature_v2_soccerbase_starters_ratings/worldcup_xgboost_model.joblib"),
    "with_soccerbase_starters_yearly": Path("models_feature_v2_soccerbase_starters_yearly_ratings/worldcup_xgboost_model.joblib"),
    "with_soccerbase_stats": Path("models_feature_v2_soccerbase_stats/worldcup_xgboost_model.joblib"),
    "with_soccerbase_stats_cards": Path("models_feature_v2_soccerbase_stats_cards/worldcup_xgboost_model.joblib"),
    "with_soccerbase_stats_lineup_shapes": Path("models_feature_v2_soccerbase_stats_lineup_shapes/worldcup_xgboost_model.joblib"),
    "with_soccerbase_stats_cards_lineup_shapes": Path("models_feature_v2_soccerbase_stats_cards_lineup_shapes/worldcup_xgboost_model.joblib"),
    "with_external_elo": Path("models_feature_v2_external_elo/worldcup_xgboost_model.joblib"),
    "with_external_elo_soccerbase_stats": Path("models_feature_v2_external_elo_soccerbase_stats/worldcup_xgboost_model.joblib"),
    "with_soccerbase_stats_lineups_yearly": Path("models_feature_v2_soccerbase_stats_lineups_yearly_ratings/worldcup_xgboost_model.joblib"),
    "with_btb_odds": Path("models_feature_v2_odds_btb/worldcup_xgboost_model.joblib"),
    "with_xfkz_perf": Path("models_feature_v3_xfkz_perf/worldcup_xgboost_model.joblib"),
    "with_geo": Path("models_feature_v2_geo/worldcup_xgboost_model.joblib"),
}
OUTPUT_PATH = Path("outputs/feature_inventory_union.csv")

KNOWN_OPTIONAL_NUMERIC_FEATURES = {
    "odds_home_win",
    "odds_draw",
    "odds_away_win",
    "odds_prob_home_win",
    "odds_prob_draw",
    "odds_prob_away_win",
    "odds_has_1x2",
    "odds_overround",
    "odds_home_draw_gap",
    "odds_home_away_gap",
    "odds_favorite_prob",
    "odds_favorite_is_home",
    "odds_favorite_is_draw",
    "odds_favorite_is_away",
    "odds_max_home_win",
    "odds_max_draw",
    "odds_max_away_win",
    "odds_max_overround",
    "odds_n_home_win",
    "odds_n_draw",
    "odds_n_away_win",
    "odds_max_prob_home_win",
    "odds_max_prob_draw",
    "odds_max_prob_away_win",
    "odds_home_win_max_gap",
    "odds_draw_max_gap",
    "odds_away_win_max_gap",
}


def feature_group(name: str) -> str:
    if name in {"tournament", "city", "country"}:
        return "categorical_match_context"
    if name.startswith("odds_"):
        return "odds_optional"
    if "external_elo" in name:
        return "external_elo_optional"
    if name.startswith("xfkz_") or "_xfkz_" in name:
        return "xfkz_optional"
    if "squad_" in name:
        return "tournament_squad"
    if "sb_stats_" in name:
        return "soccerbase_rolling_stats"
    if "sb_cards_" in name:
        return "soccerbase_rolling_cards"
    if "lineup_" in name:
        return "soccerbase_lineup"
    if (
        "scorer_" in name
        or "unique_scorers" in name
        or "penalty_goal" in name
        or "top_scorer" in name
        or "top3_scorer" in name
        or "top5_scorer" in name
    ):
        return "goalscorer_form"
    if (
        name.startswith("venue_")
        or name.endswith("_travel_km")
        or "travel_km" in name
        or "tz_" in name
        or "utc_offset" in name
        or name.endswith("_base_geo_available")
    ):
        return "geo_optional"
    if "player_" in name:
        return "transfermarkt_player_form"
    if "manager" in name:
        return "fjelstul_manager"
    if "tournament_" in name:
        return "live_tournament_state"
    if "fifa_" in name:
        return "fifa_ranking"
    if "confederation" in name:
        return "confederation_context"
    if "elo" in name:
        return "elo"
    if "h2h" in name:
        return "head_to_head"
    if "recent" in name or "form" in name:
        return "recent_team_form"
    if name.startswith(("home_", "away_")):
        return "team_history"
    if name.endswith("_diff") or name.endswith("_ratio"):
        return "matchup_difference"
    return "base_match_context"


def load_features(model_path: Path) -> tuple[set[str], set[str]]:
    if not model_path.exists():
        return set(), set()
    payload = joblib.load(model_path)
    numeric = set(payload.get("numeric_features", []))
    categorical = set(payload.get("categorical_features", []))
    return numeric, categorical


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, set[str]] = {}
    types: dict[str, str] = {}

    for model_name, model_path in MODEL_PATHS.items():
        numeric, categorical = load_features(model_path)
        by_model[model_name] = numeric | categorical
        for feature in numeric:
            types.setdefault(feature, "numeric")
        for feature in categorical:
            types[feature] = "categorical"

    all_features = sorted(set().union(*by_model.values()))
    all_features = sorted(set(all_features) | KNOWN_OPTIONAL_NUMERIC_FEATURES)
    rows = []
    for feature in all_features:
        if feature in KNOWN_OPTIONAL_NUMERIC_FEATURES:
            types.setdefault(feature, "numeric")
        rows.append(
            {
                "feature": feature,
                "type": types.get(feature, "unknown"),
                "group": feature_group(feature),
                **{f"in_{model_name}": feature in features for model_name, features in by_model.items()},
            }
        )

    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(output):,} feature rows to {OUTPUT_PATH}")
    print(output.groupby("group").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
