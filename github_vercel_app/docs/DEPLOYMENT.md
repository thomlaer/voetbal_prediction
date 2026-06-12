# Deployment Notes

This app is only the mobile dashboard. The XGBoost model should not train on Vercel.

Recommended setup:

1. Put the full `voetbal_prediction` folder in a GitHub repository.
2. Keep `github_vercel_app` as the Vercel project root.
3. Use GitHub Actions to rebuild predictions and commit `github_vercel_app/public/data` and `github_vercel_app/public/files`.
4. Vercel redeploys the dashboard after the commit.

See `docs/AUTOMATIC_UPDATES.md` for the prepared manual deploy workflow, optional rebuild+deploy flow, and required Vercel secrets.

The workflow in `.github/workflows/rebuild-predictions.yml` is stored here so it can be copied to the repository root if GitHub does not detect it. GitHub only runs workflows from the repository root `.github/workflows` directory.

Default model parity:

- Uses `train_xgboost_worldcup.py`.
- Uses Soccerbase stat features.
- Uses Soccerbase card features.
- Downloads/prepares the Transfermarkt `davidcariboo/player-scores` Kaggle data, because the current model uses player-form features and the raw files are too large for normal Git commits.
- `model_profile=best_default` matches the current best `20260610` profile: stats/cards/player-form/squads/goalscorer features on, exact lineup features off.
- `model_profile=full_with_lineups` is the information-heavy experiment: exact Soccerbase lineup features, lineup ratings, yearly SoFIFA ratings, squad market/SoFIFA ratings, stat-recency features, geo features, external Elo, and historical odds when present.
- `update_soccerbase=false` is the workflow default to save GitHub Actions minutes. Normal rebuilds reuse the tracked Soccerbase stats/cards CSVs for training. When enabled manually, the Soccerbase extractor runs in incremental mode: it preserves existing CSV rows and only fetches Soccerbase game ids that are not present yet. The full lineup CSV is not committed because it is too large for normal GitHub storage.

Lineup note:

- Historical Soccerbase lineups can train the model.
- Future WK fixture lineups are only used when the specific future match lineup exists in `soccerbase_lineups_used.csv`.
- For normal pre-tournament predictions, confirmed future lineups are not known yet. Use `full_with_lineups` after lineups become available or as an experiment, not as the locked default unless its validation improves.

Manual local publish without retraining:

```powershell
cd C:\Users\thoml\OneDrive\Documenten\voetbal_prediction\github_vercel_app
npm run publish:data
npm run dev
```

Full local rebuild:

```powershell
cd C:\Users\thoml\OneDrive\Documenten\voetbal_prediction
.\github_vercel_app\scripts\rebuild_predictions.ps1 -PreviousLabel 20260610
```

Full local rebuild with lineup experiment:

```powershell
cd C:\Users\thoml\OneDrive\Documenten\voetbal_prediction
powershell.exe -ExecutionPolicy Bypass -File .\github_vercel_app\scripts\rebuild_predictions.ps1 -PreviousLabel 20260610 -ModelProfile full_with_lineups -UpdateSoccerbase
```
