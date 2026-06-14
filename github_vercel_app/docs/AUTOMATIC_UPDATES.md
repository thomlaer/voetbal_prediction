# Automatic Updates

Current status:

- The live Vercel site can start a rebuild through the `Update Starten` panel after the Vercel secrets below are configured.
- The dashboard is static. It changes only after a rebuild publishes new files into `github_vercel_app/public/data` and `github_vercel_app/public/files`, followed by a Vercel deploy.
- The current live project is `github_vercel_app` on team `team_jh9eoFaaEJ7cf2PpE3kDcIT7`.
- The current project id is `prj_QGNrk1fttELfP2oKVfE8BalQbm5v`.

Recommended mode for the WK:

1. Keep automatic cron disabled.
2. Run the rebuild workflow manually after group stage, after round of 32, after round of 16, and when odds/injury data materially changes.
3. Use `model_profile=best_default` unless we deliberately retest `full_with_lineups`.
4. Keep `update_soccerbase=true` for normal updates; it runs incrementally and should only fetch recent completed missing matches.
5. Turn `update_soccerbase=false` only when you need the fastest possible run and are sure no new Soccerbase data matters.
6. Set `deploy_to_vercel=true` only when the new output should replace the live site.

Required GitHub secrets:

```text
VERCEL_TOKEN=your Vercel token
VERCEL_ORG_ID=team_jh9eoFaaEJ7cf2PpE3kDcIT7
VERCEL_PROJECT_ID=prj_QGNrk1fttELfP2oKVfE8BalQbm5v
```

Required Vercel environment variables for the website update button:

```text
REBUILD_ACTION_CODE=your private dashboard code
GITHUB_ACTIONS_TOKEN=GitHub fine-grained token with Actions write access for thomlaer/voetbal_prediction
GITHUB_ACTIONS_REPO=thomlaer/voetbal_prediction
GITHUB_ACTIONS_WORKFLOW=rebuild-predictions.yml
GITHUB_ACTIONS_REF=main
```

The website button sends a GitHub `workflow_dispatch` request to `.github/workflows/rebuild-predictions.yml`.
It does not expose `GITHUB_ACTIONS_TOKEN` to the browser.

Workflows:

- `.github/workflows/rebuild-predictions.yml`
  - Manual run.
  - Updates results/odds.
  - Rebuilds the model.
  - Publishes latest dashboard files.
  - Commits the changed public files.
  - Deploys to Vercel only when `deploy_to_vercel=true`.

- `.github/workflows/deploy-dashboard.yml`
  - Manual run.
  - Does not retrain the model.
  - Deploys the current dashboard files to Vercel.

Local commands:

```powershell
# Rebuild locally, do not change live Vercel site.
cd C:\Users\thoml\OneDrive\Documenten\voetbal_prediction
.\github_vercel_app\scripts\rebuild_and_deploy.ps1 -PreviousLabel 20260610

# Rebuild locally and deploy to Vercel.
cd C:\Users\thoml\OneDrive\Documenten\voetbal_prediction
.\github_vercel_app\scripts\rebuild_and_deploy.ps1 -PreviousLabel 20260610 -Deploy

# Deploy current dashboard without retraining.
cd C:\Users\thoml\OneDrive\Documenten\voetbal_prediction\github_vercel_app
.\scripts\deploy_vercel.ps1
```

If you later want a scheduled run, add this to `rebuild-predictions.yml` under `on:`:

```yaml
  schedule:
    - cron: "0 8 * * *"
```

That would run around 10:00 in the Netherlands during summer time. I would not enable this by default, because the heavy model rebuild and scrapers are more useful after tournament checkpoints than every day.

If you later want every push to deploy automatically, add this to `deploy-dashboard.yml` under `on:`:

```yaml
  push:
    branches:
      - main
    paths:
      - "github_vercel_app/src/**"
      - "github_vercel_app/public/**"
      - "github_vercel_app/package.json"
      - "github_vercel_app/package-lock.json"
      - "github_vercel_app/next.config.mjs"
      - "github_vercel_app/tsconfig.json"
```
