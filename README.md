# Voetbal Prediction

WK 2026 prediction project met:

- XGBoost training voor interlanduitslagen.
- OddsPortal WK 2026 fixture/odds scraper.
- Monte Carlo simulatie voor groepsfase, knockout-route en kampioenskansen.
- Scorito invuladvies met scoretweaks.
- Next.js dashboard in `github_vercel_app`.
- GitHub Actions workflows voor handmatige rebuilds en Vercel deploys.

## Dashboard

De live app draait op:

https://githubvercelapp.vercel.app

Lokaal:

```powershell
cd github_vercel_app
npm install
npm run dev
```

## Rebuild

Default model rebuild zonder live deploy:

```powershell
.\github_vercel_app\scripts\rebuild_and_deploy.ps1 -PreviousLabel 20260610
```

Rebuild met Vercel deploy:

```powershell
.\github_vercel_app\scripts\rebuild_and_deploy.ps1 -PreviousLabel 20260610 -Deploy
```

## GitHub Actions

Workflows staan in `.github/workflows`.

- `rebuild-predictions.yml`: handmatige model/odds/rebuild flow.
- `deploy-dashboard.yml`: handmatige deploy van alleen de dashboardbestanden.

Benodigde GitHub secrets voor Vercel deploy:

```text
VERCEL_TOKEN
VERCEL_ORG_ID=team_jh9eoFaaEJ7cf2PpE3kDcIT7
VERCEL_PROJECT_ID=prj_QGNrk1fttELfP2oKVfE8BalQbm5v
```

Grote lokale datasets, modellen, virtualenvs en build outputs staan bewust in `.gitignore`.
