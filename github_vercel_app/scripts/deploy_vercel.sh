#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

PROD_FLAG=()
if [[ "${PRODUCTION:-1}" == "1" ]]; then
  PROD_FLAG=(--prod)
fi

TOKEN_FLAG=()
if [[ -n "${VERCEL_TOKEN:-}" ]]; then
  TOKEN_FLAG=(--token "$VERCEL_TOKEN")
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  if [[ "${PREBUILT:-0}" == "1" ]]; then
    npx vercel build "${PROD_FLAG[@]}" "${TOKEN_FLAG[@]}"
  else
    npm run build
  fi
fi

DEPLOY_ARGS=(vercel deploy --yes "${PROD_FLAG[@]}")
if [[ "${PREBUILT:-0}" == "1" ]]; then
  DEPLOY_ARGS+=(--prebuilt)
fi
DEPLOY_ARGS+=("${TOKEN_FLAG[@]}")

npx "${DEPLOY_ARGS[@]}"
