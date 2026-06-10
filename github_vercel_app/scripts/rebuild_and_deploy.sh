#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/rebuild_predictions.sh"

if [[ "${DEPLOY:-0}" == "1" ]]; then
  PREBUILT="${PREBUILT_DEPLOY:-0}" "$SCRIPT_DIR/deploy_vercel.sh"
else
  echo "Rebuild complete. Live Vercel site was not changed because DEPLOY=1 was not set."
fi
