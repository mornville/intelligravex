#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="${ROOT_DIR}/frontend"
OUT_DIR="${ROOT_DIR}/voicebot/web/ui"

if [[ ! -d "${FRONTEND_DIR}" ]]; then
  echo "frontend/ not found."
  exit 1
fi

echo "Building Studio UI..."
cd "${FRONTEND_DIR}"
if [[ -f package-lock.json ]]; then
  npm ci
else
  npm install
fi
npm run build

echo "Syncing UI build to ${OUT_DIR}"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
cp -R "${FRONTEND_DIR}/dist/." "${OUT_DIR}/"

echo "Done."
