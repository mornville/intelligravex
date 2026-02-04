#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is intended for macOS."
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust (cargo) not found. Install from https://rustup.rs first."
  exit 1
fi

"${ROOT_DIR}/scripts/package_macos.sh"

SIDECAR="${ROOT_DIR}/frontend/src-tauri/sidecar/GravexStudio"
if [[ ! -f "${SIDECAR}" ]]; then
  echo "Sidecar wrapper missing at ${SIDECAR}"
  exit 1
fi
chmod +x "${SIDECAR}"
ARCH="$(uname -m)"
if [[ "${ARCH}" == "arm64" ]]; then
  SUFFIX="aarch64-apple-darwin"
else
  SUFFIX="x86_64-apple-darwin"
fi
cp -f "${SIDECAR}" "${SIDECAR}-${SUFFIX}"
chmod +x "${SIDECAR}-${SUFFIX}"

npm --prefix "${ROOT_DIR}/frontend" install
npm --prefix "${ROOT_DIR}/frontend" run tauri:build

APP_PATH="${ROOT_DIR}/frontend/src-tauri/target/release/bundle/macos/GravexWidget.app"
PLIST_PATH="${APP_PATH}/Contents/Info.plist"
if [[ -f "${PLIST_PATH}" ]]; then
  /usr/libexec/PlistBuddy -c "Delete :NSMicrophoneUsageDescription" "${PLIST_PATH}" >/dev/null 2>&1 || true
  /usr/libexec/PlistBuddy -c "Add :NSMicrophoneUsageDescription string \"Gravex needs microphone access to record your requests.\"" "${PLIST_PATH}"
fi

DMG_PATH="${ROOT_DIR}/dist/GravexWidget.dmg"
rm -f "${DMG_PATH}"
hdiutil create -volname "GravexWidget" -srcfolder "${APP_PATH}" -ov -format UDZO "${DMG_PATH}"

echo "Built widget app at: ${APP_PATH}"
echo "Built DMG at: ${DMG_PATH}"
