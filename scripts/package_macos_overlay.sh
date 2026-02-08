#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${ROOT_DIR}/.build/venv-macos"
BUILD_DIR="${ROOT_DIR}/.build/overlay"
DIST_DIR="${ROOT_DIR}/dist"
MARKETING_DIR="${ROOT_DIR}/marketing/download"
APP_NAME="GravexOverlay"
APP_DIR="${DIST_DIR}/${APP_NAME}.app"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is intended for macOS."
  exit 1
fi

"${ROOT_DIR}/scripts/build_ui.sh"

if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      if "${candidate}" - <<'PY' >/dev/null 2>&1
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
      then
        PYTHON_BIN="${candidate}"
        break
      fi
    fi
  done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Python 3.10+ is required. Set PYTHON_BIN to a compatible interpreter."
  exit 1
fi

echo "Using Python: ${PYTHON_BIN}"

mkdir -p "${ROOT_DIR}/.build" "${DIST_DIR}" "${BUILD_DIR}"
rm -rf "${VENV_DIR}" "${BUILD_DIR}/pyinstaller"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install "${ROOT_DIR}[web,packaging]"

pyinstaller --noconfirm --clean --onefile \
  --name "GravexServer" \
  --collect-all voicebot.web \
  --collect-all tiktoken \
  --copy-metadata tiktoken \
  --add-data "${ROOT_DIR}/voicebot/web/ui:voicebot/web/ui" \
  --add-data "${ROOT_DIR}/voicebot/web/static:voicebot/web/static" \
  --distpath "${BUILD_DIR}/pyinstaller" \
  "${ROOT_DIR}/voicebot/launcher.py"

SERVER_BIN="${BUILD_DIR}/pyinstaller/GravexServer"
if [[ ! -f "${SERVER_BIN}" ]]; then
  echo "Failed to build GravexServer binary."
  exit 1
fi

OVERLAY_BIN="${BUILD_DIR}/${APP_NAME}"
SDK_PATH="$(xcrun --sdk macosx --show-sdk-path)"
MACOSX_DEPLOYMENT_TARGET=11.0 xcrun --sdk macosx clang \
  -O2 \
  -fobjc-arc \
  -framework Cocoa \
  -framework WebKit \
  -framework Carbon \
  -isysroot "${SDK_PATH}" \
  -o "${OVERLAY_BIN}" \
  "${ROOT_DIR}/macos/overlay/main.m"

rm -rf "${APP_DIR}"
mkdir -p "${APP_DIR}/Contents/MacOS" "${APP_DIR}/Contents/Resources"
cp "${ROOT_DIR}/macos/overlay/Info.plist" "${APP_DIR}/Contents/Info.plist"
if [[ -f "${ROOT_DIR}/macos/overlay/GravexOverlay.icns" ]]; then
  cp "${ROOT_DIR}/macos/overlay/GravexOverlay.icns" "${APP_DIR}/Contents/Resources/GravexOverlay.icns"
fi
cp "${OVERLAY_BIN}" "${APP_DIR}/Contents/MacOS/${APP_NAME}"
cp "${SERVER_BIN}" "${APP_DIR}/Contents/Resources/GravexServer"
chmod +x "${APP_DIR}/Contents/MacOS/${APP_NAME}" "${APP_DIR}/Contents/Resources/GravexServer"

if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - "${APP_DIR}"
fi

# Reset Screen Recording permission for the overlay bundle so macOS prompts again on next launch.
if command -v tccutil >/dev/null 2>&1; then
  tccutil reset ScreenCapture com.gravex.overlay || true
fi

DMG_PATH="${DIST_DIR}/${APP_NAME}.dmg"
rm -f "${DMG_PATH}"
hdiutil create -volname "${APP_NAME}" -srcfolder "${APP_DIR}" -ov -format UDZO "${DMG_PATH}"

if [[ -d "${MARKETING_DIR}" ]]; then
  mkdir -p "${MARKETING_DIR}"
  cp -f "${DMG_PATH}" "${MARKETING_DIR}/${APP_NAME}.dmg"
fi

echo "Built macOS app: ${APP_DIR}"
echo "Built macOS DMG: ${DMG_PATH}"
