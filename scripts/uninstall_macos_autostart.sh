#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This uninstaller is for macOS only."
  exit 1
fi

LABEL="com.intelligravex.voicebot.web"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
UID_NUM="$(id -u)"

if [[ -f "${PLIST_PATH}" ]]; then
  launchctl bootout "gui/${UID_NUM}" "${PLIST_PATH}" >/dev/null 2>&1 || true
  rm -f "${PLIST_PATH}"
fi

echo "Removed autostart job: ${LABEL}"

