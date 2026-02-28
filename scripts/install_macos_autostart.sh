#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer is for macOS only."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.intelligravex.voicebot.web"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
PLIST_PATH="${LAUNCH_AGENTS_DIR}/${LABEL}.plist"
LOG_DIR="${HOME}/.intelligravex/logs"
OUT_LOG="${LOG_DIR}/voicebot-web.out.log"
ERR_LOG="${LOG_DIR}/voicebot-web.err.log"
PATH_VALUE="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
UID_NUM="$(id -u)"

mkdir -p "${LAUNCH_AGENTS_DIR}" "${LOG_DIR}"

cat >"${PLIST_PATH}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>${LABEL}</string>

    <key>ProgramArguments</key>
    <array>
      <string>/bin/bash</string>
      <string>-lc</string>
      <string>cd "${ROOT_DIR}" && ./start.sh web -v</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${ROOT_DIR}</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>EnvironmentVariables</key>
    <dict>
      <key>PATH</key>
      <string>${PATH_VALUE}</string>
    </dict>

    <key>StandardOutPath</key>
    <string>${OUT_LOG}</string>

    <key>StandardErrorPath</key>
    <string>${ERR_LOG}</string>
  </dict>
</plist>
PLIST

launchctl bootout "gui/${UID_NUM}" "${PLIST_PATH}" >/dev/null 2>&1 || true
launchctl bootstrap "gui/${UID_NUM}" "${PLIST_PATH}"
launchctl enable "gui/${UID_NUM}/${LABEL}"
launchctl kickstart -k "gui/${UID_NUM}/${LABEL}"

echo "Installed autostart job: ${LABEL}"
echo "Plist: ${PLIST_PATH}"
echo "Logs:"
echo "  ${OUT_LOG}"
echo "  ${ERR_LOG}"
echo "Check status:"
echo "  launchctl print gui/${UID_NUM}/${LABEL}"

