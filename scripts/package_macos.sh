#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${ROOT_DIR}/.build/venv-macos"

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

mkdir -p "${ROOT_DIR}/.build"
rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install "${ROOT_DIR}[web,packaging]"

pyinstaller --noconfirm --clean \
  --name "GravexStudio" \
  --windowed \
  --collect-all voicebot.web \
  --collect-all tiktoken \
  --copy-metadata tiktoken \
  --add-data "${ROOT_DIR}/voicebot/web/ui:voicebot/web/ui" \
  --add-data "${ROOT_DIR}/voicebot/web/static:voicebot/web/static" \
  "${ROOT_DIR}/voicebot/launcher.py"

echo "Built macOS app: dist/GravexStudio.app"
