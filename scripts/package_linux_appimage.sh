#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${ROOT_DIR}/.build/venv-linux"
APPDIR="${ROOT_DIR}/.build/AppDir"
BIN_NAME="GravexStudio"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is intended for Linux."
  exit 1
fi

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

"${ROOT_DIR}/scripts/build_ui.sh"

mkdir -p "${ROOT_DIR}/.build"
rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install "${ROOT_DIR}[web,packaging]"

pyinstaller --noconfirm --clean \
  --name "${BIN_NAME}" \
  --collect-all voicebot.web \
  --collect-all tiktoken \
  --copy-metadata tiktoken \
  --add-data "${ROOT_DIR}/voicebot/web/ui:voicebot/web/ui" \
  --add-data "${ROOT_DIR}/voicebot/web/static:voicebot/web/static" \
  "${ROOT_DIR}/voicebot/launcher.py"

rm -rf "${APPDIR}"
mkdir -p "${APPDIR}/usr/bin" "${APPDIR}/usr/share/icons/hicolor/256x256/apps"
cp "${ROOT_DIR}/dist/${BIN_NAME}/${BIN_NAME}" "${APPDIR}/usr/bin/${BIN_NAME}"
cp "${ROOT_DIR}/packaging/icon.png" "${APPDIR}/gravexstudio.png"
cp "${ROOT_DIR}/packaging/icon.png" "${APPDIR}/usr/share/icons/hicolor/256x256/apps/gravexstudio.png"

cat <<'DESKTOP' > "${APPDIR}/gravexstudio.desktop"
[Desktop Entry]
Type=Application
Name=GravexStudio
Exec=GravexStudio
Icon=gravexstudio
Categories=Utility;
Terminal=false
DESKTOP

cat <<'APPRUN' > "${APPDIR}/AppRun"
#!/usr/bin/env sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/GravexStudio" "$@"
APPRUN
chmod +x "${APPDIR}/AppRun"

APPIMAGETOOL="${ROOT_DIR}/.build/appimagetool"
if [[ ! -x "${APPIMAGETOOL}" ]]; then
  echo "Downloading appimagetool..."
  curl -L "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage" -o "${APPIMAGETOOL}"
  chmod +x "${APPIMAGETOOL}"
fi

"${APPIMAGETOOL}" "${APPDIR}" "${ROOT_DIR}/dist/GravexStudio-x86_64.AppImage"

echo "Built AppImage: dist/GravexStudio-x86_64.AppImage"
