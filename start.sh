#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ts() {
  date "+%Y-%m-%d %H:%M:%S"
}

log_step() {
  echo "[$(ts)] $*"
}

pick_python() {
  # Prefer newer CPython (better SSL + wheels). Fall back to python3 if needed.
  for bin in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$bin" >/dev/null 2>&1; then
      echo "$bin"
      return 0
    fi
  done
  return 1
}

log_step "Detecting Python"
PYTHON_BIN="$(pick_python || true)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "python3 not found. Install Python 3.10+ first."
  exit 1
fi

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
PY_MAJOR="$("$PYTHON_BIN" -c 'import sys; print(sys.version_info[0])')"
PY_MINOR="$("$PYTHON_BIN" -c 'import sys; print(sys.version_info[1])')"
if [[ "$PY_MAJOR" -lt 3 || "$PY_MINOR" -lt 10 ]]; then
  echo "Python $PY_VERSION detected, but this project requires Python 3.10+."
  echo "On macOS, install a modern Python (with OpenSSL) and re-run:"
  echo "  brew install python"
  exit 1
fi

log_step "Using ${PYTHON_BIN} (Python ${PY_VERSION})"

ensure_venv() {
  local desired_mm="${PY_MAJOR}.${PY_MINOR}"

  if [[ -d ".venv" ]]; then
    if [[ ! -x ".venv/bin/python" ]]; then
      rm -rf .venv
    else
      local venv_mm
      venv_mm="$(".venv/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
      if [[ -z "${venv_mm:-}" || "$venv_mm" != "$desired_mm" ]]; then
        echo "Recreating .venv (found Python ${venv_mm:-unknown}, need ${desired_mm})"
        rm -rf .venv
      fi
    fi
  fi

  if [[ ! -d ".venv" ]]; then
    "$PYTHON_BIN" -m venv .venv
  fi
}

venv_start="$(date +%s)"
log_step "Ensuring .venv"
ensure_venv
log_step "Ensuring .venv done ($(( $(date +%s) - venv_start ))s)"

# shellcheck disable=SC1091
log_step "Activating .venv"
source .venv/bin/activate

log_step "Upgrading pip"
pip_start="$(date +%s)"
python -m pip install -U pip >/dev/null
log_step "Upgrading pip done ($(( $(date +%s) - pip_start ))s)"

log_step "Ensuring setuptools"
if ! python - <<'PY' >/dev/null 2>&1
import pkg_resources  # provided by setuptools
PY
then
  python -m pip install -U "setuptools<81" >/dev/null
fi
log_step "Ensuring setuptools done"

cmd="run"
subcmd=""
if [[ "$#" -gt 0 ]]; then
  if [[ "$1" == -* ]]; then
    cmd="run"
  else
    cmd="$1"
  fi
  if [[ "$#" -gt 1 ]]; then
    subcmd="$2"
  fi
fi

if [[ "$cmd" == "package-overlay" ]]; then
  exec "${ROOT_DIR}/scripts/package_macos_overlay.sh"
fi

if [[ "$cmd" == "package-studio" ]]; then
  exec "${ROOT_DIR}/scripts/package_macos.sh"
fi

if [[ "$cmd" == "package-all" ]]; then
  "${ROOT_DIR}/scripts/package_macos.sh"
  exec "${ROOT_DIR}/scripts/package_macos_overlay.sh"
fi

need_install=0
log_step "Checking base dependencies"
dep_start="$(date +%s)"
python - <<'PY' >/dev/null 2>&1 || need_install=1
import importlib
mods = ("numpy","sounddevice","webrtcvad","openai","pydantic","pydantic_settings","rich","typer","soundfile")
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        raise SystemExit(1) from exc
PY
log_step "Checking base dependencies done ($(( $(date +%s) - dep_start ))s)"

if [[ "$need_install" -eq 0 && "$cmd" == "web" ]]; then
  log_step "Checking web dependencies"
  web_dep_start="$(date +%s)"
  python - <<'PY' >/dev/null 2>&1 || need_install=1
import importlib
for m in ("fastapi","uvicorn","sqlmodel","jinja2","cryptography"):
    importlib.import_module(m)
PY
  log_step "Checking web dependencies done ($(( $(date +%s) - web_dep_start ))s)"
fi

if [[ "$need_install" -eq 1 ]]; then
  log_step "Installing dependencies (this may take a while on first run)..."
  install_start="$(date +%s)"
  python -m pip install -U pip setuptools wheel >/dev/null
  if [[ "$cmd" == "web" ]]; then
    python -m pip install -e ".[web]"
  else
    python -m pip install -e "."
  fi
  log_step "Installing dependencies done ($(( $(date +%s) - install_start ))s)"
fi

if [[ ! -f ".env" ]]; then
  cp .env.example .env
  log_step "Created .env from .env.example"
fi

has_help=0
for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    has_help=1
    break
  fi
done

read_openai_key_from_env_file() {
  # Don't `source .env` (dotenv is not guaranteed to be shell-compatible).
  # Prefer python-dotenv (already a dependency), fall back to a simple grep.
  python - <<'PY' 2>/dev/null || true
from __future__ import annotations
import os
try:
    from dotenv import dotenv_values
except Exception:
    dotenv_values = None
if dotenv_values is not None:
    v = dotenv_values(".env").get("OPENAI_API_KEY") or ""
    print(v)
else:
    # minimal fallback: print value after first '='
    try:
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    print(line.split("=", 1)[1].strip().strip('"').strip("'"))
                    break
    except Exception:
        pass
PY
}

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  log_step "Reading OPENAI_API_KEY from .env"
  OPENAI_API_KEY="$(read_openai_key_from_env_file)"
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    export OPENAI_API_KEY
  fi
fi

if [[ "$has_help" -eq 0 && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set (OK if you run with --bot and a stored key)."
fi

if [[ "$has_help" -eq 0 && "$cmd" == "run" ]]; then
  log_step "Running diagnostics..."
  python -m voicebot doctor || true
  log_step "Starting voice bot. Use Ctrl+C to stop."
fi

if [[ "$#" -eq 0 ]]; then
  log_step "Launching: python -m voicebot run"
  exec python -m voicebot run
else
  log_step "Launching: python -m voicebot $*"
  exec python -m voicebot "$@"
fi
