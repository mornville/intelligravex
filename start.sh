#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

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

echo "Using ${PYTHON_BIN} (Python ${PY_VERSION})"

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

ensure_venv

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip >/dev/null

cmd="run"
if [[ "$#" -gt 0 ]]; then
  if [[ "$1" == -* ]]; then
    cmd="run"
  else
    cmd="$1"
  fi
fi

need_install=0
python - <<'PY' >/dev/null 2>&1 || need_install=1
import importlib
mods = ("numpy","sounddevice","webrtcvad","openai","pydantic","pydantic_settings","rich","typer","whisper","TTS")
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        raise SystemExit(1) from exc
PY

if [[ "$need_install" -eq 0 && "$cmd" == "web" ]]; then
  python - <<'PY' >/dev/null 2>&1 || need_install=1
import importlib
for m in ("fastapi","uvicorn","sqlmodel","jinja2","cryptography"):
    importlib.import_module(m)
PY
fi

if [[ "$need_install" -eq 1 ]]; then
  echo "Installing dependencies (this may take a while on first run)..."
  python -m pip install -U pip setuptools wheel >/dev/null
  if [[ "$cmd" == "web" ]]; then
    python -m pip install -e ".[asr,tts,web]"
  else
    python -m pip install -e ".[asr,tts]"
  fi
fi

if [[ ! -f ".env" ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
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
  OPENAI_API_KEY="$(read_openai_key_from_env_file)"
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    export OPENAI_API_KEY
  fi
fi

if [[ "$has_help" -eq 0 && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set (OK if you run with --bot and a stored key)."
fi

if [[ "$has_help" -eq 0 && "$cmd" == "run" ]]; then
  echo "Running diagnostics..."
  python -m voicebot doctor || true
  echo "Starting voice bot. Use Ctrl+C to stop."
fi

if [[ "$#" -eq 0 ]]; then
  exec python -m voicebot run
else
  exec python -m voicebot "$@"
fi
