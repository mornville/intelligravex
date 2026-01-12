from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PythonPostprocessResult:
    ok: bool
    result_text: str
    metadata_patch: dict[str, Any] | None
    stdout: str
    stderr: str
    exit_code: int | None
    error: str | None
    duration_ms: int


def _truncate(s: str, limit: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= limit:
        return s
    return s[:limit] + "…"


def run_python_postprocessor(
    *,
    python_code: str,
    payload: dict[str, Any],
    timeout_s: int = 60,
    max_output_chars: int = 500_000,
) -> PythonPostprocessResult:
    code = (python_code or "").strip("\n")
    code = textwrap.dedent(code).strip()
    if not code:
        return PythonPostprocessResult(
            ok=False,
            result_text="",
            metadata_patch=None,
            stdout="",
            stderr="",
            exit_code=None,
            error="empty postprocess_python",
            duration_ms=0,
        )

    start = time.time()
    input_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    # Give tool authors a convenient, consistent context without boilerplate.
    wrapped = (
        "import json, sys\n"
        "_payload = json.load(sys.stdin)\n"
        "response_json = _payload.get('response_json')\n"
        "meta = _payload.get('meta')\n"
        "args = _payload.get('args')\n"
        "fields_required = _payload.get('fields_required') or ''\n"
        "why_api_was_called = _payload.get('why_api_was_called') or ''\n"
        "\n"
        "def emit(result_text, metadata_patch=None, **extras):\n"
        "    out = {'result_text': '' if result_text is None else str(result_text)}\n"
        "    if metadata_patch is not None:\n"
        "        out['metadata_patch'] = metadata_patch\n"
        "    if extras:\n"
        "        out.update(extras)\n"
        "    sys.stdout.write(json.dumps(out, ensure_ascii=False))\n"
        "    sys.exit(0)\n"
        "\n"
        + code
        + "\n"
        "if 'result_text' in globals():\n"
        "    emit(globals().get('result_text'))\n"
        "if 'result' in globals():\n"
        "    _r = globals().get('result')\n"
        "    if isinstance(_r, dict):\n"
        "        sys.stdout.write(json.dumps(_r, ensure_ascii=False))\n"
        "        sys.exit(0)\n"
        "    emit(_r)\n"
        "sys.stderr.write('No output: call emit(...) or set result_text/result')\n"
        "sys.exit(2)\n"
    )

    # Best-effort resource limits (POSIX only).
    def _limit_resources() -> None:
        try:
            import resource  # type: ignore

            # CPU seconds (hard cap); wall time is enforced via subprocess timeout.
            cpu_soft = max(1, int(timeout_s) - 2)
            cpu_hard = int(timeout_s)
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft, cpu_hard))
            except Exception:
                pass

            # Address space (bytes). Keep generous to avoid breaking moderate payloads.
            try:
                resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
            except Exception:
                pass

            # Max file size (bytes) to discourage writing large artifacts.
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
            except Exception:
                pass
        except Exception:
            return

    env = {
        # Keep env tight; do not pass through secrets by default.
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PATH": os.environ.get("PATH", ""),
    }

    with tempfile.TemporaryDirectory(prefix="igx_tool_py_") as td:
        script_path = os.path.join(td, "postprocess.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(wrapped)
        try:
            # -I: isolated mode (no user site, no env vars like PYTHONPATH)
            # -S: don't import site on startup
            proc = subprocess.run(
                [sys.executable, "-I", "-S", script_path],
                input=input_bytes,
                cwd=td,
                env=env,
                capture_output=True,
                timeout=float(timeout_s),
                check=False,
                preexec_fn=_limit_resources if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            dur_ms = int(round((time.time() - start) * 1000.0))
            return PythonPostprocessResult(
                ok=False,
                result_text="",
                metadata_patch=None,
                stdout="",
                stderr="",
                exit_code=None,
                error=f"python postprocessor timed out after {timeout_s}s",
                duration_ms=dur_ms,
            )
        except Exception as exc:
            dur_ms = int(round((time.time() - start) * 1000.0))
            return PythonPostprocessResult(
                ok=False,
                result_text="",
                metadata_patch=None,
                stdout="",
                stderr="",
                exit_code=None,
                error=str(exc),
                duration_ms=dur_ms,
            )

    dur_ms = int(round((time.time() - start) * 1000.0))
    stdout = _truncate((proc.stdout or b"").decode("utf-8", errors="replace"), max_output_chars)
    stderr = _truncate((proc.stderr or b"").decode("utf-8", errors="replace"), max_output_chars)

    if proc.returncode != 0:
        return PythonPostprocessResult(
            ok=False,
            result_text="",
            metadata_patch=None,
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode,
            error=f"python postprocessor exited with code {proc.returncode}",
            duration_ms=dur_ms,
        )

    # Accept either JSON object output or plain text output.
    result_text = ""
    metadata_patch: dict[str, Any] | None = None
    try:
        obj = json.loads(stdout) if stdout.strip() else None
        if isinstance(obj, dict):
            rt = obj.get("result_text")
            if rt is not None:
                result_text = str(rt)
            mp = obj.get("metadata_patch")
            if isinstance(mp, dict):
                metadata_patch = mp
            elif mp is not None:
                # Non-dict patches are ignored for safety.
                metadata_patch = None
        else:
            result_text = stdout.strip()
    except Exception:
        result_text = stdout.strip()

    if not result_text and not metadata_patch:
        return PythonPostprocessResult(
            ok=False,
            result_text="",
            metadata_patch=None,
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode,
            error="python postprocessor produced no output",
            duration_ms=dur_ms,
        )

    return PythonPostprocessResult(
        ok=True,
        result_text=result_text,
        metadata_patch=metadata_patch,
        stdout=stdout,
        stderr=stderr,
        exit_code=proc.returncode,
        error=None,
        duration_ms=dur_ms,
    )
