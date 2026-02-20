from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass

from .docker_runner_constants import logger


@dataclass(frozen=True)
class ContainerCommandResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int


def _run(cmd: list[str], *, timeout_s: float = 300.0) -> subprocess.CompletedProcess[str]:
    logger.debug("Running command: %s", " ".join(shlex.quote(x) for x in cmd))
    try:
        return subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        logger.warning("Command not found: %s", cmd[0] if cmd else "<empty>")
        return subprocess.CompletedProcess(cmd, returncode=127, stdout="", stderr=str(exc))
    except Exception as exc:
        logger.exception("Failed to execute command: %s", " ".join(shlex.quote(x) for x in cmd))
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr=str(exc))


def _docker_available() -> bool:
    try:
        p = _run(["docker", "version"], timeout_s=10.0)
        if p.returncode != 0:
            logger.warning("Docker not available: docker version rc=%s stderr=%s", p.returncode, (p.stderr or "").strip())
        return p.returncode == 0
    except Exception:
        logger.exception("Docker not available: exception when running docker version")
        return False


def docker_available() -> bool:
    return _docker_available()


def run_container_command(*, container_id: str, command: str, timeout_s: float = 60.0) -> ContainerCommandResult:
    cmd = ["docker", "exec", "-i", container_id, "sh", "-lc", command]
    p = _run(cmd, timeout_s=timeout_s)
    return ContainerCommandResult(
        ok=p.returncode == 0,
        stdout=(p.stdout or "").strip(),
        stderr=(p.stderr or "").strip(),
        exit_code=int(p.returncode),
    )
