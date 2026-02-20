from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional
from uuid import UUID

from .docker_runner_auth import _extract_preferred_repo, _normalize_host_path
from .docker_runner_constants import DEFAULT_DATA_AGENT_IMAGE, logger
from .docker_runner_exec import _docker_available, _run
from .docker_runner_names import _container_name_for_conversation, _conversation_id_from_container_name
from .docker_runner_ports import (
    assign_container_id_to_ports,
    release_container_ports,
    reserve_container_ports,
    sync_container_ports_from_docker,
)

def _find_repo_root() -> Path:
    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / "packaging" / "data-agent" / "Dockerfile").exists():
            return parent
    return candidate.parents[2]


_ROOT_DIR = _find_repo_root()
_DATA_AGENT_DIR = _ROOT_DIR / "packaging" / "data-agent"
_DATA_AGENT_DOCKERFILE = _DATA_AGENT_DIR / "Dockerfile"


def list_data_agent_containers() -> dict:
    if not _docker_available():
        return {"docker_available": False, "items": []}
    p = _run(
        ["docker", "ps", "--filter", "name=^igx-data-agent-", "--format", "{{json .}}"],
        timeout_s=10.0,
    )
    if p.returncode != 0:
        logger.warning("docker ps failed rc=%s stderr=%s", p.returncode, (p.stderr or "").strip())
        return {"docker_available": True, "items": [], "error": (p.stderr or "").strip()}
    items: list[dict] = []
    ids: list[str] = []
    for line in (p.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        name = str(data.get("Names") or "")
        cid = str(data.get("ID") or "")
        if cid:
            ids.append(cid)
        items.append(
            {
                "id": cid,
                "name": name,
                "image": str(data.get("Image") or ""),
                "status": str(data.get("Status") or ""),
                "created_at": str(data.get("CreatedAt") or ""),
                "running_for": str(data.get("RunningFor") or ""),
                "conversation_id": _conversation_id_from_container_name(name),
            }
        )
    stats_map: dict[str, dict] = {}
    if ids:
        stats = _run(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}"] + ids,
            timeout_s=10.0,
        )
        if stats.returncode == 0:
            for line in (stats.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                except Exception:
                    continue
                sid = str(s.get("ID") or "")
                if not sid:
                    continue
                stats_map[sid] = {
                    "cpu": str(s.get("CPUPerc") or ""),
                    "mem": str(s.get("MemUsage") or ""),
                    "mem_perc": str(s.get("MemPerc") or ""),
                }
    for item in items:
        stat = stats_map.get(item.get("id") or "")
        if stat:
            item.update(stat)
    return {"docker_available": True, "items": items}


def stop_data_agent_container(container_id_or_name: str) -> dict:
    if not _docker_available():
        return {"docker_available": False, "stopped": False, "error": "Docker is not available"}
    target = (container_id_or_name or "").strip()
    if not target:
        return {"docker_available": True, "stopped": False, "error": "Missing container id"}
    container_name = ""
    try:
        inspect = _run(["docker", "inspect", "-f", "{{.Name}}", target], timeout_s=5.0)
        if inspect.returncode == 0:
            container_name = (inspect.stdout or "").strip().lstrip("/")
    except Exception:
        container_name = ""
    p = _run(["docker", "rm", "-f", target], timeout_s=20.0)
    if p.returncode != 0:
        return {
            "docker_available": True,
            "stopped": False,
            "error": (p.stderr or p.stdout or "").strip(),
        }
    try:
        release_container_ports(container_id=target, container_name=container_name)
    except Exception:
        logger.debug("Failed to release Isolated Workspace ports for %s", target, exc_info=True)
    return {"docker_available": True, "stopped": True}


def _get_data_agent_image() -> str:
    return (
        os.environ.get("IGX_DATA_AGENT_IMAGE")
        or os.environ.get("VOICEBOT_DATA_AGENT_IMAGE")
        or DEFAULT_DATA_AGENT_IMAGE
    )


def ensure_image_pulled() -> str:
    if not _docker_available():
        raise RuntimeError("Docker is not available (cannot start Isolated Workspace runtime).")
    image = _get_data_agent_image().strip() or DEFAULT_DATA_AGENT_IMAGE
    p = _run(["docker", "image", "inspect", image], timeout_s=10.0)
    if p.returncode == 0:
        logger.info("Isolated Workspace image present: %s", image)
        return image
    logger.info("Isolated Workspace image missing; building locally: %s", image)
    if not _DATA_AGENT_DOCKERFILE.exists():
        raise RuntimeError("Missing Isolated Workspace Dockerfile. Expected packaging/data-agent/Dockerfile.")
    build = _run(
        [
            "env",
            "DOCKER_BUILDKIT=1",
            "docker",
            "build",
            "-t",
            image,
            "-f",
            str(_DATA_AGENT_DOCKERFILE),
            str(_DATA_AGENT_DIR),
        ],
        timeout_s=1800.0,
    )
    if build.returncode != 0:
        logger.error(
            "Failed to build Isolated Workspace image rc=%s stdout_tail=%s stderr_tail=%s",
            build.returncode,
            (build.stdout or "")[-2000:],
            (build.stderr or "")[-2000:],
        )
        raise RuntimeError("Failed to build Isolated Workspace image. Run ./scripts/build_data_agent_image.sh.")
    logger.info("Built Isolated Workspace image: %s", image)
    return image


def _get_existing_container_id(name: str) -> str:
    p = _run(["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.ID}}"], timeout_s=10.0)
    if p.returncode != 0:
        return ""
    return (p.stdout or "").strip()


def _ensure_container_running(container_id: str) -> bool:
    if not container_id:
        return False
    p = _run(["docker", "inspect", "-f", "{{.State.Running}}", container_id], timeout_s=10.0)
    if p.returncode != 0:
        return False
    if (p.stdout or "").strip().lower() == "true":
        return True
    started = _run(["docker", "start", container_id], timeout_s=20.0)
    return started.returncode == 0


def get_container_status(
    *,
    conversation_id: Optional[UUID] = None,
    container_id: str = "",
    container_name: str = "",
) -> dict:
    name = (container_name or "").strip()
    if not name and conversation_id is not None:
        name = _container_name_for_conversation(conversation_id)

    cid = (container_id or "").strip()
    if not cid and name:
        cid = _get_existing_container_id(name)

    if not _docker_available():
        return {
            "docker_available": False,
            "exists": False,
            "running": False,
            "status": "",
            "container_id": cid,
            "container_name": name,
            "error": "Docker not available",
        }

    if not cid:
        return {
            "docker_available": True,
            "exists": False,
            "running": False,
            "status": "",
            "container_id": "",
            "container_name": name,
        }

    p = _run(
        [
            "docker",
            "inspect",
            "-f",
            "{{.State.Status}}|{{.State.Running}}|{{.State.StartedAt}}|{{.State.FinishedAt}}|{{.Name}}|{{.Id}}",
            cid,
        ],
        timeout_s=10.0,
    )
    if p.returncode != 0:
        return {
            "docker_available": True,
            "exists": False,
            "running": False,
            "status": "",
            "container_id": cid,
            "container_name": name,
            "error": (p.stderr or p.stdout or "").strip(),
        }

    parts = (p.stdout or "").strip().split("|")
    status = parts[0].strip() if len(parts) > 0 else ""
    running = parts[1].strip().lower() == "true" if len(parts) > 1 else False
    started_at = parts[2].strip() if len(parts) > 2 else ""
    finished_at = parts[3].strip() if len(parts) > 3 else ""
    name_out = parts[4].strip().lstrip("/") if len(parts) > 4 else ""
    cid_out = parts[5].strip() if len(parts) > 5 else cid

    return {
        "docker_available": True,
        "exists": True,
        "running": running,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "container_id": cid_out or cid,
        "container_name": name_out or name,
    }


def ensure_conversation_container(
    *,
    conversation_id: UUID,
    workspace_dir: str,
    openai_api_key: str,
    git_token: str = "",
    auth_json: str = "",
) -> str:
    image = ensure_image_pulled()
    name = _container_name_for_conversation(conversation_id)
    existing_id = _get_existing_container_id(name)
    if existing_id:
        if _ensure_container_running(existing_id):
            logger.info("Reusing Isolated Workspace container %s for conversation %s", existing_id, conversation_id)
            try:
                sync_container_ports_from_docker(
                    container_id=existing_id,
                    container_name=name,
                    conversation_id=conversation_id,
                )
            except Exception:
                logger.debug("Failed to sync Isolated Workspace ports for %s", existing_id, exc_info=True)
            return existing_id
        logger.warning("Isolated Workspace container exists but is not running; recreating conv=%s container_id=%s", conversation_id, existing_id)

    Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    # Back-compat cleanup: older versions wrote API spec to API_SPEC.md.
    # Keep the workspace consistent with the new api_spec.json naming.
    try:
        legacy = Path(workspace_dir) / "API_SPEC.md"
        if legacy.exists():
            legacy.unlink()
    except Exception:
        pass
    logger.info("Starting Isolated Workspace container for conversation %s (workspace=%s)", conversation_id, workspace_dir)

    repo_url, repo_cache_path, repo_source_path = _extract_preferred_repo(auth_json or "")
    host_cache_path = _normalize_host_path(repo_cache_path)
    host_source_path = _normalize_host_path(repo_source_path)
    if host_cache_path:
        try:
            Path(host_cache_path).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    ports = reserve_container_ports(conversation_id=conversation_id, container_name=name)
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-e",
        f"CODEX_API_KEY={openai_api_key}",
        "-e",
        "HOME=/work",
        "-e",
        "PYTHONUNBUFFERED=1",
    ]
    for mapping in ports:
        host_port = int(mapping.get("host") or 0)
        container_port = int(mapping.get("container") or 0)
        if host_port > 0 and container_port > 0:
            cmd.extend(["-p", f"127.0.0.1:{host_port}:{container_port}"])
    if git_token:
        cmd.extend(["-e", f"GIT_TOKEN={git_token}", "-e", f"GITHUB_TOKEN={git_token}"])
    if host_cache_path:
        cmd.extend(["-v", f"{host_cache_path}:/work/.repo_cache/preferred.git"])
    if host_source_path:
        cmd.extend(["-v", f"{host_source_path}:/work/.repo_source:ro"])
    cmd.extend(
        [
            "-v",
            f"{workspace_dir}:/work",
            "-w",
            "/work",
            image,
            "sh",
            "-lc",
            "mkdir -p /work/.codex && tail -f /dev/null",
        ]
    )
    p = _run(cmd, timeout_s=30.0)
    if p.returncode != 0:
        # If another concurrent kickoff created the container after we checked, docker run may fail
        # with a name conflict. In that case, reuse the existing container instead of failing.
        stderr = (p.stderr or "")
        if "Conflict. The container name" in stderr and name in stderr:
            existing_id = _get_existing_container_id(name)
            if existing_id and _ensure_container_running(existing_id):
                release_container_ports(container_name=name)
                try:
                    sync_container_ports_from_docker(
                        container_id=existing_id,
                        container_name=name,
                        conversation_id=conversation_id,
                    )
                except Exception:
                    logger.debug("Failed to sync Isolated Workspace ports for %s", existing_id, exc_info=True)
                logger.warning(
                    "Isolated Workspace container name conflict; reusing existing container %s for conversation %s",
                    existing_id,
                    conversation_id,
                )
                return existing_id
        release_container_ports(container_name=name)
        logger.error(
            "Failed to start Isolated Workspace container rc=%s stdout_tail=%s stderr_tail=%s",
            p.returncode,
            (p.stdout or "")[-2000:],
            (p.stderr or "")[-2000:],
        )
        raise RuntimeError(f"Failed to start Isolated Workspace container: {p.stderr.strip() or p.stdout.strip()}")
    cid = (p.stdout or "").strip()
    if not cid:
        release_container_ports(container_name=name)
        logger.error("Failed to start Isolated Workspace container: no container id returned (stdout=%s)", (p.stdout or "").strip())
        raise RuntimeError("Failed to start Isolated Workspace container: no container id returned.")
    assign_container_id_to_ports(container_id=cid, container_name=name)
    logger.info("Started Isolated Workspace container %s for conversation %s", cid, conversation_id)
    return cid
