from __future__ import annotations

import base64
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

from .docker_runner_constants import logger
from .docker_runner_exec import run_container_command
from .docker_runner_stream import _safe_str


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _setup_ssh_from_auth(ws: Path, auth_json: str) -> None:
    try:
        auth_obj = json.loads((auth_json or "").strip() or "{}")
    except Exception:
        return
    if not isinstance(auth_obj, dict):
        return

    key = str(auth_obj.get("ssh_private_key") or auth_obj.get("ssh_key") or auth_obj.get("ssh_private_key_pem") or "")
    if not key:
        key_path = str(auth_obj.get("ssh_private_key_path") or auth_obj.get("ssh_key_path") or "").strip()
        if key_path:
            try:
                key = Path(key_path).read_text(encoding="utf-8")
            except Exception:
                key = ""
    if not key:
        key_b64 = str(auth_obj.get("ssh_private_key_b64") or auth_obj.get("ssh_private_key_base64") or "")
        if key_b64:
            try:
                key = base64.b64decode(key_b64).decode("utf-8", errors="ignore")
            except Exception:
                key = ""
    if not key.strip():
        return

    ssh_dir = ws / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    key_name = str(auth_obj.get("ssh_key_filename") or "id_ed25519").strip() or "id_ed25519"
    key_path = ssh_dir / key_name
    _write_text(key_path, key.strip() + "\n")
    try:
        os.chmod(ssh_dir, 0o700)
        os.chmod(key_path, 0o600)
    except Exception:
        pass

    pub_key = str(auth_obj.get("ssh_public_key") or "")
    if not pub_key:
        pub_key_path = str(auth_obj.get("ssh_public_key_path") or "").strip()
        if pub_key_path:
            try:
                pub_key = Path(pub_key_path).read_text(encoding="utf-8")
            except Exception:
                pub_key = ""
    if pub_key.strip():
        _write_text(ssh_dir / f"{key_name}.pub", pub_key.strip() + "\n")

    known_hosts = str(auth_obj.get("ssh_known_hosts") or auth_obj.get("known_hosts") or "").strip()
    if not known_hosts:
        known_hosts_path = str(auth_obj.get("ssh_known_hosts_path") or "").strip()
        if known_hosts_path:
            try:
                known_hosts = Path(known_hosts_path).read_text(encoding="utf-8").strip()
            except Exception:
                known_hosts = ""
    if known_hosts:
        _write_text(ssh_dir / "known_hosts", known_hosts + "\n")

    passphrase = str(
        auth_obj.get("ssh_key_passphrase")
        or auth_obj.get("ssh_private_key_passphrase")
        or auth_obj.get("ssh_passphrase")
        or ""
    )
    passphrase = passphrase.strip()
    if passphrase:
        pass_path = ssh_dir / "passphrase"
        _write_text(pass_path, passphrase)
        askpass_path = ssh_dir / "askpass.sh"
        _write_text(askpass_path, "#!/bin/sh\ncat /work/.ssh/passphrase\n")
        try:
            os.chmod(pass_path, 0o600)
            os.chmod(askpass_path, 0o700)
        except Exception:
            pass

    ssh_user = str(auth_obj.get("ssh_user") or auth_obj.get("git_ssh_user") or "git").strip() or "git"
    ssh_host = str(auth_obj.get("ssh_host") or auth_obj.get("git_host") or "github.com").strip() or "github.com"
    strict = str(auth_obj.get("ssh_strict_host_key_checking") or "").strip().lower()
    if not strict:
        strict = "yes" if known_hosts else "accept-new"

    config = (
        f"Host {ssh_host}\n"
        f"  HostName {ssh_host}\n"
        f"  User {ssh_user}\n"
        f"  IdentityFile /work/.ssh/{key_name}\n"
        "  IdentitiesOnly yes\n"
        f"  StrictHostKeyChecking {strict}\n"
        "  UserKnownHostsFile /work/.ssh/known_hosts\n"
    )
    _write_text(ssh_dir / "config", config)


def _parse_auth_json(auth_json: str) -> dict[str, Any]:
    try:
        obj = json.loads((auth_json or "").strip() or "{}")
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    return obj


def _extract_preferred_repo(auth_json: str) -> tuple[str, str, str]:
    obj = _parse_auth_json(auth_json)
    repo_url = str(
        obj.get("preferred_repo_url")
        or obj.get("git_preferred_repo_url")
        or obj.get("git_repo_url")
        or obj.get("preferred_repo")
        or ""
    ).strip()
    cache_path = str(
        obj.get("preferred_repo_cache_path")
        or obj.get("git_repo_cache_path")
        or obj.get("preferred_repo_path")
        or ""
    ).strip()
    source_path = str(
        obj.get("preferred_repo_source_path")
        or obj.get("git_repo_source_path")
        or obj.get("preferred_repo_working_path")
        or ""
    ).strip()
    return repo_url, cache_path, source_path


def _repo_name_from_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u:
        return ""
    if u.endswith(".git"):
        u = u[: -len(".git")]
    if ":" in u and "://" not in u:
        u = u.split(":", 1)[1]
    if "/" in u:
        u = u.rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", u) or "repo"


def _build_ssh_env_prefix(ws: Path) -> str:
    env_prefix: list[str] = []
    ssh_config = ws / ".ssh" / "config"
    if ssh_config.exists():
        env_prefix.append('GIT_SSH_COMMAND="ssh -F /work/.ssh/config"')
    askpass = ws / ".ssh" / "askpass.sh"
    if askpass.exists():
        env_prefix.append("SSH_ASKPASS=/work/.ssh/askpass.sh")
        env_prefix.append("SSH_ASKPASS_REQUIRE=force")
        env_prefix.append("DISPLAY=1")
    return " ".join(env_prefix)


def _normalize_host_path(raw: str) -> str:
    if not raw:
        return ""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)
    try:
        return str((Path.cwd() / p).resolve())
    except Exception:
        return str(p)


def _ensure_preferred_repo(container_id: str, *, ws: Path, auth_json: str) -> None:
    repo_url, cache_path, source_path = _extract_preferred_repo(auth_json)
    if not repo_url or (not cache_path and not source_path):
        return
    repo_name = _repo_name_from_url(repo_url)
    if not repo_name:
        return
    logger.info(
        "Preferred repo setup: repo=%s cache=%s source=%s",
        _safe_str(repo_url, limit=120),
        _safe_str(cache_path, limit=120),
        _safe_str(source_path, limit=120),
    )
    mirror_path = "/work/.repo_cache/preferred.git"
    source_mount = "/work/.repo_source"
    if source_path:
        check = run_container_command(
            container_id=container_id,
            command=f"[ -d {source_mount}/.git ] || [ -f {source_mount}/HEAD ]",
            timeout_s=5.0,
        )
        if not check.ok:
            logger.warning(
                "Preferred repo source path not mounted; reuse requires a new container. source=%s",
                _safe_str(source_path, limit=120),
            )
            return
    env_prefix = _build_ssh_env_prefix(ws)
    # Ensure mirror exists/updated, then ensure working tree exists.
    ref_expr = (
        f"REF=''; "
        f"if [ -d {source_mount}/.git ] || [ -f {source_mount}/HEAD ]; then REF='{source_mount}'; "
        f"elif [ -d {mirror_path}/objects ]; then REF='{mirror_path}'; "
        f"fi; "
        f"REF_ARG=''; if [ -n \"$REF\" ]; then REF_ARG=\"--reference $REF\"; fi; "
    )
    mirror_setup = (
        f"mkdir -p /work/.repo_cache && "
        f"if [ -d {mirror_path}/objects ]; then "
        f"  git -C {mirror_path} remote update --prune; "
        f"elif [ -z \"$REF\" ]; then "
        f"  git clone --mirror {shlex.quote(repo_url)} {mirror_path}; "
        f"fi; "
    )
    clone_setup = (
        f"if [ -d /work/{repo_name}/.git ]; then "
        f"  git -C /work/{repo_name} fetch --all --prune; "
        f"else "
        f"  git clone $REF_ARG {shlex.quote(repo_url)} /work/{repo_name}; "
        f"fi"
    )
    script = ref_expr + mirror_setup + clone_setup
    if env_prefix:
        script = f"{env_prefix} {script}"
    res = run_container_command(container_id=container_id, command=script, timeout_s=600.0)
    if not res.ok:
        logger.warning(
            "Preferred repo prefetch failed: repo=%s cache=%s exit=%s stderr=%s",
            _safe_str(repo_url, limit=120),
            _safe_str(cache_path, limit=120),
            res.exit_code,
            _safe_str(res.stderr, limit=240),
        )
