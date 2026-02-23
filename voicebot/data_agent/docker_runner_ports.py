from __future__ import annotations

import json
import os
import re
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from .docker_runner_exec import _run


_PORT_ALLOC_LOCK = threading.Lock()
_DEFAULT_PORT_RANGE = (8000, 8100)
_DEFAULT_PORTS_PER_CONTAINER = 5


def _data_agent_state_dir() -> Path:
    raw = (
        os.environ.get("VOICEBOT_DATA_DIR")
        or os.environ.get("DATA_DIR")
        or str(Path.home() / ".gravexstudio")
    )
    path = Path(raw).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return path


def _port_allocations_path() -> Path:
    return _data_agent_state_dir() / "data_agent_ports.json"


def _load_port_allocations() -> dict[str, Any]:
    path = _port_allocations_path()
    if not path.exists():
        return {"version": 1, "ports": {}}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "ports": {}}
    if not isinstance(obj, dict):
        return {"version": 1, "ports": {}}
    ports = obj.get("ports")
    if not isinstance(ports, dict):
        obj["ports"] = {}
    if "version" not in obj:
        obj["version"] = 1
    return obj


def _save_port_allocations(obj: dict[str, Any]) -> None:
    path = _port_allocations_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _parse_port_range(raw: str) -> tuple[int, int]:
    s = (raw or "").strip()
    if not s:
        return _DEFAULT_PORT_RANGE
    m = re.match(r"^\\s*(\\d+)\\s*[-:]\\s*(\\d+)\\s*$", s)
    if not m:
        return _DEFAULT_PORT_RANGE
    try:
        start = int(m.group(1))
        end = int(m.group(2))
    except Exception:
        return _DEFAULT_PORT_RANGE
    if start < 1 or end < 1 or start > 65535 or end > 65535:
        return _DEFAULT_PORT_RANGE
    if end < start:
        start, end = end, start
    return start, end


def _port_range() -> tuple[int, int]:
    raw = (
        os.environ.get("IGX_DATA_AGENT_PORT_RANGE")
        or os.environ.get("VOICEBOT_DATA_AGENT_PORT_RANGE")
        or ""
    )
    return _parse_port_range(raw)


def _ports_per_container() -> int:
    raw = (
        os.environ.get("IGX_DATA_AGENT_PORTS_PER_CONTAINER")
        or os.environ.get("VOICEBOT_DATA_AGENT_PORTS_PER_CONTAINER")
        or ""
    )
    try:
        val = int(raw)
    except Exception:
        val = _DEFAULT_PORTS_PER_CONTAINER
    if val < 0:
        val = 0
    if val > 50:
        val = 50
    return val


def _port_available(port: int) -> bool:
    if port <= 0 or port > 65535:
        return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
        return True
    except Exception:
        return False


def _ports_for_container(data: dict[str, Any], *, container_id: str, container_name: str) -> list[dict[str, int]]:
    ports_obj = data.get("ports")
    if not isinstance(ports_obj, dict):
        return []
    items: list[dict[str, int]] = []
    for host_str, entry in ports_obj.items():
        if not isinstance(entry, dict):
            continue
        match_id = container_id and entry.get("container_id") == container_id
        match_name = container_name and entry.get("container_name") == container_name
        if not (match_id or match_name):
            continue
        try:
            host_port = int(host_str)
        except Exception:
            continue
        try:
            container_port = int(entry.get("container_port") or host_port)
        except Exception:
            container_port = host_port
        items.append({"host": host_port, "container": container_port})
    items.sort(key=lambda x: x["host"])
    return items


def reserve_container_ports(
    *,
    conversation_id: UUID,
    container_name: str,
    container_id: str = "",
    extra: int = 0,
) -> list[dict[str, int]]:
    name = (container_name or "").strip()
    cid = (container_id or "").strip()
    if not name:
        return []
    with _PORT_ALLOC_LOCK:
        data = _load_port_allocations()
        existing = _ports_for_container(data, container_id=cid, container_name=name)
        if existing:
            return existing
        count = _ports_per_container()
        try:
            extra_int = int(extra)
        except Exception:
            extra_int = 0
        if extra_int > 0:
            count = min(50, count + extra_int)
        if count <= 0:
            return []
        start, end = _port_range()
        ports_obj = data.get("ports")
        if not isinstance(ports_obj, dict):
            ports_obj = {}
            data["ports"] = ports_obj
        used = set()
        for key in ports_obj.keys():
            try:
                used.add(int(key))
            except Exception:
                continue
        picked: list[int] = []
        for port in range(start, end + 1):
            if port in used:
                continue
            if not _port_available(port):
                continue
            picked.append(port)
            if len(picked) >= count:
                break
        now = datetime.now(timezone.utc).isoformat()
        for port in picked:
            ports_obj[str(port)] = {
                "container_port": port,
                "container_id": cid,
                "container_name": name,
                "conversation_id": str(conversation_id),
                "assigned_at": now,
            }
        _save_port_allocations(data)
        return _ports_for_container(data, container_id=cid, container_name=name)


def assign_container_id_to_ports(*, container_id: str, container_name: str) -> None:
    cid = (container_id or "").strip()
    name = (container_name or "").strip()
    if not cid or not name:
        return
    with _PORT_ALLOC_LOCK:
        data = _load_port_allocations()
        ports_obj = data.get("ports")
        if not isinstance(ports_obj, dict):
            return
        for entry in ports_obj.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("container_name") != name:
                continue
            entry["container_id"] = cid
        _save_port_allocations(data)


def release_container_ports(*, container_id: str = "", container_name: str = "") -> list[dict[str, int]]:
    cid = (container_id or "").strip()
    name = (container_name or "").strip()
    if not cid and not name:
        return []
    removed: list[dict[str, int]] = []
    with _PORT_ALLOC_LOCK:
        data = _load_port_allocations()
        ports_obj = data.get("ports")
        if not isinstance(ports_obj, dict):
            return []
        to_delete: list[str] = []
        for host_port, entry in ports_obj.items():
            if not isinstance(entry, dict):
                continue
            if cid and entry.get("container_id") == cid:
                to_delete.append(host_port)
                continue
            if name and entry.get("container_name") == name:
                to_delete.append(host_port)
        for key in to_delete:
            entry = ports_obj.pop(key, None)
            if isinstance(entry, dict):
                try:
                    host = int(key)
                except Exception:
                    host = 0
                try:
                    container_port = int(entry.get("container_port") or host)
                except Exception:
                    container_port = host
                if host:
                    removed.append({"host": host, "container": container_port})
        _save_port_allocations(data)
    removed.sort(key=lambda x: x["host"])
    return removed


def _inspect_container_ports(container_id: str) -> list[dict[str, int]]:
    cid = (container_id or "").strip()
    if not cid:
        return []
    p = _run(["docker", "inspect", cid], timeout_s=10.0)
    if p.returncode != 0:
        return []
    try:
        obj = json.loads(p.stdout or "")
    except Exception:
        return []
    if not isinstance(obj, list) or not obj:
        return []
    data = obj[0] if isinstance(obj[0], dict) else {}
    ports = data.get("NetworkSettings", {}).get("Ports", {})
    if not isinstance(ports, dict):
        return []
    out: list[dict[str, int]] = []
    for key, host_list in ports.items():
        if not isinstance(key, str) or not isinstance(host_list, list):
            continue
        try:
            container_port = int(key.split("/", 1)[0])
        except Exception:
            continue
        for host in host_list:
            if not isinstance(host, dict):
                continue
            try:
                host_port = int(host.get("HostPort") or 0)
            except Exception:
                host_port = 0
            if host_port:
                out.append({"host": host_port, "container": container_port})
    dedup: dict[int, dict[str, int]] = {}
    for item in out:
        dedup[item["host"]] = item
    return sorted(dedup.values(), key=lambda x: x["host"])


def sync_container_ports_from_docker(
    *,
    container_id: str,
    container_name: str,
    conversation_id: UUID | None = None,
) -> list[dict[str, int]]:
    cid = (container_id or "").strip()
    name = (container_name or "").strip()
    if not cid or not name:
        return []
    mappings = _inspect_container_ports(cid)
    if not mappings:
        return []
    with _PORT_ALLOC_LOCK:
        data = _load_port_allocations()
        ports_obj = data.get("ports")
        if not isinstance(ports_obj, dict):
            ports_obj = {}
            data["ports"] = ports_obj
        now = datetime.now(timezone.utc).isoformat()
        for item in mappings:
            host_port = int(item.get("host") or 0)
            if not host_port:
                continue
            ports_obj[str(host_port)] = {
                "container_port": int(item.get("container") or host_port),
                "container_id": cid,
                "container_name": name,
                "conversation_id": str(conversation_id) if conversation_id else "",
                "assigned_at": now,
            }
        _save_port_allocations(data)
        return _ports_for_container(data, container_id=cid, container_name=name)


def get_container_ports(
    *,
    container_id: str = "",
    container_name: str = "",
    conversation_id: UUID | None = None,
) -> list[dict[str, int]]:
    cid = (container_id or "").strip()
    name = (container_name or "").strip()
    with _PORT_ALLOC_LOCK:
        data = _load_port_allocations()
        existing = _ports_for_container(data, container_id=cid, container_name=name)
    if existing:
        return existing
    if cid and name:
        return sync_container_ports_from_docker(
            container_id=cid,
            container_name=name,
            conversation_id=conversation_id,
        )
    return []
