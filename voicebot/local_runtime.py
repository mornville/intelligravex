from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx


DEFAULT_PORT = 11432
STATE_IDLE = "idle"
STATE_DOWNLOADING_RUNTIME = "downloading_runtime"
STATE_DOWNLOADING_MODEL = "downloading_model"
STATE_VERIFYING = "verifying"
STATE_STARTING = "starting"
STATE_READY = "ready"
STATE_ERROR = "error"


@dataclass
class LocalModel:
    id: str
    name: str
    download_url: str
    filename: str
    size_gb: float | None = None
    min_ram_gb: float | None = None
    supports_tools: bool = True
    tool_support: str | None = None
    recommended: bool = False


@dataclass
class LocalRuntimeState:
    state: str = STATE_IDLE
    message: str = ""
    model_id: str = ""
    bytes_total: int = 0
    bytes_downloaded: int = 0
    error: str = ""
    server_port: int = DEFAULT_PORT
    server_pid: int | None = None
    last_update_ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        total = int(self.bytes_total or 0)
        done = int(self.bytes_downloaded or 0)
        pct = 0.0
        if total > 0:
            pct = min(100.0, max(0.0, (done / total) * 100.0))
        out["percent"] = round(pct, 2)
        return out


class LocalRuntimeManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = LocalRuntimeState()
        self._proc: subprocess.Popen | None = None
        self._worker: threading.Thread | None = None
        self._model_path: Path | None = None
        self._catalog_cache: dict[str, Any] = {"ts": 0.0, "models": []}

    def status(self) -> dict[str, Any]:
        self._refresh_ready_state()
        with self._lock:
            return self._state.to_dict()

    def is_ready(self) -> bool:
        self._refresh_ready_state()
        with self._lock:
            if not self._proc:
                return False
            if self._proc.poll() is not None:
                return False
            return self._state.state == STATE_READY

    def stop(self, *, reset_state: bool = True) -> None:
        proc: subprocess.Popen | None = None
        with self._lock:
            proc = self._proc
            self._proc = None
            self._worker = None
            if reset_state:
                self._state = LocalRuntimeState(state=STATE_IDLE, message="")
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except Exception:
                    proc.kill()
        except Exception:
            pass

    def _refresh_ready_state(self) -> None:
        with self._lock:
            if not self._proc:
                return
            if self._proc.poll() is not None:
                return
            port = int(self._state.server_port or 0) or 0
            if not port:
                return
        if _is_server_healthy(port):
            with self._lock:
                self._state.state = STATE_READY
                self._state.message = "Local model ready."
                self._state.error = ""
                self._state.last_update_ts = time.time()

    def list_models(self) -> list[dict[str, Any]]:
        models = self._load_catalog()
        total_ram = _get_total_ram_gb()
        out = []
        for m in models:
            rec = bool(m.recommended)
            if not rec and m.min_ram_gb is not None:
                rec = total_ram >= float(m.min_ram_gb)
            out.append(
                {
                    "id": m.id,
                    "name": m.name,
                    "download_url": m.download_url,
                    "filename": m.filename,
                    "size_gb": m.size_gb,
                    "min_ram_gb": m.min_ram_gb,
                    "supports_tools": bool(m.supports_tools),
                    "tool_support": m.tool_support,
                    "recommended": bool(rec),
                }
            )
        return out

    def start(self, *, model_id: str, custom_url: str | None = None, custom_name: str | None = None) -> dict[str, Any]:
        model_id = (model_id or "").strip()
        custom_url = (custom_url or "").strip()
        custom_name = (custom_name or "").strip()
        if not model_id and not custom_url:
            raise RuntimeError("Model id or custom URL required.")

        with self._lock:
            if self._worker and self._worker.is_alive():
                return self._state.to_dict()
            self._state = LocalRuntimeState(state=STATE_STARTING, message="Starting local runtime...", model_id=model_id)

            def _run():
                try:
                    model = self._resolve_model(model_id, custom_url=custom_url, custom_name=custom_name)
                    with self._lock:
                        self._state.model_id = model.id
                    self._ensure_runtime_binary()
                    model_path = self._ensure_model(model)
                    self._start_server(model_path)
                    with self._lock:
                        self._state.state = STATE_READY
                        self._state.message = "Local model ready."
                        self._state.error = ""
                        self._state.last_update_ts = time.time()
                except Exception as exc:
                    with self._lock:
                        self._state.state = STATE_ERROR
                        self._state.error = str(exc)
                        self._state.message = "Local setup failed."
                        self._state.last_update_ts = time.time()

            self._worker = threading.Thread(target=_run, daemon=True)
            self._worker.start()
            return self._state.to_dict()

    def _resolve_model(self, model_id: str, *, custom_url: str, custom_name: str) -> LocalModel:
        if custom_url:
            filename = _safe_filename_from_url(custom_url)
            name = custom_name or model_id or filename
            return LocalModel(
                id=model_id or filename,
                name=name,
                download_url=custom_url,
                filename=filename,
                size_gb=None,
                min_ram_gb=None,
                supports_tools=True,
                recommended=False,
            )
        models = self._load_catalog()
        for m in models:
            if m.id == model_id:
                return m
        raise RuntimeError("Unknown local model.")

    def _load_catalog(self) -> list[LocalModel]:
        now = time.time()
        if (now - float(self._catalog_cache.get("ts") or 0.0)) < 900.0:
            return list(self._catalog_cache.get("models") or [])
        models: list[LocalModel] = []
        try:
            models = _load_catalog_from_env()
        except Exception:
            models = []
        if not models:
            try:
                models = _load_catalog_from_file()
            except Exception:
                models = []
        if not models:
            models = _default_catalog()
        self._catalog_cache = {"ts": now, "models": models}
        return models

    def _ensure_runtime_binary(self) -> Path:
        path = _resolve_llama_server_path()
        if path and path.exists():
            return path
        url = (os.environ.get("IGX_LLAMA_SERVER_URL") or os.environ.get("IGX_LOCAL_LLM_SERVER_URL") or "").strip()
        if not url:
            raise RuntimeError("Local runtime not found. Bundle llama-server or set IGX_LLAMA_SERVER_URL.")
        bin_dir = _local_bin_dir()
        bin_dir.mkdir(parents=True, exist_ok=True)
        dest = bin_dir / "llama-server"
        self._download_asset(url, dest)
        try:
            os.chmod(dest, 0o755)
        except Exception:
            pass
        return dest

    def _download_asset(self, url: str, dest: Path) -> None:
        with self._lock:
            self._state.state = STATE_DOWNLOADING_RUNTIME
            self._state.message = "Downloading local runtime..."
            self._state.bytes_total = 0
            self._state.bytes_downloaded = 0
            self._state.last_update_ts = time.time()
        tmp = dest.with_suffix(".download")
        _download_stream(url, tmp, progress_fn=self._update_progress)
        if _is_archive(tmp):
            extracted = _extract_runtime(tmp, dest.parent)
            if not extracted:
                raise RuntimeError("Failed to extract llama-server.")
            if extracted != dest:
                try:
                    extracted.replace(dest)
                except Exception:
                    shutil.copy2(extracted, dest)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return
        tmp.replace(dest)

    def _ensure_model(self, model: LocalModel) -> Path:
        if not model.download_url:
            raise RuntimeError("Model download URL missing. Update local model catalog.")
        if not model.filename:
            raise RuntimeError("Model filename missing. Update local model catalog.")
        models_dir = _models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        dest = models_dir / model.filename
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        with self._lock:
            self._state.state = STATE_DOWNLOADING_MODEL
            self._state.message = f"Downloading {model.name}..."
            self._state.bytes_total = 0
            self._state.bytes_downloaded = 0
            self._state.model_id = model.id
            self._state.last_update_ts = time.time()
        tmp = dest.with_suffix(".download")
        _download_stream(model.download_url, tmp, progress_fn=self._update_progress)
        with self._lock:
            self._state.state = STATE_VERIFYING
            self._state.message = "Verifying model..."
            self._state.last_update_ts = time.time()
        tmp.replace(dest)
        return dest

    def _start_server(self, model_path: Path) -> None:
        port = _local_port()
        with self._lock:
            self._state.state = STATE_STARTING
            self._state.message = "Starting local runtime..."
            self._state.server_port = port
            self._state.last_update_ts = time.time()
        if self._proc and self._proc.poll() is None:
            if self._model_path and self._model_path == model_path:
                return
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        server_path = _resolve_llama_server_path()
        if not server_path:
            raise RuntimeError("llama-server binary not found.")
        cmd = [
            str(server_path),
            "-m",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        extra = (os.environ.get("IGX_LLAMA_SERVER_ARGS") or "").strip()
        if extra:
            cmd.extend(shlex.split(extra))
        log_dir = _local_bin_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "llama-server.out.log"
        stderr_path = log_dir / "llama-server.err.log"
        stdout_f = open(stdout_path, "a", encoding="utf-8")
        stderr_f = open(stderr_path, "a", encoding="utf-8")
        env = os.environ.copy()
        if platform.system() == "Darwin":
            existing = env.get("DYLD_LIBRARY_PATH", "")
            env["DYLD_LIBRARY_PATH"] = f"{server_path.parent}:{existing}" if existing else str(server_path.parent)
        else:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{server_path.parent}:{existing}" if existing else str(server_path.parent)
        self._proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, env=env)
        with self._lock:
            self._state.server_pid = self._proc.pid
            self._state.last_update_ts = time.time()
            self._model_path = model_path
        _wait_for_server_ready(port)

    def _update_progress(self, done: int, total: int) -> None:
        with self._lock:
            self._state.bytes_downloaded = int(done)
            self._state.bytes_total = int(total)
            self._state.last_update_ts = time.time()


def _get_total_ram_gb() -> float:
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return float(pages * page_size) / (1024**3)
    except Exception:
        pass
    return 0.0


def _app_support_dir() -> Path:
    home = Path.home()
    if platform.system() == "Darwin":
        return home / "Library" / "Application Support" / "GravexOverlay"
    return home / ".igx"


def _models_dir() -> Path:
    return _app_support_dir() / "local-models"


def _local_bin_dir() -> Path:
    return _app_support_dir() / "local-runtime"


def _local_port() -> int:
    try:
        raw = int(os.environ.get("IGX_LOCAL_LLM_PORT") or 0)
        if raw > 0:
            return raw
    except Exception:
        pass
    return DEFAULT_PORT


def _resolve_llama_server_path() -> Optional[Path]:
    raw = (os.environ.get("IGX_LLAMA_SERVER_PATH") or os.environ.get("IGX_LOCAL_LLM_SERVER_PATH") or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if p.exists():
            return p
    try:
        exe = Path(sys.executable).resolve()
        for parent in exe.parents:
            if parent.name == "Contents":
                candidate = parent / "Resources" / "llama-server"
                if candidate.exists():
                    return candidate
                break
    except Exception:
        pass
    bundled = _local_bin_dir() / "llama-server"
    if bundled.exists():
        return bundled
    for candidate in ("llama-server", "server"):
        found = shutil.which(candidate)
        if found:
            return Path(found)
    return None


def _wait_for_server_ready(port: int) -> None:
    for _ in range(120):
        if _is_server_healthy(port):
            return
        time.sleep(0.5)
    raise RuntimeError("Local runtime failed to start.")


def _is_server_healthy(port: int) -> bool:
    base = f"http://127.0.0.1:{port}"
    try:
        resp = httpx.get(f"{base}/v1/models", timeout=2.0)
        return resp.status_code < 500
    except Exception:
        return False


def _download_stream(url: str, dest: Path, *, progress_fn) -> None:
    headers: dict[str, str] = {}
    token = (os.environ.get("IGX_HF_TOKEN") or os.environ.get("HF_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.stream("GET", url, timeout=None, follow_redirects=True, headers=headers) as resp:
        if resp.status_code >= 400:
            raise RuntimeError(f"Download failed ({resp.status_code}).")
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes():
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                progress_fn(done, total)


def _safe_filename_from_url(url: str) -> str:
    name = url.rsplit("/", 1)[-1].split("?", 1)[0].strip()
    if not name:
        name = "model.gguf"
    return name


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".zip") or name.endswith(".tar.gz") or name.endswith(".tgz")


def _extract_runtime(archive: Path, dest_dir: Path) -> Optional[Path]:
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest_dir)
        return _find_runtime_binary(dest_dir)
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        import tarfile

        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest_dir)
        return _find_runtime_binary(dest_dir)
    return None


def _find_runtime_binary(root: Path) -> Optional[Path]:
    for path in root.rglob("llama-server"):
        return path
    for path in root.rglob("server"):
        return path
    return None


def _load_catalog_from_file() -> list[LocalModel]:
    path = Path(__file__).parent / "local_models.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8") or "[]")
    return _parse_models(raw)


def _load_catalog_from_env() -> list[LocalModel]:
    url = (os.environ.get("IGX_LOCAL_MODEL_CATALOG_URL") or "").strip()
    if not url:
        return []
    resp = httpx.get(url, timeout=15.0)
    if resp.status_code >= 400:
        return []
    return _parse_models(resp.json())


def _parse_models(raw: Any) -> list[LocalModel]:
    if not isinstance(raw, list):
        return []
    out: list[LocalModel] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        name = str(item.get("name") or mid).strip()
        url = str(item.get("download_url") or "").strip()
        filename = str(item.get("filename") or _safe_filename_from_url(url)).strip()
        if not mid or not url or not filename:
            continue
        out.append(
            LocalModel(
                id=mid,
                name=name or mid,
                download_url=url,
                filename=filename,
                size_gb=_to_float(item.get("size_gb")),
                min_ram_gb=_to_float(item.get("min_ram_gb")),
                supports_tools=bool(item.get("supports_tools", True)),
                tool_support=str(item.get("tool_support") or "").strip() or None,
                recommended=bool(item.get("recommended", False)),
            )
        )
    return out


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _default_catalog() -> list[LocalModel]:
    return []


LOCAL_RUNTIME = LocalRuntimeManager()
