from __future__ import annotations
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlmodel import Session


def register(app, ctx) -> None:
    router = APIRouter()
    DOCKER_DOWNLOAD_URL = "https://www.docker.com/products/docker-desktop/"

    def _ts() -> str:
        return ctx.dt.datetime.now(ctx.dt.timezone.utc).isoformat()

    def _setup_snapshot(*, refresh: bool = True) -> dict:
        with ctx.data_agent_setup_lock:
            state = dict(ctx.data_agent_setup_state)
            state["logs"] = list(ctx.data_agent_setup_state.get("logs") or [])
            worker = ctx.data_agent_setup_thread
        if not refresh:
            return state
        if worker is not None and worker.is_alive():
            state["status"] = "building"
            return state
        try:
            status = ctx.data_agent_image_status()
        except Exception as exc:
            status = {
                "docker_available": bool(ctx.docker_available()),
                "image": str(state.get("image") or ctx.data_agent_image_name()),
                "image_present": False,
                "error": str(exc) or "Failed to check Isolated Workspace image status.",
            }
        docker_available = bool(status.get("docker_available", False))
        image = str(status.get("image") or state.get("image") or "")
        image_ready = bool(status.get("image_present", False))
        next_state = dict(state)
        next_state["docker_available"] = docker_available
        next_state["image"] = image
        next_state["image_ready"] = image_ready
        if image_ready:
            next_state["status"] = "ready"
            next_state["error"] = ""
            next_state["message"] = f"{image} is ready. You may use the Isolated Workspace now."
        elif not docker_available:
            next_state["status"] = "error"
            next_state["error"] = f"Docker is not available. Install Docker Desktop: {DOCKER_DOWNLOAD_URL}"
            next_state["message"] = next_state["error"]
        elif str(next_state.get("status") or "") not in {"building", "error"}:
            next_state["status"] = "idle"
            if not str(next_state.get("message") or "").strip():
                next_state["message"] = "Isolated Workspace image is not built yet."
        next_state["updated_at"] = _ts()
        with ctx.data_agent_setup_lock:
            ctx.data_agent_setup_state.update(next_state)
            state = dict(ctx.data_agent_setup_state)
            state["logs"] = list(ctx.data_agent_setup_state.get("logs") or [])
        return state

    def _append_setup_log(line: str) -> None:
        text = str(line or "").strip()
        if not text:
            return
        now = _ts()
        with ctx.data_agent_setup_lock:
            logs = list(ctx.data_agent_setup_state.get("logs") or [])
            logs.append(text)
            if len(logs) > 120:
                logs = logs[-120:]
            ctx.data_agent_setup_state["logs"] = logs
            ctx.data_agent_setup_state["message"] = text
            ctx.data_agent_setup_state["updated_at"] = now

    def _start_image_build() -> dict:
        snap = _setup_snapshot(refresh=True)
        if snap.get("image_ready"):
            return snap
        if not snap.get("docker_available"):
            raise ctx.HTTPException(status_code=400, detail=f"Docker not available. Install Docker Desktop: {DOCKER_DOWNLOAD_URL}")
        with ctx.data_agent_setup_lock:
            worker = ctx.data_agent_setup_thread
            if worker is not None and worker.is_alive():
                snap = dict(ctx.data_agent_setup_state)
                snap["logs"] = list(ctx.data_agent_setup_state.get("logs") or [])
                return snap
            now = _ts()
            image = str(snap.get("image") or ctx.data_agent_image_name())
            ctx.data_agent_setup_state.update(
                {
                    "status": "building",
                    "docker_available": True,
                    "image": image,
                    "image_ready": False,
                    "message": f"Building {image}...",
                    "error": "",
                    "logs": [f"Starting build for {image}..."],
                    "started_at": now,
                    "updated_at": now,
                    "finished_at": None,
                }
            )

            def _worker() -> None:
                try:
                    image_name = ctx.ensure_image_pulled(on_progress=_append_setup_log)
                    done = _ts()
                    with ctx.data_agent_setup_lock:
                        ctx.data_agent_setup_state.update(
                            {
                                "status": "ready",
                                "docker_available": True,
                                "image": image_name,
                                "image_ready": True,
                                "message": f"{image_name} is ready. You may use the Isolated Workspace now.",
                                "error": "",
                                "updated_at": done,
                                "finished_at": done,
                            }
                        )
                        logs = list(ctx.data_agent_setup_state.get("logs") or [])
                        logs.append("Build complete.")
                        ctx.data_agent_setup_state["logs"] = logs[-120:]
                except Exception as exc:
                    done = _ts()
                    msg = str(exc) or "Failed to build Isolated Workspace image."
                    with ctx.data_agent_setup_lock:
                        ctx.data_agent_setup_state.update(
                            {
                                "status": "error",
                                "docker_available": bool(ctx.docker_available()),
                                "image_ready": False,
                                "message": msg,
                                "error": msg,
                                "updated_at": done,
                                "finished_at": done,
                            }
                        )
                        logs = list(ctx.data_agent_setup_state.get("logs") or [])
                        logs.append(f"Build failed: {msg}")
                        ctx.data_agent_setup_state["logs"] = logs[-120:]
                finally:
                    with ctx.data_agent_setup_lock:
                        ctx.data_agent_setup_thread = None

            thread = ctx.threading.Thread(target=_worker, daemon=True, name="igx-data-agent-image-build")
            ctx.data_agent_setup_thread = thread
            thread.start()
            snap = dict(ctx.data_agent_setup_state)
            snap["logs"] = list(ctx.data_agent_setup_state.get("logs") or [])
            return snap

    @router.get("/api/data-agent/setup/status")
    def api_data_agent_setup_status() -> dict:
        return _setup_snapshot(refresh=True)

    @router.post("/api/data-agent/setup/ensure")
    def api_data_agent_setup_ensure() -> dict:
        snap = _setup_snapshot(refresh=True)
        if snap.get("image_ready"):
            return snap
        if not snap.get("docker_available"):
            raise ctx.HTTPException(status_code=400, detail=f"Docker not available. Install Docker Desktop: {DOCKER_DOWNLOAD_URL}")
        if str(snap.get("status") or "") == "building":
            return snap
        return _start_image_build()

    @router.post("/api/data-agent/pull-image")
    def api_pull_data_agent_image() -> dict:
        if not ctx.docker_available():
            raise ctx.HTTPException(status_code=400, detail=f"Docker not available. Install Docker Desktop: {DOCKER_DOWNLOAD_URL}")
        try:
            image = ctx.ensure_image_pulled()
        except Exception as exc:
            with ctx.data_agent_setup_lock:
                now = _ts()
                ctx.data_agent_setup_state.update(
                    {
                        "status": "error",
                        "docker_available": True,
                        "image_ready": False,
                        "message": str(exc),
                        "error": str(exc),
                        "updated_at": now,
                        "finished_at": now,
                    }
                )
            raise ctx.HTTPException(status_code=500, detail=str(exc))
        with ctx.data_agent_setup_lock:
            now = _ts()
            ctx.data_agent_setup_state.update(
                {
                    "status": "ready",
                    "docker_available": True,
                    "image": image,
                    "image_ready": True,
                    "message": f"{image} is ready. You may use the Isolated Workspace now.",
                    "error": "",
                    "updated_at": now,
                    "finished_at": now,
                }
            )
        return {"ok": True, "image": image}

    @router.get("/api/data-agent/containers")
    def api_data_agent_containers() -> dict:
        return ctx.list_data_agent_containers()

    @router.post("/api/data-agent/containers/{container_id}/stop")
    def api_data_agent_container_stop(container_id: str) -> dict:
        res = ctx.stop_data_agent_container(container_id)
        if not res.get("docker_available", True):
            raise ctx.HTTPException(status_code=400, detail="Docker not available")
        if not res.get("stopped", False):
            raise ctx.HTTPException(status_code=500, detail=str(res.get("error") or "Failed to stop container"))
        return res

    @router.get("/api/conversations/{conversation_id}/data-agent")
    def api_conversation_data_agent(conversation_id: UUID, session: Session = Depends(ctx.get_session)) -> dict:
        _ = ctx.get_conversation(session, conversation_id)
        meta = ctx._get_conversation_meta(session, conversation_id=conversation_id)
        da = ctx._data_agent_meta(meta)
        container_id = str(da.get("container_id") or "").strip()
        session_id = str(da.get("session_id") or "").strip()
        ports = da.get("ports") if isinstance(da.get("ports"), list) else []
        ide_port = da.get("ide_port")
        workspace_dir = str(da.get("workspace_dir") or "").strip() or ctx.default_workspace_dir_for_conversation(conversation_id)
        status = ctx.get_container_status(conversation_id=conversation_id, container_id=container_id)
        status["conversation_id"] = str(conversation_id)
        status["workspace_dir"] = workspace_dir
        status["session_id"] = session_id
        if not ports and container_id:
            try:
                name = status.get("container_name") or ctx.container_name_for_conversation(conversation_id)
                ports = ctx.get_container_ports(container_id=container_id, container_name=name, conversation_id=conversation_id)
            except Exception:
                ports = []
        if ide_port is None:
            ide_port = 0
            if ports:
                for item in ports:
                    try:
                        host = int((item or {}).get("host") or 0)
                    except Exception:
                        host = 0
                    if host > ide_port:
                        ide_port = host
        if ide_port:
            ports = [p for p in ports if int((p or {}).get("host") or 0) != int(ide_port)]
        status["ports"] = ports
        status["ide_port"] = ide_port
        if container_id and not status.get("container_id"):
            status["container_id"] = container_id
        return status

    @router.post("/api/conversations/{conversation_id}/data-agent/cancel")
    def api_conversation_data_agent_cancel(conversation_id: UUID, session: Session = Depends(ctx.get_session)) -> dict:
        _ = ctx.get_conversation(session, conversation_id)
        meta = ctx._get_conversation_meta(session, conversation_id=conversation_id)
        da = ctx._data_agent_meta(meta)
        container_id = str(da.get("container_id") or "").strip()
        if not container_id:
            return {"ok": False, "error": "No Isolated Workspace container for this conversation."}
        kill_script = (
            "for p in /proc/[0-9]*; do "
            "cmd=$(tr '\\0' ' ' < \"$p\"/cmdline 2>/dev/null); "
            "case \"$cmd\" in "
            "*codex\\ exec*|*'/codex/codex exec'*|*'/usr/local/bin/codex exec'*|*'@openai/codex'*|*'git clone'*|*'git-upload-pack'*|*'git index-pack'*) "
            "pid=${p##*/}; "
            "if [ \"$pid\" != \"$$\" ]; then kill -9 \"$pid\" 2>/dev/null || true; fi "
            ";; "
            "esac; "
            "done; "
            "echo cancelled"
        )
        res = ctx.run_container_command(container_id=container_id, command=kill_script, timeout_s=15.0)
        return {
            "ok": res.ok,
            "exit_code": res.exit_code,
            "stdout": res.stdout,
            "stderr": res.stderr,
        }

    app.include_router(router)
