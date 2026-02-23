from __future__ import annotations
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlmodel import Session


def register(app, ctx) -> None:
    router = APIRouter()

    @router.post("/api/data-agent/pull-image")
    def api_pull_data_agent_image() -> dict:
        if not ctx.docker_available():
            raise ctx.HTTPException(status_code=400, detail="Docker not available")
        try:
            image = ctx.ensure_image_pulled()
        except Exception as exc:
            raise ctx.HTTPException(status_code=500, detail=str(exc))
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
