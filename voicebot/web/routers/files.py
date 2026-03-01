from __future__ import annotations
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, UploadFile
from fastapi.responses import FileResponse
from sqlmodel import Session
from starlette.datastructures import UploadFile as StarletteUploadFile


def register(app, ctx) -> None:
    router = APIRouter()

    @router.get("/api/conversations/{conversation_id}/files")
    def api_conversation_files(
        conversation_id: UUID,
        path: str = Query("", description="Directory path relative to the data-agent workspace"),
        recursive: bool = Query(False, description="List files recursively"),
        include_hidden: bool = Query(False, description="Include dotfiles (still blocks secrets like auth.json)"),
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        conv = ctx.get_conversation(session, conversation_id)
        bot = ctx.get_bot(session, conv.bot_id)
        return ctx._conversation_files_payload(
            session=session,
            conversation_id=conversation_id,
            conv=conv,
            bot=bot,
            path=path,
            recursive=recursive,
            include_hidden=include_hidden,
            download_url_for=lambda rel: f"/api/conversations/{conversation_id}/files/download?path={ctx._url_quote(rel)}",
        )

    @router.get("/api/conversations/{conversation_id}/files/download")
    def api_conversation_file_download(
        conversation_id: UUID,
        path: str = Query(..., description="File path relative to the data-agent workspace"),
        session: Session = Depends(ctx.get_session),
    ) -> FileResponse:
        _ = ctx.get_conversation(session, conversation_id)
        req_rel = (path or "").lstrip("/").strip()
        if not req_rel:
            raise ctx.HTTPException(status_code=400, detail="Missing path")
        _root, _req_rel, target = ctx._resolve_data_agent_target(
            session,
            conversation_id=conversation_id,
            path=req_rel,
            include_hidden=False,
        )
        if not target.exists() or not target.is_file():
            raise ctx.HTTPException(status_code=404, detail="File not found")
        mt, _ = ctx.mimetypes.guess_type(str(target))
        return FileResponse(
            path=str(target),
            media_type=mt or "application/octet-stream",
            filename=target.name,
        )

    @router.post("/api/conversations/{conversation_id}/files/upload")
    async def api_conversation_files_upload(
        conversation_id: UUID,
        request: Request,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        conv = ctx.get_conversation(session, conversation_id)
        if not conv:
            raise ctx.HTTPException(status_code=404, detail="Conversation not found")
        bot = ctx.get_bot(session, conv.bot_id)
        if not bot or not bool(getattr(bot, "enable_data_agent", False)):
            raise ctx.HTTPException(status_code=400, detail="Enable Isolated Workspace to upload files.")

        meta = ctx._get_conversation_meta(session, conversation_id=conversation_id)
        try:
            ctx._ensure_data_agent_container(session, bot=bot, conversation_id=conversation_id, meta_current=meta)
        except Exception as exc:
            raise ctx.HTTPException(status_code=400, detail=str(exc))

        workspace_dir = ctx._initialize_data_agent_workspace(session, bot=bot, conversation_id=conversation_id, meta=meta)
        root = ctx.Path(workspace_dir).resolve()

        try:
            form = await request.form(
                max_files=10_000_000,
                max_fields=10_000_000,
                max_part_size=10 * 1024 * 1024 * 1024,
            )
        except Exception as exc:
            raise ctx.HTTPException(status_code=400, detail=str(exc))

        files: list[UploadFile] = []
        for key, value in form.multi_items():
            if key != "files":
                continue
            # Starlette form parser may return starlette UploadFile objects directly.
            if isinstance(value, (UploadFile, StarletteUploadFile)):
                files.append(value)
        if not files:
            raise ctx.HTTPException(status_code=400, detail="No files uploaded")

        saved: list[str] = []
        try:
            for f in files:
                rel = ctx._sanitize_upload_path(f.filename or "")
                if not rel:
                    continue
                target = (root / rel).resolve()
                if not ctx._is_path_within_root(root, target):
                    raise ctx.HTTPException(status_code=400, detail="Invalid filename")
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as out:
                    ctx.shutil.copyfileobj(f.file, out)
                saved.append(rel)
        finally:
            for f in files:
                try:
                    await f.close()
                except Exception:
                    pass

        return {"ok": True, "files": saved, "workspace_dir": str(root)}

    app.include_router(router)
