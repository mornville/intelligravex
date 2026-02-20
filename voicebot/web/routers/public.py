from __future__ import annotations
from uuid import UUID

from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from sqlmodel import Session


def register(app, ctx) -> None:
    router = APIRouter()

    @router.get("/conversations/{conversation_id}")
    def public_conversation_transcript(
        conversation_id: UUID,
        request: Request,
        key: str = Query("", description="Client key secret (igx_...)"),
        include_tools: bool = Query(False, description="Include tool call/result messages"),
        include_system: bool = Query(False, description="Include system messages"),
        format: str = Query("html", description="html|json"),
        session: Session = Depends(ctx.get_session),
    ) -> Response:
        if not key.strip():
            if ctx.ui_index.exists() and ctx._accepts_html(request.headers.get("accept") or ""):
                return FileResponse(str(ctx.ui_index))
            raise ctx.HTTPException(status_code=401, detail="Missing key")
        ck = ctx.verify_client_key(session, secret=key.strip())
        if not ck:
            raise ctx.HTTPException(status_code=401, detail="Invalid key")
        origin = request.headers.get("origin") if request else None
        if origin and (not ctx._origin_allowed(ck, origin)):
            raise ctx.HTTPException(status_code=403, detail="Origin not allowed")

        try:
            conv = ctx.get_conversation(session, conversation_id)
        except Exception:
            raise ctx.HTTPException(status_code=404, detail="Conversation not found")
        if conv.client_key_id != ck.id:
            raise ctx.HTTPException(status_code=403, detail="Conversation not accessible with this key")
        if not ctx._bot_allowed(ck, conv.bot_id):
            raise ctx.HTTPException(status_code=403, detail="Bot not allowed for this key")

        payload = ctx._conversation_messages_payload(
            session=session,
            conversation_id=conversation_id,
            include_tools=bool(include_tools),
            include_system=bool(include_system),
        )

        fmt = (format or "").strip().lower()
        if fmt == "json":
            return Response(content=ctx.json.dumps(payload, ensure_ascii=False), media_type="application/json")

        title = "Conversation Transcript"
        page = ctx._render_conversation_html(
            title=title,
            conversation_id=conversation_id,
            key=key.strip(),
            include_tools=bool(include_tools),
            include_system=bool(include_system),
            payload=payload,
        )
        return HTMLResponse(content=page)

    @router.get("/conversations/{conversation_id}/files")
    def public_conversation_files(
        conversation_id: UUID,
        request: Request,
        key: str = Query("", description="Client key secret (igx_...)"),
        path: str = Query("", description="Directory path relative to the data-agent workspace"),
        recursive: bool = Query(False, description="List files recursively"),
        include_hidden: bool = Query(False, description="Include dotfiles (still blocks secrets like auth.json)"),
        format: str = Query("html", description="html|json"),
        session: Session = Depends(ctx.get_session),
    ) -> Response:
        _ck, conv, bot = ctx._require_public_conversation_access(
            session=session, request=request, conversation_id=conversation_id, key=key
        )
        base_key = (key or "").strip()
        payload = ctx._conversation_files_payload(
            session=session,
            conversation_id=conversation_id,
            conv=conv,
            bot=bot,
            path=path,
            recursive=recursive,
            include_hidden=include_hidden,
            download_url_for=lambda rel: (
                f"/conversations/{conversation_id}/files/download?key={ctx._url_quote(base_key)}&path={ctx._url_quote(rel)}"
            ),
        )
        items = payload.get("items") or []
        max_items = int(payload.get("max_items") or 0)
        req_rel = str(payload.get("path") or "")

        fmt = (format or "").strip().lower()
        if fmt == "json":
            return Response(content=ctx.json.dumps(payload, ensure_ascii=False), media_type="application/json")

        def _fmt_size(sz: Optional[int]) -> str:
            if sz is None:
                return ""
            n = float(sz)
            for unit in ["B", "KB", "MB", "GB"]:
                if n < 1024.0:
                    return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
                n /= 1024.0
            return f"{n:.1f} TB"

        def _q(**params: str) -> str:
            parts = []
            for k, v in params.items():
                parts.append(f"{k}={v}")
            return "&".join(parts)

        base_params = {"key": ctx._url_quote((key or "").strip())}
        cur_path_q = ctx._url_quote(req_rel)
        json_href = (
            f"/conversations/{conversation_id}/files?"
            f"{_q(**base_params, path=cur_path_q, recursive=('1' if recursive else '0'), include_hidden=('1' if include_hidden else '0'), format='json')}"
        )
        rec_on = (
            f"/conversations/{conversation_id}/files?"
            f"{_q(**base_params, path=cur_path_q, recursive='1', include_hidden=('1' if include_hidden else '0'))}"
        )
        rec_off = (
            f"/conversations/{conversation_id}/files?"
            f"{_q(**base_params, path=cur_path_q, recursive='0', include_hidden=('1' if include_hidden else '0'))}"
        )
        hidden_on = (
            f"/conversations/{conversation_id}/files?"
            f"{_q(**base_params, path=cur_path_q, recursive=('1' if recursive else '0'), include_hidden='1')}"
        )
        hidden_off = (
            f"/conversations/{conversation_id}/files?"
            f"{_q(**base_params, path=cur_path_q, recursive=('1' if recursive else '0'), include_hidden='0')}"
        )
        transcript_href = f"/conversations/{conversation_id}?{_q(**base_params)}"

        rows: list[str] = []
        for it in items:
            rel = str(it.get("path") or "")
            is_dir = bool(it.get("is_dir"))
            mtime = str(it.get("mtime") or "")
            size = _fmt_size(it.get("size_bytes"))  # type: ignore[arg-type]
            dl = str(it.get("download_url") or "")
            name = rel
            href = ""
            if is_dir:
                href = (
                    f"/conversations/{conversation_id}/files?"
                    f"{_q(**base_params, path=ctx._url_quote(rel), recursive='0', include_hidden=('1' if include_hidden else '0'))}"
                )
            elif dl:
                href = dl
            link = f"<a href='{ctx.html.escape(href)}'>{ctx.html.escape(name)}</a>" if href else ctx.html.escape(name)
            kind = "dir" if is_dir else "file"
            rows.append(
                "<tr>"
                f"<td class='c-kind'>{ctx.html.escape(kind)}</td>"
                f"<td class='c-path'>{link}</td>"
                f"<td class='c-size'>{ctx.html.escape(size)}</td>"
                f"<td class='c-time'>{ctx.html.escape(mtime)}</td>"
                "</tr>"
            )

        page = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Conversation Files</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #111936;
      --border: rgba(255,255,255,0.12);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.70);
      --link: #7dd3fc;
      --chip: rgba(255,255,255,0.08);
    }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: var(--bg); color: var(--text); }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
    .header {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }}
    h1 {{ font-size: 18px; margin: 0 0 6px; }}
    .sub {{ color: var(--muted); font-size: 13px; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }}
    a.btn {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 10px; border: 1px solid var(--border); border-radius: 10px; background: var(--chip); color: var(--text); text-decoration: none; font-size: 13px; }}
    a.btn:hover {{ border-color: rgba(255,255,255,0.25); }}
    a.btn.primary {{ border-color: rgba(125, 211, 252, 0.6); color: var(--link); }}
    .panel {{ margin-top: 14px; background: var(--panel); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }}
    table {{ width: 100%; border-collapse: collapse; }}
    thead th {{ position: sticky; top: 0; background: rgba(17,25,54,0.92); backdrop-filter: blur(10px); text-align: left; font-size: 12px; color: var(--muted); padding: 10px; border-bottom: 1px solid var(--border); }}
    tbody td {{ vertical-align: top; padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.06); }}
    .c-kind {{ width: 80px; white-space: nowrap; color: var(--muted); font-size: 12px; }}
    .c-size {{ width: 110px; white-space: nowrap; color: var(--muted); font-size: 12px; }}
    .c-time {{ width: 260px; white-space: nowrap; color: var(--muted); font-size: 12px; }}
    .c-path a {{ color: var(--link); text-decoration: none; }}
    .c-path a:hover {{ text-decoration: underline; }}
    .footer {{ margin-top: 12px; font-size: 12px; color: var(--muted); }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"header\">
      <div>
        <h1>Files for conversation {ctx.html.escape(str(conversation_id))}</h1>
        <div class=\"sub\">Path: /{ctx.html.escape(req_rel)} • Items: {len(items)} (max {max_items})</div>
      </div>
      <div class=\"actions\">
        <a class=\"btn\" href=\"{ctx.html.escape(transcript_href)}\">Transcript</a>
        <a class=\"btn primary\" href=\"{ctx.html.escape(rec_on)}\">Recursive on</a>
        <a class=\"btn\" href=\"{ctx.html.escape(rec_off)}\">Recursive off</a>
        <a class=\"btn\" href=\"{ctx.html.escape(hidden_on)}\">Show hidden</a>
        <a class=\"btn\" href=\"{ctx.html.escape(hidden_off)}\">Hide hidden</a>
        <a class=\"btn\" href=\"{ctx.html.escape(json_href)}\">JSON</a>
      </div>
    </div>
    <div class=\"panel\">
      <table>
        <thead>
          <tr>
            <th>Type</th>
            <th>Path</th>
            <th>Size</th>
            <th>Modified (UTC)</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    <div class=\"footer\">
      Hidden/secrets are filtered (e.g. <code>auth.json</code>, <code>AGENTS.md</code>, <code>.codex/</code>).
    </div>
  </div>
</body>
</html>
"""
        return HTMLResponse(content=page)

    @router.get("/conversations/{conversation_id}/files/download")
    def public_conversation_file_download(
        conversation_id: UUID,
        request: Request,
        key: str = Query("", description="Client key secret (igx_...)"),
        path: str = Query(..., description="File path relative to the data-agent workspace"),
        session: Session = Depends(ctx.get_session),
    ) -> FileResponse:
        _ck, _conv, _bot = ctx._require_public_conversation_access(
            session=session, request=request, conversation_id=conversation_id, key=key
        )
        req_rel = (path or "").lstrip("/").strip()
        if not req_rel:
            raise ctx.HTTPException(status_code=400, detail="Missing path")
        _root, req_rel, target = ctx._resolve_data_agent_target(
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

    app.include_router(router)
