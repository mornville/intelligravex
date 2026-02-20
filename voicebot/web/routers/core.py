from __future__ import annotations
from uuid import UUID

from typing import Optional

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlmodel import Session

from ..schemas import OpenDashboardRequest, WidgetConfigRequest


def register(app, ctx) -> None:
    router = APIRouter()

    @router.get("/api/system-bot")
    def api_system_bot(session: Session = Depends(ctx.get_session)) -> dict:
        bot = ctx._get_or_create_system_bot(session)
        return {"id": str(bot.id), "name": bot.name}

    @router.get("/api/widget-config")
    def api_widget_config(session: Session = Depends(ctx.get_session)) -> dict:
        bot_id = ctx._get_app_setting(session, ctx.WIDGET_BOT_KEY)
        widget_mode = (ctx._get_app_setting(session, ctx.WIDGET_MODE_KEY) or "").strip().lower()
        if widget_mode not in ("mic", "text"):
            widget_mode = "mic"
        bot_name = None
        if bot_id:
            try:
                bot = ctx.get_bot(session, UUID(bot_id))
                bot_name = bot.name
            except Exception:
                bot_id = None
        return {"bot_id": bot_id, "bot_name": bot_name, "widget_mode": widget_mode}

    @router.post("/api/widget-config")
    def api_widget_config_update(
        payload: WidgetConfigRequest = Body(...),
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        bot_id: Optional[str] = None
        bot_name: Optional[str] = None
        if payload.bot_id is not None:
            raw_bot_id = str(payload.bot_id or "").strip()
            if not raw_bot_id:
                ctx._set_app_setting(session, ctx.WIDGET_BOT_KEY, "")
            else:
                bot = ctx.get_bot(session, UUID(raw_bot_id))
                ctx._set_app_setting(session, ctx.WIDGET_BOT_KEY, str(bot.id))
                bot_id = str(bot.id)
                bot_name = bot.name

        if payload.widget_mode is not None:
            raw_mode = str(payload.widget_mode or "").strip().lower()
            if raw_mode not in ("mic", "text"):
                raise ctx.HTTPException(status_code=400, detail="widget_mode must be 'mic' or 'text'")
            ctx._set_app_setting(session, ctx.WIDGET_MODE_KEY, raw_mode)

        if bot_id is None:
            stored_bot_id = ctx._get_app_setting(session, ctx.WIDGET_BOT_KEY)
            if stored_bot_id:
                try:
                    bot = ctx.get_bot(session, UUID(stored_bot_id))
                    bot_id = str(bot.id)
                    bot_name = bot.name
                except Exception:
                    bot_id = None
                    bot_name = None

        widget_mode = (ctx._get_app_setting(session, ctx.WIDGET_MODE_KEY) or "").strip().lower()
        if widget_mode not in ("mic", "text"):
            widget_mode = "mic"
        return {"bot_id": bot_id, "bot_name": bot_name, "widget_mode": widget_mode}

    @router.post("/api/open-dashboard")
    def api_open_dashboard(payload: Optional[OpenDashboardRequest] = Body(None)) -> dict:
        host = (ctx.os.environ.get("VOICEBOT_LAUNCH_HOST") or "127.0.0.1").strip() or "127.0.0.1"
        port = (ctx.os.environ.get("VOICEBOT_LAUNCH_PORT") or "8000").strip() or "8000"
        requested = ""
        if payload and payload.path:
            requested = str(payload.path or "").strip()
        path = requested or (ctx.os.environ.get("VOICEBOT_OPEN_PATH") or "/dashboard").strip() or "/dashboard"
        if "://" in path:
            path = "/dashboard"
        if not path.startswith("/"):
            path = f"/{path}"
        url = f"http://{host}:{port}{path}"
        try:
            ctx.webbrowser.open(url)
        except Exception:
            return {"ok": False, "url": url}
        return {"ok": True, "url": url}

    @router.get("/", include_in_schema=False)
    def root(request: Request):
        if ctx.ui_index.exists() and ctx._accepts_html(request.headers.get("accept") or ""):
            return FileResponse(str(ctx.ui_index))
        return {"ok": True, "api_base": "/api", "public_widget_js": "/public/widget.js", "docs": "/docs"}

    async def spa_fallback(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            path = request.url.path or ""
            if not path.startswith(("/api", "/public", "/static", "/ws", "/docs", "/openapi.json")):
                if ctx.ui_index.exists() and ctx._accepts_html(request.headers.get("accept") or ""):
                    return FileResponse(str(ctx.ui_index))
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    app.add_exception_handler(StarletteHTTPException, spa_fallback)
    app.include_router(router)
