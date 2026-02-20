from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import Response


def register(app, ctx) -> None:
    router = APIRouter()

    @router.get("/public/widget.js")
    def public_widget_js() -> Response:
        p = Path(__file__).parent.parent / "static" / "embed-widget.js"
        if not p.exists():
            raise ctx.HTTPException(status_code=404, detail="widget.js not found")
        return Response(content=p.read_text("utf-8"), media_type="application/javascript")

    app.include_router(router)
