from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse


def register(app, ctx) -> None:
    router = APIRouter()

    @router.get("/api/downloads/{token}")
    def download_file(token: str) -> FileResponse:
        """
        Download a previously exported file by token.

        NOTE: This uses an unguessable token and a strict allowlist of temp roots.
        TODO(cleanup): add TTL + authentication/authorization as needed for production.
        """
        obj = ctx.load_download_token(token=token)
        if not obj:
            raise ctx.HTTPException(status_code=404, detail="Download token not found")
        file_path = str(obj.get("file_path") or "").strip()
        if not file_path or not ctx.os.path.exists(file_path):
            raise ctx.HTTPException(status_code=404, detail="File not found")
        if not ctx.is_allowed_download_path(file_path):
            raise ctx.HTTPException(status_code=403, detail="File path not allowed")
        filename = str(obj.get("filename") or ctx.os.path.basename(file_path) or "download").strip()
        mime_type = str(obj.get("mime_type") or "application/octet-stream").strip()
        return FileResponse(path=file_path, media_type=mime_type, filename=filename)

    app.include_router(router)
