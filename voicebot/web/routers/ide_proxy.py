from __future__ import annotations

import asyncio
from uuid import UUID

import httpx
import websockets
from fastapi import APIRouter, Depends, Request, WebSocket
from fastapi.responses import Response, StreamingResponse
from sqlmodel import Session


def register(app, ctx) -> None:
    router = APIRouter()

    def _resolve_ide_port(session: Session, conversation_id: UUID) -> int:
        meta = ctx._get_conversation_meta(session, conversation_id=conversation_id)
        da = ctx._data_agent_meta(meta)
        ide_port = 0
        try:
            ide_port = int(da.get("ide_port") or 0)
        except Exception:
            ide_port = 0
        if ide_port:
            return ide_port
        container_id = str(da.get("container_id") or "").strip()
        if not container_id:
            return 0
        try:
            name = ctx.container_name_for_conversation(conversation_id)
            ports = ctx.get_container_ports(
                container_id=container_id,
                container_name=name,
                conversation_id=conversation_id,
            )
        except Exception:
            ports = []
        for item in ports or []:
            try:
                host = int((item or {}).get("host") or 0)
            except Exception:
                host = 0
            if host > ide_port:
                ide_port = host
        return ide_port

    def _proxy_headers(headers: httpx.Headers) -> dict[str, str]:
        hop_by_hop = {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
        }
        out: dict[str, str] = {}
        for k, v in headers.items():
            if k.lower() in hop_by_hop:
                continue
            out[k] = v
        return out

    def _upstream_base(conversation_id: UUID, port: int) -> str:
        return f"http://127.0.0.1:{port}/ide/{conversation_id}"

    @router.api_route(
        "/ide/{conversation_id}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    )
    @router.api_route(
        "/ide/{conversation_id}/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    )
    async def ide_proxy(
        request: Request,
        conversation_id: UUID,
        path: str = "",
        session: Session = Depends(ctx.get_session),
    ) -> Response:
        _ = ctx.get_conversation(session, conversation_id)
        meta = ctx._get_conversation_meta(session, conversation_id=conversation_id)
        da = ctx._data_agent_meta(meta)
        container_id = str(da.get("container_id") or "").strip()
        status = ctx.get_container_status(conversation_id=conversation_id, container_id=container_id)
        if not status.get("running"):
            raise ctx.HTTPException(status_code=409, detail="Isolated Workspace is not running.")
        ide_port = _resolve_ide_port(session, conversation_id)
        if ide_port <= 0:
            raise ctx.HTTPException(status_code=404, detail="IDE port not available.")

        upstream = _upstream_base(conversation_id, ide_port)
        if path:
            upstream = f"{upstream}/{path}"
        if request.url.query:
            upstream = f"{upstream}?{request.url.query}"

        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)
        body = await request.body()

        client = httpx.AsyncClient(timeout=None, follow_redirects=False)
        req = client.build_request(
            request.method,
            upstream,
            headers=headers,
            content=body if body else None,
        )
        resp = await client.send(req, stream=True)

        async def _iter():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        response_headers = _proxy_headers(resp.headers)
        return StreamingResponse(
            _iter(),
            status_code=resp.status_code,
            headers=response_headers,
            media_type=resp.headers.get("content-type"),
        )

    @router.websocket("/ide/{conversation_id}/{path:path}")
    async def ide_proxy_ws(websocket: WebSocket, conversation_id: UUID, path: str = "") -> None:
        if not ctx._basic_auth_ok(ctx._ws_auth_header(websocket)):
            await websocket.accept()
            await websocket.send_json({"type": "error", "error": "Unauthorized"})
            await websocket.close(code=4401)
            return

        try:
            with Session(ctx.engine) as session:
                _ = ctx.get_conversation(session, conversation_id)
                meta = ctx._get_conversation_meta(session, conversation_id=conversation_id)
                da = ctx._data_agent_meta(meta)
                container_id = str(da.get("container_id") or "").strip()
                status = ctx.get_container_status(conversation_id=conversation_id, container_id=container_id)
                if not status.get("running"):
                    await websocket.accept()
                    await websocket.send_json({"type": "error", "error": "Isolated Workspace is not running."})
                    await websocket.close(code=4409)
                    return
                ide_port = _resolve_ide_port(session, conversation_id)
        except Exception as exc:
            await websocket.accept()
            await websocket.send_json({"type": "error", "error": str(exc)})
            await websocket.close(code=1011)
            return

        if ide_port <= 0:
            await websocket.accept()
            await websocket.send_json({"type": "error", "error": "IDE port not available."})
            await websocket.close(code=1011)
            return

        query = websocket.url.query
        upstream = f"ws://127.0.0.1:{ide_port}/ide/{conversation_id}"
        if path:
            upstream = f"{upstream}/{path}"
        if query:
            upstream = f"{upstream}?{query}"

        await websocket.accept()

        async with websockets.connect(upstream, max_size=None) as upstream_ws:
            async def client_to_upstream() -> None:
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if msg.get("text") is not None:
                        await upstream_ws.send(msg["text"])
                    elif msg.get("bytes") is not None:
                        await upstream_ws.send(msg["bytes"])

            async def upstream_to_client() -> None:
                async for msg in upstream_ws:
                    if isinstance(msg, (bytes, bytearray)):
                        await websocket.send_bytes(bytes(msg))
                    else:
                        await websocket.send_text(str(msg))

            tasks = [
                asyncio.create_task(client_to_upstream()),
                asyncio.create_task(upstream_to_client()),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                if task.exception():
                    raise task.exception()

    app.include_router(router)
