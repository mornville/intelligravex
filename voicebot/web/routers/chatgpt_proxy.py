from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlmodel import Session


def _codex_base_url() -> str:
    raw = (os.environ.get("CHATGPT_OAUTH_BASE_URL") or "").strip()
    base = raw.rstrip("/")
    if not base:
        return "https://chatgpt.com/backend-api/codex/responses"
    if base.endswith("/codex/responses"):
        return base
    if base.endswith("/codex"):
        return f"{base}/responses"
    if base.endswith("/backend-api"):
        return f"{base}/codex/responses"
    return base


def _messages_to_responses(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    items: list[dict[str, Any]] = []
    instructions_parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip()
        content = m.get("content")
        if not role or not isinstance(content, str):
            continue
        if role == "system":
            if content:
                instructions_parts.append(content)
            continue
        content_type = "output_text" if role == "assistant" else "input_text"
        items.append({"role": role, "content": [{"type": content_type, "text": content}]})
    return items, "\n\n".join(instructions_parts).strip()


def _normalize_payload(body: dict[str, Any]) -> dict[str, Any]:
    payload = dict(body)
    if "messages" in payload and "input" not in payload:
        messages = payload.get("messages")
        if isinstance(messages, list):
            input_items, instructions = _messages_to_responses(messages)
            if input_items:
                payload["input"] = input_items
            if instructions and not payload.get("instructions"):
                payload["instructions"] = instructions
    if not payload.get("instructions"):
        payload["instructions"] = "You are a helpful assistant."
    payload.setdefault("store", False)
    return payload


def register(app, ctx) -> None:
    router = APIRouter()

    @router.post("/api/chatgpt/proxy/v1/responses")
    @router.post("/api/chatgpt/proxy/responses")
    async def chatgpt_proxy_responses(request: Request, session: Session = Depends(ctx.get_session)) -> Response:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        token = ctx._get_chatgpt_api_key(session)
        if not token:
            return JSONResponse({"error": "No ChatGPT OAuth token configured"}, status_code=400)

        payload = _normalize_payload(body)
        url = _codex_base_url()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "OpenAI-Beta": "responses=codex",
        }
        stream = bool(payload.get("stream"))
        client = httpx.AsyncClient(timeout=None)
        req = client.build_request("POST", url, json=payload, headers=headers)
        resp = await client.send(req, stream=stream)
        if resp.status_code >= 400:
            data = await resp.aread()
            await resp.aclose()
            await client.aclose()
            return Response(data, status_code=resp.status_code, media_type=resp.headers.get("content-type") or "application/json")
        if not stream:
            data = await resp.aread()
            await resp.aclose()
            await client.aclose()
            return Response(data, status_code=resp.status_code, media_type=resp.headers.get("content-type") or "application/json")

        async def _iter_stream():
            try:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(
            _iter_stream(),
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type") or "text/event-stream",
        )

    app.include_router(router)
