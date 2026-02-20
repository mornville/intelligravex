from __future__ import annotations

import base64
import secrets

from fastapi import Request, WebSocket


def basic_auth_ok(
    auth_header: str,
    *,
    basic_auth_enabled: bool,
    basic_user: str,
    basic_pass: str,
) -> bool:
    if not basic_auth_enabled:
        return True
    if not auth_header:
        return False
    if not auth_header.lower().startswith("basic "):
        return False
    token = auth_header.split(" ", 1)[1].strip()
    try:
        decoded = base64.b64decode(token).decode("utf-8")
    except Exception:
        return False
    user, sep, pwd = decoded.partition(":")
    if not sep:
        return False
    return secrets.compare_digest(user, basic_user) and secrets.compare_digest(pwd, basic_pass)


def viewer_id_from_request(request: Request) -> str:
    auth_header = (request.headers.get("authorization") or "").strip()
    if auth_header.lower().startswith("basic "):
        token = auth_header.split(" ", 1)[1].strip()
        try:
            decoded = base64.b64decode(token).decode("utf-8")
        except Exception:
            decoded = ""
        user, sep, _ = decoded.partition(":")
        if sep and user:
            return f"basic:{user}"
    return "local"


def ws_auth_header(ws: WebSocket) -> str:
    header = (ws.headers.get("authorization") or "").strip()
    if header:
        return header
    token = (ws.query_params.get("auth") or "").strip()
    if not token:
        return ""
    if token.lower().startswith("basic "):
        return token
    return f"Basic {token}"


def accepts_html(accept_header: str) -> bool:
    accept = (accept_header or "").lower()
    if not accept:
        return True
    if "text/html" in accept:
        return True
    if "*/*" in accept:
        return True
    return False
