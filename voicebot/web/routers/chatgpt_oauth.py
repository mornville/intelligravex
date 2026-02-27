from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from fastapi import APIRouter, Depends
from sqlmodel import Session


AUTH_BASE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
DEFAULT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_REDIRECT_URI = "http://localhost:1455/auth/callback"
DEFAULT_SCOPE = "openid profile email offline_access"


@dataclass
class OAuthSession:
    state: str
    code_verifier: str
    created_at: float
    code: Optional[str] = None
    error: Optional[str] = None


_sessions: dict[str, OAuthSession] = {}
_sessions_lock = threading.Lock()
_server_started = False
_server_error: Optional[str] = None


def _client_id() -> str:
    return (os.environ.get("CHATGPT_OAUTH_CLIENT_ID") or "").strip() or DEFAULT_CLIENT_ID


def _redirect_uri() -> str:
    return (os.environ.get("CHATGPT_OAUTH_REDIRECT_URI") or "").strip() or DEFAULT_REDIRECT_URI


def _scope() -> str:
    return (os.environ.get("CHATGPT_OAUTH_SCOPE") or "").strip() or DEFAULT_SCOPE


def _pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")
    challenge = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge_b64 = base64.urlsafe_b64encode(challenge).decode("utf-8").rstrip("=")
    return verifier, challenge_b64


def _auth_url(state: str, code_challenge: str) -> str:
    params = {
        "response_type": "code",
        "client_id": _client_id(),
        "redirect_uri": _redirect_uri(),
        "scope": _scope(),
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
    }
    return f"{AUTH_BASE_URL}?{urlencode(params)}"


def _exchange_code(code: str, code_verifier: str) -> dict:
    data = {
        "grant_type": "authorization_code",
        "client_id": _client_id(),
        "code": code,
        "redirect_uri": _redirect_uri(),
        "code_verifier": code_verifier,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = httpx.post(TOKEN_URL, data=data, headers=headers, timeout=12.0)
    if resp.status_code >= 400:
        raise RuntimeError(f"OAuth token exchange failed ({resp.status_code}).")
    obj = resp.json()
    if not isinstance(obj, dict) or not obj.get("access_token"):
        raise RuntimeError("OAuth token exchange failed (missing access token).")
    return obj


def _ensure_server() -> None:
    global _server_started, _server_error
    if _server_started:
        return
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", 1455), _make_handler())
    except Exception as exc:
        _server_error = str(exc)
        return
    _server_started = True

    def _serve():
        try:
            httpd.serve_forever()
        except Exception:
            pass

    t = threading.Thread(target=_serve, daemon=True)
    t.start()


def _make_handler():
    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/auth/callback":
                self.send_response(404)
                self.end_headers()
                return
            qs = parse_qs(parsed.query)
            state = (qs.get("state") or [None])[0]
            code = (qs.get("code") or [None])[0]
            error = (qs.get("error") or [None])[0]
            if state:
                with _sessions_lock:
                    sess = _sessions.get(state)
                    if sess:
                        sess.code = code
                        sess.error = error
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='font-family:system-ui;margin:40px;'>"
                b"<h2>ChatGPT sign-in complete</h2>"
                b"<p>You can close this window and return to GravexStudio.</p>"
                b"</body></html>"
            )

        def log_message(self, format, *args):  # noqa: A003
            return

    return CallbackHandler


def register(app, ctx) -> None:
    router = APIRouter()

    @router.post("/api/chatgpt/oauth/start")
    def chatgpt_oauth_start() -> dict:
        _ensure_server()
        if _server_error:
            raise ctx.HTTPException(status_code=500, detail=f"OAuth callback server failed: {_server_error}")
        state = secrets.token_urlsafe(24)
        verifier, challenge = _pkce_pair()
        with _sessions_lock:
            _sessions[state] = OAuthSession(state=state, code_verifier=verifier, created_at=time.time())
        return {"state": state, "auth_url": _auth_url(state, challenge)}

    @router.get("/api/chatgpt/oauth/status")
    def chatgpt_oauth_status(state: str, session: Session = Depends(ctx.get_session)) -> dict:
        with _sessions_lock:
            sess = _sessions.get(state)
        if not sess:
            return {"status": "expired"}
        if sess.error:
            with _sessions_lock:
                _sessions.pop(state, None)
            return {"status": "error", "error": sess.error}
        if not sess.code:
            # expire after 10 minutes
            if (time.time() - sess.created_at) > 600:
                with _sessions_lock:
                    _sessions.pop(state, None)
                return {"status": "expired"}
            return {"status": "pending"}
        try:
            tokens = _exchange_code(sess.code, sess.code_verifier)
            expires_in = tokens.get("expires_in")
            if isinstance(expires_in, (int, float)):
                tokens["expires_at"] = time.time() + float(expires_in)
            secret = json.dumps(tokens, ensure_ascii=False)
            crypto = ctx.require_crypto()
            ctx._upsert_key(session, crypto=crypto, provider="chatgpt", name="ChatGPT OAuth", secret=secret)
            ctx._set_app_setting(session, "default_llm_provider", "chatgpt")
            ctx._set_app_setting(session, "default_llm_model", "gpt-5.2")
            ctx._get_or_create_showcase_bot(session)
        except Exception as exc:
            with _sessions_lock:
                _sessions.pop(state, None)
            return {"status": "error", "error": str(exc)}
        with _sessions_lock:
            _sessions.pop(state, None)
        return {"status": "ready"}

    app.include_router(router)
