from __future__ import annotations
import base64
import hashlib
import json
import os
import secrets
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, parse_qsl, quote_plus, urlencode, urlparse
from uuid import UUID

import httpx
from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel
from sqlmodel import Session

from ..schemas import BotCreateRequest, BotUpdateRequest, IntegrationToolCreateRequest, IntegrationToolUpdateRequest


GMAIL_AUTH_BASE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GMAIL_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
GMAIL_DEFAULT_REDIRECT_URI = "http://localhost:1466/auth/callback"
GMAIL_DEFAULT_SCOPE = "openid email profile https://www.googleapis.com/auth/gmail.send https://www.googleapis.com/auth/gmail.readonly"
SLACK_AUTH_BASE_URL = "https://slack.com/oauth/v2/authorize"
SLACK_TOKEN_URL = "https://slack.com/api/oauth.v2.access"
SLACK_AUTH_TEST_URL = "https://slack.com/api/auth.test"
SLACK_DEFAULT_REDIRECT_URI = "http://localhost:1467/auth/callback"
SLACK_DEFAULT_SCOPE = "chat:write,channels:read,groups:read,im:read,mpim:read"


@dataclass
class GmailOAuthSession:
    state: str
    bot_id: str
    code_verifier: str
    created_at: float
    code: Optional[str] = None
    error: Optional[str] = None


@dataclass
class SlackOAuthSession:
    state: str
    bot_id: str
    created_at: float
    client_id: str
    client_secret: str
    redirect_uri: str
    scope: str
    code: Optional[str] = None
    error: Optional[str] = None


_gmail_sessions: dict[str, GmailOAuthSession] = {}
_gmail_sessions_lock = threading.Lock()
_gmail_server_started = False
_gmail_server_error: Optional[str] = None
_slack_sessions: dict[str, SlackOAuthSession] = {}
_slack_sessions_lock = threading.Lock()
_slack_server_started = False
_slack_server_error: Optional[str] = None
_slack_server_host: Optional[str] = None
_slack_server_port: Optional[int] = None
_slack_server_path: Optional[str] = None


def _env_value(name: str) -> str:
    raw = (os.environ.get(name) or "").strip()
    if raw:
        return raw
    prefixed = (os.environ.get(f"VOICEBOT_{name}") or "").strip()
    if prefixed:
        return prefixed
    try:
        from dotenv import dotenv_values  # type: ignore

        values = dotenv_values(".env")
        if isinstance(values, dict):
            from_file = str(values.get(name) or "").strip()
            if from_file:
                return from_file
            from_file_prefixed = str(values.get(f"VOICEBOT_{name}") or "").strip()
            if from_file_prefixed:
                return from_file_prefixed
    except Exception:
        pass
    return ""


def _gmail_client_id() -> str:
    return _env_value("GMAIL_OAUTH_CLIENT_ID")


def _gmail_client_secret() -> str:
    return _env_value("GMAIL_OAUTH_CLIENT_SECRET")


def _gmail_redirect_uri() -> str:
    return _env_value("GMAIL_OAUTH_REDIRECT_URI") or GMAIL_DEFAULT_REDIRECT_URI


def _gmail_scope() -> str:
    return _env_value("GMAIL_OAUTH_SCOPE") or GMAIL_DEFAULT_SCOPE


def _gmail_pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")
    challenge = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge_b64 = base64.urlsafe_b64encode(challenge).decode("utf-8").rstrip("=")
    return verifier, challenge_b64


def _gmail_auth_url(state: str, code_challenge: str) -> str:
    params = {
        "response_type": "code",
        "client_id": _gmail_client_id(),
        "redirect_uri": _gmail_redirect_uri(),
        "scope": _gmail_scope(),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{GMAIL_AUTH_BASE_URL}?{urlencode(params)}"


def _gmail_exchange_code(code: str, code_verifier: str) -> dict:
    data = {
        "grant_type": "authorization_code",
        "client_id": _gmail_client_id(),
        "client_secret": _gmail_client_secret(),
        "code": code,
        "redirect_uri": _gmail_redirect_uri(),
        "code_verifier": code_verifier,
    }
    resp = httpx.post(GMAIL_TOKEN_URL, data=data, timeout=12.0)
    if resp.status_code >= 400:
        try:
            err = resp.json()
            msg = str((err or {}).get("error_description") or (err or {}).get("error") or "")
        except Exception:
            msg = ""
        raise RuntimeError(msg or f"Gmail OAuth token exchange failed ({resp.status_code}).")
    obj = resp.json()
    if not isinstance(obj, dict) or not obj.get("access_token"):
        raise RuntimeError("Gmail OAuth token exchange failed (missing access token).")
    return obj


def _gmail_userinfo(access_token: str) -> dict:
    token = str(access_token or "").strip()
    if not token:
        return {}
    resp = httpx.get(
        GMAIL_USERINFO_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0,
    )
    if resp.status_code >= 400:
        return {}
    try:
        obj = resp.json()
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _make_gmail_handler():
    expected_path = urlparse(_gmail_redirect_uri()).path or "/auth/callback"

    class GmailCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != expected_path:
                self.send_response(404)
                self.end_headers()
                return
            qs = parse_qs(parsed.query)
            state = (qs.get("state") or [None])[0]
            code = (qs.get("code") or [None])[0]
            error = (qs.get("error") or [None])[0]
            if state:
                with _gmail_sessions_lock:
                    sess = _gmail_sessions.get(state)
                    if sess:
                        sess.code = code
                        sess.error = error
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='font-family:system-ui;margin:40px;'>"
                b"<h2>Google sign-in complete</h2>"
                b"<p>You can close this window and return to GravexStudio.</p>"
                b"</body></html>"
            )

        def log_message(self, format, *args):  # noqa: A003
            return

    return GmailCallbackHandler


def _ensure_gmail_server() -> None:
    global _gmail_server_started, _gmail_server_error
    if _gmail_server_started:
        return
    parsed = urlparse(_gmail_redirect_uri())
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 80)
    if parsed.scheme not in {"http", ""}:
        _gmail_server_error = "GMAIL_OAUTH_REDIRECT_URI must use http:// for local callback."
        return
    try:
        httpd = ThreadingHTTPServer((host, port), _make_gmail_handler())
    except Exception as exc:
        _gmail_server_error = str(exc)
        return
    _gmail_server_started = True

    def _serve():
        try:
            httpd.serve_forever()
        except Exception:
            pass

    t = threading.Thread(target=_serve, daemon=True, name="gmail-oauth-callback")
    t.start()


def _slack_client_id() -> str:
    return _env_value("SLACK_OAUTH_CLIENT_ID")


def _slack_client_secret() -> str:
    return _env_value("SLACK_OAUTH_CLIENT_SECRET")


def _slack_redirect_uri() -> str:
    return _env_value("SLACK_OAUTH_REDIRECT_URI") or SLACK_DEFAULT_REDIRECT_URI


def _slack_scope() -> str:
    return _env_value("SLACK_OAUTH_SCOPE") or SLACK_DEFAULT_SCOPE


def _slack_auth_url(*, state: str, client_id: str, redirect_uri: str, scope: str) -> str:
    params = {
        "client_id": client_id,
        "scope": scope,
        "redirect_uri": redirect_uri,
        "state": state,
    }
    return f"{SLACK_AUTH_BASE_URL}?{urlencode(params)}"


def _slack_exchange_code(*, code: str, client_id: str, client_secret: str, redirect_uri: str) -> dict:
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    resp = httpx.post(SLACK_TOKEN_URL, data=data, timeout=12.0)
    if resp.status_code >= 400:
        raise RuntimeError(f"Slack OAuth token exchange failed ({resp.status_code}).")
    obj = resp.json()
    if not isinstance(obj, dict) or not bool(obj.get("ok")):
        msg = str((obj or {}).get("error") or "").strip() if isinstance(obj, dict) else ""
        raise RuntimeError(msg or "Slack OAuth token exchange failed.")
    if not str(obj.get("access_token") or "").strip():
        raise RuntimeError("Slack OAuth token exchange failed (missing access token).")
    return obj


def _slack_auth_test(access_token: str) -> dict:
    token = str(access_token or "").strip()
    if not token:
        raise RuntimeError("Slack access token is missing.")
    resp = httpx.post(
        SLACK_AUTH_TEST_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Slack auth test failed ({resp.status_code}).")
    obj = resp.json()
    if not isinstance(obj, dict) or not bool(obj.get("ok")):
        msg = str((obj or {}).get("error") or "").strip() if isinstance(obj, dict) else ""
        raise RuntimeError(msg or "Slack auth test failed.")
    return obj


def _make_slack_handler(expected_path: str):
    class SlackCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != expected_path:
                self.send_response(404)
                self.end_headers()
                return
            qs = parse_qs(parsed.query)
            state = (qs.get("state") or [None])[0]
            code = (qs.get("code") or [None])[0]
            error = (qs.get("error") or [None])[0]
            error_description = (qs.get("error_description") or [None])[0]
            if error and error_description:
                error = f"{error}: {error_description}"
            if state:
                with _slack_sessions_lock:
                    sess = _slack_sessions.get(state)
                    if sess:
                        sess.code = code
                        sess.error = error
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='font-family:system-ui;margin:40px;'>"
                b"<h2>Slack sign-in complete</h2>"
                b"<p>You can close this window and return to GravexStudio.</p>"
                b"</body></html>"
            )

        def log_message(self, format, *args):  # noqa: A003
            return

    return SlackCallbackHandler


def _ensure_slack_server(redirect_uri: str) -> None:
    global _slack_server_started, _slack_server_error, _slack_server_host, _slack_server_port, _slack_server_path
    parsed = urlparse(redirect_uri)
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 80)
    path = parsed.path or "/auth/callback"
    if parsed.scheme not in {"http", ""}:
        _slack_server_error = "SLACK_OAUTH_REDIRECT_URI must use http:// for local callback."
        return
    if _slack_server_started:
        if _slack_server_host == host and _slack_server_port == port and _slack_server_path == path:
            _slack_server_error = None
            return
        _slack_server_error = (
            "Slack OAuth callback server is already running with a different redirect URI. "
            "Restart GravexStudio to change it."
        )
        return
    try:
        httpd = ThreadingHTTPServer((host, port), _make_slack_handler(path))
    except Exception as exc:
        _slack_server_error = str(exc)
        return
    _slack_server_started = True
    _slack_server_error = None
    _slack_server_host = host
    _slack_server_port = port
    _slack_server_path = path

    def _serve():
        try:
            httpd.serve_forever()
        except Exception:
            pass

    t = threading.Thread(target=_serve, daemon=True, name="slack-oauth-callback")
    t.start()


class DatabaseCredentialTestRequest(BaseModel):
    id: str = ""
    nickname: str = ""
    engine: str = "postgresql"
    host: str = ""
    port: str = ""
    database: str = ""
    user: str = ""
    password: str = ""
    options: str = ""
    server_ca: str = ""
    client_cert: str = ""
    client_key: str = ""


class SlackOAuthStartRequest(BaseModel):
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = ""
    scope: str = ""


class SlackTokenSetRequest(BaseModel):
    access_token: str = ""
    refresh_token: str = ""
    scope: str = ""


def _parse_options(raw: str) -> dict[str, str]:
    text = str(raw or "").strip().lstrip("?")
    if not text:
        return {}
    out: dict[str, str] = {}
    for k, v in parse_qsl(text, keep_blank_values=True):
        key = str(k or "").strip()
        if not key:
            continue
        out[key] = str(v or "").strip()
    return out


def _default_port(engine: str) -> int:
    norm = str(engine or "").strip().lower()
    if norm in {"postgres", "postgresql"}:
        return 5432
    if norm in {"mysql"}:
        return 3306
    if norm in {"mssql", "sqlserver", "sql_server"}:
        return 1433
    if norm in {"mongodb", "mongo"}:
        return 27017
    if norm in {"redis"}:
        return 6379
    return 0


def _coerce_port(port_raw: str, engine: str) -> int:
    text = str(port_raw or "").strip()
    if text:
        try:
            n = int(text)
        except Exception:
            raise ValueError("Port must be a number.")
        if n < 1 or n > 65535:
            raise ValueError("Port must be between 1 and 65535.")
        return n
    default = _default_port(engine)
    if default > 0:
        return default
    raise ValueError("Port is required for this database engine.")


def _tcp_probe(host: str, port: int, timeout_s: float = 5.0) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True, "TCP reachability check passed."
    except Exception as exc:
        return False, str(exc) or "TCP reachability check failed."


def _build_db_test_result(
    *,
    ok: bool,
    engine: str,
    message: str,
    code: str,
    driver: str = "",
    latency_ms: float = 0.0,
) -> dict:
    return {
        "ok": bool(ok),
        "engine": str(engine or "").strip().lower() or "unknown",
        "driver": driver,
        "code": code,
        "message": message,
        "latency_ms": round(float(latency_ms), 1),
    }


def _test_database_credential(payload: DatabaseCredentialTestRequest) -> dict:
    started = time.perf_counter()
    engine = str(payload.engine or "").strip().lower() or "postgresql"
    host = str(payload.host or "").strip()
    database = str(payload.database or "").strip()
    user = str(payload.user or "").strip()
    password = str(payload.password or "")
    options = _parse_options(payload.options)
    timeout_s = 5.0
    if options.get("connect_timeout"):
        try:
            timeout_s = max(1.0, min(20.0, float(options["connect_timeout"])))
        except Exception:
            timeout_s = 5.0
    if not host:
        raise ValueError("Host is required.")
    port = _coerce_port(payload.port, engine)

    def _done(ok: bool, message: str, code: str, driver: str = "") -> dict:
        return _build_db_test_result(
            ok=ok,
            engine=engine,
            message=message,
            code=code,
            driver=driver,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    if engine in {"postgres", "postgresql"}:
        if not database or not user or not password:
            raise ValueError("PostgreSQL requires database, user, and password.")
        try:
            import psycopg  # type: ignore

            kwargs = {
                "host": host,
                "port": port,
                "dbname": database,
                "user": user,
                "password": password,
                "connect_timeout": int(timeout_s),
            }
            if payload.server_ca:
                kwargs["sslrootcert"] = payload.server_ca
            if payload.client_cert:
                kwargs["sslcert"] = payload.client_cert
            if payload.client_key:
                kwargs["sslkey"] = payload.client_key
            with psycopg.connect(**kwargs) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return _done(True, "PostgreSQL connection succeeded.", "ok", "psycopg")
        except ImportError:
            try:
                import psycopg2  # type: ignore

                kwargs = {
                    "host": host,
                    "port": port,
                    "dbname": database,
                    "user": user,
                    "password": password,
                    "connect_timeout": int(timeout_s),
                }
                if payload.server_ca:
                    kwargs["sslrootcert"] = payload.server_ca
                if payload.client_cert:
                    kwargs["sslcert"] = payload.client_cert
                if payload.client_key:
                    kwargs["sslkey"] = payload.client_key
                conn = psycopg2.connect(**kwargs)
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT 1")
                    cur.fetchone()
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
                return _done(True, "PostgreSQL connection succeeded.", "ok", "psycopg2")
            except ImportError:
                ok, tcp_msg = _tcp_probe(host, port, timeout_s=timeout_s)
                if ok:
                    return _done(
                        False,
                        "PostgreSQL driver missing (install psycopg or psycopg2). TCP connectivity is reachable.",
                        "driver_missing",
                        "tcp",
                    )
                return _done(
                    False,
                    f"PostgreSQL driver missing and host unreachable: {tcp_msg}",
                    "driver_missing_unreachable",
                    "tcp",
                )
            except Exception as exc:
                return _done(False, str(exc) or "PostgreSQL test failed.", "auth_or_network_error", "psycopg2")
        except Exception as exc:
            return _done(False, str(exc) or "PostgreSQL test failed.", "auth_or_network_error", "psycopg")

    if engine in {"mysql"}:
        if not database or not user or not password:
            raise ValueError("MySQL requires database, user, and password.")
        try:
            import pymysql  # type: ignore

            kwargs = {
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "database": database,
                "connect_timeout": int(timeout_s),
                "read_timeout": int(timeout_s),
                "write_timeout": int(timeout_s),
            }
            conn = pymysql.connect(**kwargs)
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            return _done(True, "MySQL connection succeeded.", "ok", "pymysql")
        except ImportError:
            ok, tcp_msg = _tcp_probe(host, port, timeout_s=timeout_s)
            if ok:
                return _done(
                    False,
                    "MySQL driver missing (install pymysql). TCP connectivity is reachable.",
                    "driver_missing",
                    "tcp",
                )
            return _done(
                False,
                f"MySQL driver missing and host unreachable: {tcp_msg}",
                "driver_missing_unreachable",
                "tcp",
            )
        except Exception as exc:
            return _done(False, str(exc) or "MySQL test failed.", "auth_or_network_error", "pymysql")

    if engine in {"mssql", "sqlserver", "sql_server"}:
        if not database or not user or not password:
            raise ValueError("MS SQL requires database, user, and password.")
        try:
            import pymssql  # type: ignore

            conn = pymssql.connect(
                server=host,
                port=port,
                user=user,
                password=password,
                database=database,
                login_timeout=int(timeout_s),
                timeout=int(timeout_s),
            )
            try:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.fetchone()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            return _done(True, "MS SQL connection succeeded.", "ok", "pymssql")
        except ImportError:
            ok, tcp_msg = _tcp_probe(host, port, timeout_s=timeout_s)
            if ok:
                return _done(
                    False,
                    "MS SQL driver missing (install pymssql). TCP connectivity is reachable.",
                    "driver_missing",
                    "tcp",
                )
            return _done(
                False,
                f"MS SQL driver missing and host unreachable: {tcp_msg}",
                "driver_missing_unreachable",
                "tcp",
            )
        except Exception as exc:
            return _done(False, str(exc) or "MS SQL test failed.", "auth_or_network_error", "pymssql")

    if engine in {"mongodb", "mongo"}:
        try:
            import pymongo  # type: ignore

            auth = ""
            if user:
                auth = quote_plus(user)
                if password:
                    auth = f"{auth}:{quote_plus(password)}"
                auth += "@"
            db_path = f"/{database}" if database else ""
            query = payload.options.strip().lstrip("?")
            uri = f"mongodb://{auth}{host}:{port}{db_path}"
            if query:
                uri += f"?{query}"
            client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=int(timeout_s * 1000))
            try:
                client.admin.command("ping")
            finally:
                try:
                    client.close()
                except Exception:
                    pass
            return _done(True, "MongoDB connection succeeded.", "ok", "pymongo")
        except ImportError:
            ok, tcp_msg = _tcp_probe(host, port, timeout_s=timeout_s)
            if ok:
                return _done(
                    False,
                    "MongoDB driver missing (install pymongo). TCP connectivity is reachable.",
                    "driver_missing",
                    "tcp",
                )
            return _done(
                False,
                f"MongoDB driver missing and host unreachable: {tcp_msg}",
                "driver_missing_unreachable",
                "tcp",
            )
        except Exception as exc:
            return _done(False, str(exc) or "MongoDB test failed.", "auth_or_network_error", "pymongo")

    if engine == "redis":
        try:
            import redis  # type: ignore

            db_index = 0
            if database.strip():
                try:
                    db_index = int(database.strip())
                except Exception:
                    db_index = 0
            client = redis.Redis(
                host=host,
                port=port,
                db=db_index,
                username=user or None,
                password=password or None,
                socket_connect_timeout=timeout_s,
                socket_timeout=timeout_s,
            )
            client.ping()
            return _done(True, "Redis connection succeeded.", "ok", "redis-py")
        except ImportError:
            ok, tcp_msg = _tcp_probe(host, port, timeout_s=timeout_s)
            if ok:
                return _done(
                    False,
                    "Redis driver missing (install redis). TCP connectivity is reachable.",
                    "driver_missing",
                    "tcp",
                )
            return _done(
                False,
                f"Redis driver missing and host unreachable: {tcp_msg}",
                "driver_missing_unreachable",
                "tcp",
            )
        except Exception as exc:
            return _done(False, str(exc) or "Redis test failed.", "auth_or_network_error", "redis-py")

    ok, tcp_msg = _tcp_probe(host, port, timeout_s=timeout_s)
    if ok:
        return _done(
            True,
            "TCP reachability check passed. Engine-specific auth test is not implemented for this engine yet.",
            "tcp_only",
            "tcp",
        )
    return _done(False, f"TCP reachability failed: {tcp_msg}", "tcp_unreachable", "tcp")


def register(app, ctx) -> None:
    router = APIRouter()

    def _auth_obj(raw: str) -> dict:
        text = (raw or "").strip() or "{}"
        try:
            obj = json.loads(text)
        except Exception:
            return {}
        if not isinstance(obj, dict):
            return {}
        return obj

    def _gmail_public_payload(gmail_obj: dict) -> dict:
        return {
            "connected": bool(gmail_obj.get("connected")),
            "account_email": str(gmail_obj.get("account_email") or "").strip(),
            "scope": str(gmail_obj.get("scope") or "").strip(),
            "expires_at": gmail_obj.get("expires_at"),
            "connected_at": str(gmail_obj.get("connected_at") or "").strip(),
            "error": str(gmail_obj.get("error") or "").strip(),
        }

    def _slack_public_payload(slack_obj: dict) -> dict:
        return {
            "connected": bool(slack_obj.get("connected")),
            "workspace_name": str(slack_obj.get("workspace_name") or "").strip(),
            "workspace_id": str(slack_obj.get("workspace_id") or "").strip(),
            "bot_user_id": str(slack_obj.get("bot_user_id") or "").strip(),
            "scope": str(slack_obj.get("scope") or "").strip(),
            "expires_at": slack_obj.get("expires_at"),
            "connected_at": str(slack_obj.get("connected_at") or "").strip(),
            "error": str(slack_obj.get("error") or "").strip(),
        }

    def _apply_gmail_auth_json(bot_auth_json: str, *, tokens: dict, account_email: str) -> str:
        base = _auth_obj(bot_auth_json)
        connected_apps = base.get("connected_apps")
        if not isinstance(connected_apps, dict):
            connected_apps = {}
        old_gmail = connected_apps.get("gmail")
        old_gmail_obj = old_gmail if isinstance(old_gmail, dict) else {}
        refresh_token = str(tokens.get("refresh_token") or old_gmail_obj.get("refresh_token") or "").strip()
        access_token = str(tokens.get("access_token") or "").strip()
        expires_at = None
        try:
            expires_in = float(tokens.get("expires_in") or 0.0)
            if expires_in > 0:
                expires_at = time.time() + expires_in
        except Exception:
            expires_at = old_gmail_obj.get("expires_at")
        scope = str(tokens.get("scope") or old_gmail_obj.get("scope") or "").strip()
        now_iso = ctx.dt.datetime.now(ctx.dt.timezone.utc).isoformat()
        gmail_obj = {
            "connected": bool(refresh_token or access_token),
            "account_email": str(account_email or old_gmail_obj.get("account_email") or "").strip(),
            "scope": scope,
            "token_type": str(tokens.get("token_type") or old_gmail_obj.get("token_type") or "").strip(),
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
            "connected_at": str(old_gmail_obj.get("connected_at") or now_iso).strip(),
            "error": "",
        }
        connected_apps["gmail"] = gmail_obj
        base["connected_apps"] = connected_apps
        base["gmail_integration"] = gmail_obj
        if gmail_obj.get("account_email"):
            base["gmail_account_email"] = gmail_obj.get("account_email")
        else:
            base.pop("gmail_account_email", None)
        if gmail_obj.get("scope"):
            base["gmail_scope"] = gmail_obj.get("scope")
        else:
            base.pop("gmail_scope", None)
        if gmail_obj.get("token_type"):
            base["gmail_token_type"] = gmail_obj.get("token_type")
        else:
            base.pop("gmail_token_type", None)
        if gmail_obj.get("access_token"):
            base["gmail_access_token"] = gmail_obj.get("access_token")
        else:
            base.pop("gmail_access_token", None)
        if gmail_obj.get("refresh_token"):
            base["gmail_refresh_token"] = gmail_obj.get("refresh_token")
        else:
            base.pop("gmail_refresh_token", None)
        if isinstance(gmail_obj.get("expires_at"), (int, float)):
            base["gmail_expires_at"] = gmail_obj.get("expires_at")
        else:
            base.pop("gmail_expires_at", None)
        if gmail_obj.get("connected_at"):
            base["gmail_connected_at"] = gmail_obj.get("connected_at")
        else:
            base.pop("gmail_connected_at", None)
        base["gmail_connected"] = bool(gmail_obj.get("connected"))
        base.pop("gmail_error", None)
        base.pop("gmail_client_id", None)
        base.pop("gmail_client_secret", None)
        base.pop("gmail_sender_email", None)
        base.pop("gmail_reply_to_email", None)
        return json.dumps(base, ensure_ascii=False, indent=2)

    def _apply_slack_auth_json(
        bot_auth_json: str,
        *,
        tokens: dict,
        auth_test: dict,
        oauth_client_id: str,
        oauth_client_secret: str,
        oauth_redirect_uri: str,
        oauth_scope: str,
    ) -> str:
        base = _auth_obj(bot_auth_json)
        connected_apps = base.get("connected_apps")
        if not isinstance(connected_apps, dict):
            connected_apps = {}
        old_slack = connected_apps.get("slack")
        old_slack_obj = old_slack if isinstance(old_slack, dict) else {}
        access_token = str(tokens.get("access_token") or old_slack_obj.get("access_token") or "").strip()
        refresh_token = str(tokens.get("refresh_token") or old_slack_obj.get("refresh_token") or "").strip()
        scope = str(tokens.get("scope") or old_slack_obj.get("scope") or "").strip()
        expires_at = None
        try:
            expires_in = float(tokens.get("expires_in") or 0.0)
            if expires_in > 0:
                expires_at = time.time() + expires_in
        except Exception:
            expires_at = old_slack_obj.get("expires_at")
        team_obj = tokens.get("team") if isinstance(tokens.get("team"), dict) else {}
        workspace_name = str(
            auth_test.get("team")
            or (team_obj.get("name") if isinstance(team_obj, dict) else "")
            or old_slack_obj.get("workspace_name")
            or ""
        ).strip()
        workspace_id = str(
            auth_test.get("team_id")
            or (team_obj.get("id") if isinstance(team_obj, dict) else "")
            or old_slack_obj.get("workspace_id")
            or ""
        ).strip()
        bot_user_id = str(auth_test.get("user_id") or old_slack_obj.get("bot_user_id") or "").strip()
        now_iso = ctx.dt.datetime.now(ctx.dt.timezone.utc).isoformat()
        slack_obj = {
            "connected": bool(access_token),
            "workspace_name": workspace_name,
            "workspace_id": workspace_id,
            "bot_user_id": bot_user_id,
            "oauth_client_id": str(oauth_client_id or old_slack_obj.get("oauth_client_id") or "").strip(),
            "oauth_client_secret": str(oauth_client_secret or old_slack_obj.get("oauth_client_secret") or "").strip(),
            "oauth_redirect_uri": str(oauth_redirect_uri or old_slack_obj.get("oauth_redirect_uri") or "").strip(),
            "oauth_scope": str(oauth_scope or old_slack_obj.get("oauth_scope") or "").strip(),
            "scope": scope,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
            "connected_at": str(old_slack_obj.get("connected_at") or now_iso).strip(),
            "error": "",
        }
        connected_apps["slack"] = slack_obj
        base["connected_apps"] = connected_apps
        base["slack_integration"] = slack_obj
        if slack_obj.get("workspace_name"):
            base["slack_workspace"] = slack_obj.get("workspace_name")
        else:
            base.pop("slack_workspace", None)
        if slack_obj.get("workspace_id"):
            base["slack_workspace_id"] = slack_obj.get("workspace_id")
        else:
            base.pop("slack_workspace_id", None)
        if slack_obj.get("bot_user_id"):
            base["slack_bot_user_id"] = slack_obj.get("bot_user_id")
        else:
            base.pop("slack_bot_user_id", None)
        if slack_obj.get("oauth_client_id"):
            base["slack_oauth_client_id"] = slack_obj.get("oauth_client_id")
        else:
            base.pop("slack_oauth_client_id", None)
        if slack_obj.get("oauth_client_secret"):
            base["slack_oauth_client_secret"] = slack_obj.get("oauth_client_secret")
        else:
            base.pop("slack_oauth_client_secret", None)
        if slack_obj.get("oauth_redirect_uri"):
            base["slack_oauth_redirect_uri"] = slack_obj.get("oauth_redirect_uri")
        else:
            base.pop("slack_oauth_redirect_uri", None)
        if slack_obj.get("oauth_scope"):
            base["slack_oauth_scope"] = slack_obj.get("oauth_scope")
        else:
            base.pop("slack_oauth_scope", None)
        if slack_obj.get("scope"):
            base["slack_scope"] = slack_obj.get("scope")
        else:
            base.pop("slack_scope", None)
        if slack_obj.get("access_token"):
            base["slack_access_token"] = slack_obj.get("access_token")
        else:
            base.pop("slack_access_token", None)
        if slack_obj.get("refresh_token"):
            base["slack_refresh_token"] = slack_obj.get("refresh_token")
        else:
            base.pop("slack_refresh_token", None)
        if isinstance(slack_obj.get("expires_at"), (int, float)):
            base["slack_expires_at"] = slack_obj.get("expires_at")
        else:
            base.pop("slack_expires_at", None)
        if slack_obj.get("connected_at"):
            base["slack_connected_at"] = slack_obj.get("connected_at")
        else:
            base.pop("slack_connected_at", None)
        base["slack_connected"] = bool(slack_obj.get("connected"))
        base.pop("slack_error", None)
        base.pop("slack_client_id", None)
        base.pop("slack_client_secret", None)
        return json.dumps(base, ensure_ascii=False, indent=2)

    @router.get("/api/bots")
    def api_list_bots(session: Session = Depends(ctx.get_session)) -> dict:
        bots = ctx.list_bots(session)
        stats = ctx.bots_aggregate_metrics(session)
        items = []
        for b in bots:
            d = ctx._bot_to_dict(b)
            d["stats"] = stats.get(
                b.id,
                {
                    "conversations": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "avg_llm_ttfb_ms": None,
                    "avg_llm_total_ms": None,
                    "avg_total_ms": None,
                },
            )
            items.append(d)
        return {"items": items}

    @router.post("/api/bots")
    def api_create_bot(payload: BotCreateRequest, session: Session = Depends(ctx.get_session)) -> dict:
        default_provider = (ctx._get_app_setting(session, "default_llm_provider") or "").strip().lower()
        default_model = (ctx._get_app_setting(session, "default_llm_model") or "").strip()
        provider = ctx._normalize_llm_provider(payload.llm_provider or default_provider or "openai")
        openai_model = (payload.openai_model or default_model or "o4-mini").strip() or "o4-mini"
        bot = ctx.Bot(
            name=payload.name,
            llm_provider=provider,
            openai_model=openai_model,
            openai_asr_model=(payload.openai_asr_model or "gpt-4o-mini-transcribe").strip() or "gpt-4o-mini-transcribe",
            web_search_model=(payload.web_search_model or openai_model).strip() or openai_model,
            codex_model=(payload.codex_model or "gpt-5.1-codex-mini").strip() or "gpt-5.1-codex-mini",
            summary_model=(payload.summary_model or openai_model or "gpt-5-nano").strip() or (openai_model or "gpt-5-nano"),
            history_window_turns=int(payload.history_window_turns or 16),
            enable_data_agent=bool(getattr(payload, "enable_data_agent", False)),
            data_agent_api_spec_text=(payload.data_agent_api_spec_text or ""),
            data_agent_auth_json=(payload.data_agent_auth_json or "{}"),
            data_agent_system_prompt=(payload.data_agent_system_prompt or ""),
            data_agent_return_result_directly=bool(getattr(payload, "data_agent_return_result_directly", False)),
            data_agent_prewarm_on_start=bool(getattr(payload, "data_agent_prewarm_on_start", False)),
            data_agent_prewarm_prompt=(payload.data_agent_prewarm_prompt or ""),
            data_agent_model=(payload.data_agent_model or "gpt-5.2").strip() or "gpt-5.2",
            data_agent_reasoning_effort=(payload.data_agent_reasoning_effort or "high").strip() or "high",
            enable_host_actions=bool(getattr(payload, "enable_host_actions", False)),
            enable_host_shell=bool(getattr(payload, "enable_host_shell", False)),
            require_host_action_approval=bool(getattr(payload, "require_host_action_approval", False)),
            system_prompt=payload.system_prompt,
            language=payload.language,
            openai_tts_model=(payload.openai_tts_model or "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts",
            openai_tts_voice=(payload.openai_tts_voice or "alloy").strip() or "alloy",
            openai_tts_speed=float(payload.openai_tts_speed or 1.0),
            start_message_mode=(payload.start_message_mode or "llm").strip() or "llm",
            start_message_text=payload.start_message_text or "",
        )
        ctx.create_bot(session, bot)
        return ctx._bot_to_dict(bot)

    @router.get("/api/bots/{bot_id}")
    def api_get_bot(bot_id: UUID, session: Session = Depends(ctx.get_session)) -> dict:
        bot = ctx.get_bot(session, bot_id)
        return ctx._bot_to_dict(bot)

    @router.put("/api/bots/{bot_id}")
    def api_update_bot(bot_id: UUID, payload: BotUpdateRequest, session: Session = Depends(ctx.get_session)) -> dict:
        patch = {}
        for k, v in payload.model_dump(exclude_unset=True).items():
            if k == "llm_provider":
                raw = (v or "").strip().lower()
                if raw and raw not in ("openai", "openrouter", "local", "chatgpt"):
                    raise ctx.HTTPException(status_code=400, detail="Unsupported LLM provider.")
                patch[k] = raw or "openai"
            elif k in (
                "openai_tts_model",
                "openai_tts_voice",
                "web_search_model",
                "codex_model",
                "summary_model",
                "openai_asr_model",
            ):
                patch[k] = (v or "").strip()
            elif k == "openai_tts_speed":
                patch[k] = float(v) if v is not None else 1.0
            elif k == "history_window_turns":
                try:
                    n = int(v) if v is not None else 16
                except Exception:
                    n = 16
                if n < 1:
                    n = 1
                patch[k] = n
            elif k in ("data_agent_model", "data_agent_reasoning_effort"):
                patch[k] = (v or "").strip()
            elif k == "disabled_tools":
                patch[k] = sorted({str(x) for x in (v or []) if str(x).strip()})
            else:
                patch[k] = v
        bot = ctx.update_bot(session, bot_id=bot_id, patch=patch)
        return ctx._bot_to_dict(bot)

    @router.delete("/api/bots/{bot_id}")
    def api_delete_bot(bot_id: UUID, session: Session = Depends(ctx.get_session)) -> dict:
        ctx.delete_bot(session, bot_id)
        return {"ok": True}

    @router.get("/api/bots/{bot_id}/tools")
    def api_list_tools(bot_id: UUID, session: Session = Depends(ctx.get_session)) -> dict:
        bot = ctx.get_bot(session, bot_id)
        tools = ctx.list_integration_tools(session, bot_id=bot_id)
        disabled = ctx._disabled_tool_names(bot)
        system_tools = ctx._system_tools_public_list(bot=bot, disabled=disabled)
        return {"items": [ctx._tool_to_dict(t) for t in tools], "system_tools": system_tools}

    @router.post("/api/bots/{bot_id}/tools")
    def api_create_tool(
        bot_id: UUID,
        payload: IntegrationToolCreateRequest,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        bot = ctx.get_bot(session, bot_id)
        tool = ctx.IntegrationTool(
            bot_id=bot.id,
            name=payload.name,
            description=payload.description,
            url=payload.url,
            method=payload.method,
            use_codex_response=payload.use_codex_response,
            enabled=payload.enabled,
            args_required_json=ctx.json.dumps(payload.args_required or [], ensure_ascii=False),
            headers_template_json=payload.headers_template_json,
            request_body_template=payload.request_body_template,
            parameters_schema_json=payload.parameters_schema_json,
            response_schema_json=payload.response_schema_json,
            codex_prompt=payload.codex_prompt,
            postprocess_python=payload.postprocess_python,
            return_result_directly=payload.return_result_directly,
            response_mapper_json=payload.response_mapper_json,
            pagination_json=payload.pagination_json,
            static_reply_template=payload.static_reply_template,
        )
        ctx.create_integration_tool(session, tool)
        return ctx._tool_to_dict(tool)

    @router.put("/api/bots/{bot_id}/tools/{tool_id}")
    def api_update_tool(
        bot_id: UUID,
        tool_id: UUID,
        payload: IntegrationToolUpdateRequest,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        tool = ctx.get_integration_tool(session, tool_id)
        if tool.bot_id != bot_id:
            raise ctx.HTTPException(status_code=404, detail="Tool not found")
        patch = payload.model_dump(exclude_unset=True)
        if "args_required" in patch:
            patch["args_required_json"] = ctx.json.dumps(patch.pop("args_required") or [], ensure_ascii=False)
        if "headers_template_json" in patch and patch["headers_template_json"] is None:
            patch["headers_template_json"] = "{}"
        tool = ctx.update_integration_tool(session, tool_id, patch)
        return ctx._tool_to_dict(tool)

    @router.delete("/api/bots/{bot_id}/tools/{tool_id}")
    def api_delete_tool(bot_id: UUID, tool_id: UUID, session: Session = Depends(ctx.get_session)) -> dict:
        tool = ctx.get_integration_tool(session, tool_id)
        if tool.bot_id != bot_id:
            raise ctx.HTTPException(status_code=404, detail="Tool not found")
        ctx.delete_integration_tool(session, tool_id)
        return {"ok": True}

    @router.post("/api/bots/{bot_id}/connected-apps/gmail/oauth/start")
    def api_gmail_oauth_start(bot_id: UUID, session: Session = Depends(ctx.get_session)) -> dict:
        _ = ctx.get_bot(session, bot_id)
        if not _gmail_client_id() or not _gmail_client_secret():
            raise ctx.HTTPException(
                status_code=400,
                detail="Gmail OAuth is not configured. Set GMAIL_OAUTH_CLIENT_ID and GMAIL_OAUTH_CLIENT_SECRET.",
            )
        _ensure_gmail_server()
        if _gmail_server_error:
            raise ctx.HTTPException(status_code=500, detail=f"Gmail OAuth callback server failed: {_gmail_server_error}")
        state = secrets.token_urlsafe(24)
        verifier, challenge = _gmail_pkce_pair()
        with _gmail_sessions_lock:
            _gmail_sessions[state] = GmailOAuthSession(
                state=state,
                bot_id=str(bot_id),
                code_verifier=verifier,
                created_at=time.time(),
            )
        return {"state": state, "auth_url": _gmail_auth_url(state, challenge)}

    @router.get("/api/bots/{bot_id}/connected-apps/gmail/oauth/status")
    def api_gmail_oauth_status(bot_id: UUID, state: str, session: Session = Depends(ctx.get_session)) -> dict:
        bot = ctx.get_bot(session, bot_id)
        with _gmail_sessions_lock:
            sess = _gmail_sessions.get(state)
        if not sess:
            return {"status": "expired"}
        if sess.bot_id != str(bot_id):
            return {"status": "expired"}
        if sess.error:
            with _gmail_sessions_lock:
                _gmail_sessions.pop(state, None)
            return {"status": "error", "error": sess.error}
        if not sess.code:
            if (time.time() - sess.created_at) > 600:
                with _gmail_sessions_lock:
                    _gmail_sessions.pop(state, None)
                return {"status": "expired"}
            return {"status": "pending"}
        with _gmail_sessions_lock:
            consumed = _gmail_sessions.pop(state, None)
        if not consumed:
            return {"status": "expired"}
        try:
            tokens = _gmail_exchange_code(consumed.code or "", consumed.code_verifier)
            userinfo = _gmail_userinfo(str(tokens.get("access_token") or ""))
            account_email = str(userinfo.get("email") or "").strip()
            next_auth_json = _apply_gmail_auth_json(
                getattr(bot, "data_agent_auth_json", "") or "{}",
                tokens=tokens,
                account_email=account_email,
            )
            bot = ctx.update_bot(session, bot_id=bot_id, patch={"data_agent_auth_json": next_auth_json})
            auth_obj = _auth_obj(getattr(bot, "data_agent_auth_json", "") or "{}")
            connected_apps = auth_obj.get("connected_apps")
            gmail_obj = connected_apps.get("gmail") if isinstance(connected_apps, dict) else {}
            public_payload = _gmail_public_payload(gmail_obj if isinstance(gmail_obj, dict) else {})
            return {"status": "ready", **public_payload}
        except Exception as exc:
            return {"status": "error", "error": str(exc) or "Gmail sign-in failed."}

    @router.post("/api/bots/{bot_id}/connected-apps/slack/oauth/start")
    def api_slack_oauth_start(
        bot_id: UUID,
        payload: Optional[SlackOAuthStartRequest] = Body(None),
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        bot = ctx.get_bot(session, bot_id)
        auth_obj = _auth_obj(getattr(bot, "data_agent_auth_json", "") or "{}")
        connected_apps_obj = auth_obj.get("connected_apps")
        connected_apps = connected_apps_obj if isinstance(connected_apps_obj, dict) else {}
        slack_cfg_obj = connected_apps.get("slack")
        slack_cfg = slack_cfg_obj if isinstance(slack_cfg_obj, dict) else {}
        client_id = str(
            (payload.client_id if payload else "")
            or slack_cfg.get("oauth_client_id")
            or auth_obj.get("slack_oauth_client_id")
            or _slack_client_id()
            or ""
        ).strip()
        client_secret = str(
            (payload.client_secret if payload else "")
            or slack_cfg.get("oauth_client_secret")
            or auth_obj.get("slack_oauth_client_secret")
            or _slack_client_secret()
            or ""
        ).strip()
        redirect_uri = str(
            (payload.redirect_uri if payload else "")
            or slack_cfg.get("oauth_redirect_uri")
            or auth_obj.get("slack_oauth_redirect_uri")
            or _slack_redirect_uri()
            or SLACK_DEFAULT_REDIRECT_URI
        ).strip()
        scope = str(
            (payload.scope if payload else "")
            or slack_cfg.get("oauth_scope")
            or auth_obj.get("slack_oauth_scope")
            or _slack_scope()
            or SLACK_DEFAULT_SCOPE
        ).strip()
        if not client_id or not client_secret:
            raise ctx.HTTPException(
                status_code=400,
                detail="Slack OAuth requires Client ID and Client Secret. Add them in Connected apps > Slack.",
            )
        _ensure_slack_server(redirect_uri)
        if _slack_server_error:
            raise ctx.HTTPException(status_code=500, detail=f"Slack OAuth callback server failed: {_slack_server_error}")
        state = secrets.token_urlsafe(24)
        with _slack_sessions_lock:
            _slack_sessions[state] = SlackOAuthSession(
                state=state,
                bot_id=str(bot_id),
                created_at=time.time(),
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=scope,
            )
        return {"state": state, "auth_url": _slack_auth_url(state=state, client_id=client_id, redirect_uri=redirect_uri, scope=scope)}

    @router.get("/api/bots/{bot_id}/connected-apps/slack/oauth/status")
    def api_slack_oauth_status(bot_id: UUID, state: str, session: Session = Depends(ctx.get_session)) -> dict:
        bot = ctx.get_bot(session, bot_id)
        with _slack_sessions_lock:
            sess = _slack_sessions.get(state)
        if not sess:
            return {"status": "expired"}
        if sess.bot_id != str(bot_id):
            return {"status": "expired"}
        if sess.error:
            with _slack_sessions_lock:
                _slack_sessions.pop(state, None)
            return {"status": "error", "error": sess.error}
        if not sess.code:
            if (time.time() - sess.created_at) > 600:
                with _slack_sessions_lock:
                    _slack_sessions.pop(state, None)
                return {"status": "expired"}
            return {"status": "pending"}
        with _slack_sessions_lock:
            consumed = _slack_sessions.pop(state, None)
        if not consumed:
            return {"status": "expired"}
        try:
            tokens = _slack_exchange_code(
                code=consumed.code or "",
                client_id=consumed.client_id,
                client_secret=consumed.client_secret,
                redirect_uri=consumed.redirect_uri,
            )
            auth_test = _slack_auth_test(str(tokens.get("access_token") or ""))
            next_auth_json = _apply_slack_auth_json(
                getattr(bot, "data_agent_auth_json", "") or "{}",
                tokens=tokens,
                auth_test=auth_test,
                oauth_client_id=consumed.client_id,
                oauth_client_secret=consumed.client_secret,
                oauth_redirect_uri=consumed.redirect_uri,
                oauth_scope=consumed.scope,
            )
            bot = ctx.update_bot(session, bot_id=bot_id, patch={"data_agent_auth_json": next_auth_json})
            auth_obj = _auth_obj(getattr(bot, "data_agent_auth_json", "") or "{}")
            connected_apps = auth_obj.get("connected_apps")
            slack_obj = connected_apps.get("slack") if isinstance(connected_apps, dict) else {}
            public_payload = _slack_public_payload(slack_obj if isinstance(slack_obj, dict) else {})
            return {"status": "ready", **public_payload}
        except Exception as exc:
            return {"status": "error", "error": str(exc) or "Slack sign-in failed."}

    @router.post("/api/bots/{bot_id}/connected-apps/slack/token/set")
    def api_slack_token_set(
        bot_id: UUID,
        payload: SlackTokenSetRequest,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        bot = ctx.get_bot(session, bot_id)
        access_token = str(payload.access_token or "").strip()
        refresh_token = str(payload.refresh_token or "").strip()
        scope = str(payload.scope or "").strip()
        if not access_token:
            raise ctx.HTTPException(status_code=400, detail="access_token is required.")
        try:
            auth_test = _slack_auth_test(access_token)
            auth_obj = _auth_obj(getattr(bot, "data_agent_auth_json", "") or "{}")
            connected_apps_obj = auth_obj.get("connected_apps")
            connected_apps = connected_apps_obj if isinstance(connected_apps_obj, dict) else {}
            old_slack_obj = connected_apps.get("slack")
            old_slack = old_slack_obj if isinstance(old_slack_obj, dict) else {}
            oauth_client_id = str(
                old_slack.get("oauth_client_id")
                or auth_obj.get("slack_oauth_client_id")
                or ""
            ).strip()
            oauth_client_secret = str(
                old_slack.get("oauth_client_secret")
                or auth_obj.get("slack_oauth_client_secret")
                or ""
            ).strip()
            oauth_redirect_uri = str(
                old_slack.get("oauth_redirect_uri")
                or auth_obj.get("slack_oauth_redirect_uri")
                or _slack_redirect_uri()
                or SLACK_DEFAULT_REDIRECT_URI
            ).strip()
            oauth_scope = str(
                old_slack.get("oauth_scope")
                or auth_obj.get("slack_oauth_scope")
                or _slack_scope()
                or SLACK_DEFAULT_SCOPE
            ).strip()
            tokens_obj = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "scope": scope,
            }
            next_auth_json = _apply_slack_auth_json(
                getattr(bot, "data_agent_auth_json", "") or "{}",
                tokens=tokens_obj,
                auth_test=auth_test,
                oauth_client_id=oauth_client_id,
                oauth_client_secret=oauth_client_secret,
                oauth_redirect_uri=oauth_redirect_uri,
                oauth_scope=oauth_scope,
            )
            bot = ctx.update_bot(session, bot_id=bot_id, patch={"data_agent_auth_json": next_auth_json})
            auth_obj = _auth_obj(getattr(bot, "data_agent_auth_json", "") or "{}")
            connected_apps = auth_obj.get("connected_apps")
            slack_obj = connected_apps.get("slack") if isinstance(connected_apps, dict) else {}
            public_payload = _slack_public_payload(slack_obj if isinstance(slack_obj, dict) else {})
            return {"status": "ready", **public_payload}
        except ctx.HTTPException:
            raise
        except Exception as exc:
            return {"status": "error", "error": str(exc) or "Slack token validation failed."}

    @router.post("/api/bots/{bot_id}/connected-apps/database/test")
    def api_test_database_credential(
        bot_id: UUID,
        payload: DatabaseCredentialTestRequest,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        _ = ctx.get_bot(session, bot_id)
        try:
            return _test_database_credential(payload)
        except ValueError as exc:
            return _build_db_test_result(
                ok=False,
                engine=str(payload.engine or "").strip(),
                message=str(exc) or "Invalid credential payload.",
                code="invalid_payload",
            )
        except Exception as exc:
            return _build_db_test_result(
                ok=False,
                engine=str(payload.engine or "").strip(),
                message=str(exc) or "Database connection test failed.",
                code="test_failed",
            )

    app.include_router(router)
