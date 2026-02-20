from __future__ import annotations

import json
from typing import Any, Optional

from sqlmodel import Session, select

from voicebot.models import AppSetting


def mask_secret(value: str, *, keep_start: int = 10, keep_end: int = 6) -> str:
    v = value or ""
    if len(v) <= keep_start + keep_end + 3:
        return "***"
    return f"{v[:keep_start]}...{v[-keep_end:]}"


def mask_headers_json(headers_json: str) -> str:
    try:
        obj = json.loads(headers_json or "{}")
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    out: dict[str, Any] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, str):
            out[k] = v
            continue
        if k.lower() == "authorization":
            vv = v.strip()
            if vv.lower().startswith("bearer "):
                token = vv.split(" ", 1)[1].strip()
                out[k] = f"Bearer {mask_secret(token)}"
            else:
                out[k] = mask_secret(vv)
        else:
            out[k] = v
    try:
        return json.dumps(out, ensure_ascii=False)
    except Exception:
        return ""


def headers_configured(headers_json: str) -> bool:
    try:
        obj = json.loads(headers_json or "{}")
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    return bool(obj)


def get_app_setting(session: Session, key: str) -> Optional[str]:
    stmt = select(AppSetting).where(AppSetting.key == key).limit(1)
    rec = session.exec(stmt).first()
    return rec.value if rec else None


def set_app_setting(session: Session, key: str, value: str) -> None:
    stmt = select(AppSetting).where(AppSetting.key == key).limit(1)
    rec = session.exec(stmt).first()
    if rec:
        rec.value = value
        session.add(rec)
        session.commit()
        return
    rec = AppSetting(key=key, value=value)
    session.add(rec)
    session.commit()


def download_url_for_token(download_base_url: str, token: str) -> str:
    base = (download_base_url or "").strip()
    if not base:
        return f"/api/downloads/{token}"
    if not (base.startswith("http://") or base.startswith("https://")):
        base = "http://" + base
    base = base.rstrip("/")
    return f"{base}/api/downloads/{token}"


def read_key_from_env_file(env_key: str) -> str:
    env_key = str(env_key or "").strip()
    if not env_key:
        return ""
    try:
        from dotenv import dotenv_values
    except Exception:
        dotenv_values = None
    if dotenv_values is not None:
        try:
            v = dotenv_values(".env").get(env_key) or ""
            return str(v).strip()
        except Exception:
            pass
    try:
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{env_key}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""
