from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import HTTPException
from sqlmodel import Session

from voicebot.store import get_git_token
from voicebot.web.deps import require_crypto


def normalize_git_provider(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p in ("github", "gh"):
        return "github"
    raise HTTPException(status_code=400, detail="Unsupported provider")


def get_git_token_plaintext(session: Session, *, provider: str) -> str:
    try:
        crypto = require_crypto()
    except Exception:
        return ""
    rec = get_git_token(session, provider=provider)
    if not rec:
        return ""
    try:
        return crypto.decrypt_str(rec.token_ciphertext)
    except Exception:
        return ""


def parse_auth_json(auth_json: str) -> dict[str, Any]:
    try:
        obj = json.loads((auth_json or "").strip() or "{}")
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    return obj


def git_auth_mode(auth_json: str) -> str:
    obj = parse_auth_json(auth_json)
    method = str(obj.get("git_auth_method") or "").strip().lower()
    if method in ("ssh", "ssh-key", "ssh_key"):
        return "ssh"
    if method in ("token", "pat", "github_token", "github_pat"):
        return "token"
    for key in (
        "ssh_private_key",
        "ssh_private_key_path",
        "ssh_private_key_b64",
        "ssh_private_key_base64",
        "ssh_key",
        "ssh_key_path",
    ):
        if str(obj.get(key) or "").strip():
            return "ssh"
    return "token"


def merge_git_token_auth(auth_json: str, git_token: str) -> str:
    if not git_token or git_auth_mode(auth_json) != "token":
        return (auth_json or "{}").strip() or "{}"
    obj = parse_auth_json(auth_json)
    if not obj.get("github_token"):
        obj["github_token"] = git_token
    if not obj.get("GITHUB_TOKEN"):
        obj["GITHUB_TOKEN"] = git_token
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return (auth_json or "{}").strip() or "{}"


async def validate_github_token(token: str) -> tuple[bool, str | None]:
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "GravexStudio-VoiceBot",
                },
            )
        if resp.status_code == 200:
            return True, None
        if resp.status_code in (401, 403):
            return False, "Invalid GitHub token"
        return False, f"GitHub validation error (status {resp.status_code})"
    except Exception as exc:
        return False, f"GitHub validation failed: {exc}"
