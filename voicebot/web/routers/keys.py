from __future__ import annotations
from uuid import UUID

from typing import Optional

from fastapi import APIRouter, Depends
from sqlmodel import Session

from ..schemas import ApiKeyCreateRequest, ClientKeyCreateRequest, GitTokenRequest


def register(app, ctx) -> None:
    router = APIRouter()

    @router.get("/api/keys")
    def api_list_keys(provider: Optional[str] = None, session: Session = Depends(ctx.get_session)) -> dict:
        provider = (provider or "").strip().lower() or None
        keys = ctx.list_keys(session, provider=provider)
        items = []
        for k in keys:
            updated_at = getattr(k, "updated_at", None) or k.created_at
            items.append(
                {
                    "id": str(k.id),
                    "provider": k.provider,
                    "name": k.name,
                    "created_at": k.created_at.isoformat(),
                    "updated_at": updated_at.isoformat(),
                }
            )
        return {"items": items}

    @router.post("/api/keys")
    def api_create_key(payload: ApiKeyCreateRequest, session: Session = Depends(ctx.get_session)) -> dict:
        provider = ctx._normalize_llm_provider(payload.provider)
        if provider not in ("openai", "openrouter", "chatgpt"):
            raise ctx.HTTPException(status_code=400, detail="Unsupported provider")
        if not payload.secret:
            raise ctx.HTTPException(status_code=400, detail="Missing API key or token")
        crypto = ctx.require_crypto()
        key = ctx.create_key(session, provider=provider, name=payload.name, secret=payload.secret, crypto=crypto)
        ctx._set_app_setting(session, "default_llm_provider", provider)
        if provider in ("openai", "chatgpt"):
            ctx._set_app_setting(session, "default_llm_model", "gpt-5.2")
        ctx._get_or_create_showcase_bot(session)
        updated_at = getattr(key, "updated_at", None) or key.created_at
        return {
            "id": str(key.id),
            "provider": key.provider,
            "name": key.name,
            "created_at": key.created_at.isoformat(),
            "updated_at": updated_at.isoformat(),
        }

    @router.delete("/api/keys/{key_id}")
    def api_delete_key(key_id: UUID, session: Session = Depends(ctx.get_session)) -> dict:
        ctx.delete_key(session, key_id)
        return {"ok": True}

    @router.get("/api/client-keys")
    def api_list_client_keys(session: Session = Depends(ctx.get_session)) -> dict:
        keys = ctx.list_client_keys(session)
        items = []
        for k in keys:
            updated_at = getattr(k, "updated_at", None) or k.created_at
            items.append(
                {
                    "id": str(k.id),
                    "name": k.name,
                    "allowed_origins": k.allowed_origins or "",
                    "allowed_bot_ids": ctx.json.loads(k.allowed_bot_ids_json or "[]"),
                    "secret": k.secret,
                    "created_at": k.created_at.isoformat(),
                    "updated_at": updated_at.isoformat(),
                }
            )
        return {"items": items}

    @router.post("/api/client-keys")
    def api_create_client_key(payload: ClientKeyCreateRequest, session: Session = Depends(ctx.get_session)) -> dict:
        allowed_bot_ids = [str(x) for x in payload.allowed_bot_ids or [] if str(x).strip()]
        secret = (payload.secret or "").strip() or None
        key = ctx.create_client_key(
            session,
            name=payload.name,
            allowed_origins=payload.allowed_origins or "",
            allowed_bot_ids_json=ctx.json.dumps(allowed_bot_ids, ensure_ascii=False),
            secret=secret,
        )
        updated_at = getattr(key, "updated_at", None) or key.created_at
        return {
            "id": str(key.id),
            "name": key.name,
            "allowed_origins": key.allowed_origins or "",
            "allowed_bot_ids": ctx.json.loads(key.allowed_bot_ids_json or "[]"),
            "secret": key.secret,
            "created_at": key.created_at.isoformat(),
            "updated_at": updated_at.isoformat(),
        }

    @router.delete("/api/client-keys/{key_id}")
    def api_delete_client_key(key_id: UUID, session: Session = Depends(ctx.get_session)) -> dict:
        ctx.delete_client_key(session, key_id)
        return {"ok": True}

    @router.get("/api/user/git-token")
    def api_get_git_token(session: Session = Depends(ctx.get_session)) -> dict:
        provider = ctx._get_app_setting(session, "git_provider") or "github"
        token = ctx._get_git_token_plaintext(session, provider=provider)
        return {"provider": provider, "token": token}

    @router.post("/api/user/git-token")
    def api_set_git_token(payload: GitTokenRequest, session: Session = Depends(ctx.get_session)) -> dict:
        provider = ctx._normalize_git_provider(payload.provider)
        if provider not in ("github", "gitlab"):
            raise ctx.HTTPException(status_code=400, detail="Provider must be github or gitlab")
        if not payload.token:
            raise ctx.HTTPException(status_code=400, detail="Missing token")
        ctx._set_app_setting(session, "git_provider", provider)
        ctx.upsert_git_token(session, provider=provider, token=payload.token, crypto=ctx.require_crypto())
        return {"provider": provider, "token": payload.token}

    app.include_router(router)
