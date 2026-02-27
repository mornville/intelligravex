from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from sqlmodel import Session, delete, select

from ..schemas import LocalSetupRequest
from voicebot.models import (
    AppSetting,
    Bot,
    Conversation,
    ConversationMessage,
    ConversationReadState,
    GitToken,
    HostAction,
    IntegrationTool,
)
from voicebot.web.constants import SHOWCASE_BOT_NAME, SYSTEM_BOT_NAME


def register(app, ctx) -> None:
    router = APIRouter()

    @router.get("/api/options")
    def api_options(session: Session = Depends(ctx.get_session)) -> dict:
        pricing = ctx._get_openai_pricing()
        dynamic_models: list[str] = []
        try:
            now = time.time()
            if (now - float(ctx.openai_models_cache.get("ts") or 0.0)) > 3600.0:
                ctx.openai_models_cache["ts"] = now
                ctx.openai_models_cache["models"] = []
                api_key = (ctx.os.environ.get("OPENAI_API_KEY") or ctx.settings.openai_api_key or "").strip()
                if api_key:
                    try:
                        from openai import OpenAI  # type: ignore

                        client = OpenAI(api_key=api_key)
                        resp = client.models.list()
                        data = getattr(resp, "data", None) or []
                        ids: list[str] = []
                        for m in data:
                            mid = getattr(m, "id", None)
                            if not isinstance(mid, str) or not mid.strip():
                                continue
                            mid = mid.strip()
                            if not (mid.startswith("gpt-") or mid.startswith("o")):
                                continue
                            if mid.startswith(("tts-", "whisper-", "text-embedding-", "omni-moderation", "gpt-4o-mini-tts")):
                                continue
                            ids.append(mid)
                        ctx.openai_models_cache["models"] = sorted(set(ids))
                    except Exception:
                        ctx.openai_models_cache["models"] = []
            dynamic_models = list(ctx.openai_models_cache.get("models") or [])
        except Exception:
            dynamic_models = []

        openai_models = sorted(set(ctx.ui_options.get("openai_models", []) + list(pricing.keys()) + dynamic_models))
        ctx._refresh_openrouter_models_cache(session)
        openrouter_models = list(ctx.openrouter_models_cache.get("models") or [])
        openrouter_pricing = dict(ctx.openrouter_models_cache.get("pricing") or {})
        local_models = ctx.LOCAL_RUNTIME.list_models()
        default_provider = (ctx._get_app_setting(session, "default_llm_provider") or "").strip().lower() or "openai"
        default_model = (ctx._get_app_setting(session, "default_llm_model") or "").strip()
        return {
            "openai_models": openai_models,
            "openai_pricing": {k: {"input_per_1m": v.input_per_1m, "output_per_1m": v.output_per_1m} for k, v in pricing.items()},
            "openrouter_models": openrouter_models,
            "openrouter_pricing": {
                k: {"input_per_1m": v.input_per_1m, "output_per_1m": v.output_per_1m}
                for k, v in openrouter_pricing.items()
            },
            "local_models": local_models,
            "default_llm_provider": default_provider,
            "default_llm_model": default_model,
            "llm_providers": ["openai", "chatgpt", "openrouter", "local"],
            "openai_asr_models": ctx.ui_options.get("openai_asr_models", []),
            "languages": ctx.ui_options.get("languages", []),
            "openai_tts_models": ctx.ui_options.get("openai_tts_models", []),
            "openai_tts_voices": ctx.ui_options.get("openai_tts_voices", []),
            "start_message_modes": ["llm", "static"],
            "http_methods": ["GET", "POST", "PUT", "PATCH", "DELETE"],
        }

    @router.get("/api/status")
    def api_status(session: Session = Depends(ctx.get_session)) -> dict:
        openai_key = bool(ctx._get_openai_api_key(session))
        openrouter_key = bool(ctx._get_openrouter_api_key(session))
        chatgpt_key = bool(ctx._get_chatgpt_api_key(session))
        local_ready = ctx.LOCAL_RUNTIME.is_ready()
        return {
            "openai_key_configured": openai_key,
            "openrouter_key_configured": openrouter_key,
            "chatgpt_key_configured": chatgpt_key,
            "local_ready": local_ready,
            "local_status": ctx.LOCAL_RUNTIME.status(),
            "llm_key_configured": openai_key or openrouter_key or chatgpt_key or local_ready,
            "docker_available": ctx.docker_available(),
        }

    @router.get("/api/local/status")
    def api_local_status() -> dict:
        return ctx.LOCAL_RUNTIME.status()

    @router.get("/api/local/models")
    def api_local_models() -> dict:
        return {"items": ctx.LOCAL_RUNTIME.list_models()}

    @router.post("/api/local/setup")
    def api_local_setup(payload: LocalSetupRequest, session: Session = Depends(ctx.get_session)) -> dict:
        model_id = (payload.model_id or "").strip()
        custom_url = (payload.custom_url or "").strip()
        custom_name = (payload.custom_name or "").strip()
        try:
            status = ctx.LOCAL_RUNTIME.start(model_id=model_id, custom_url=custom_url, custom_name=custom_name)
        except Exception as exc:
            raise ctx.HTTPException(status_code=400, detail=str(exc)) from exc
        default_model = model_id or custom_name
        if not default_model and custom_url:
            default_model = custom_url.rsplit("/", 1)[-1].split("?", 1)[0].strip()
        if default_model:
            ctx._set_app_setting(session, "default_llm_provider", "local")
            ctx._set_app_setting(session, "default_llm_model", default_model)
        try:
            bot = ctx._get_or_create_system_bot(session)
            bot.llm_provider = "local"
            if default_model:
                bot.openai_model = default_model
            bot.updated_at = ctx.dt.datetime.now(ctx.dt.timezone.utc)
            session.add(bot)
            session.commit()
        except Exception:
            pass
        try:
            ctx._get_or_create_showcase_bot(session)
        except Exception:
            pass
        return status

    @router.post("/api/onboarding/reset")
    def api_onboarding_reset(session: Session = Depends(ctx.get_session)) -> dict:
        for k in ctx.list_keys(session):
            ctx.delete_key(session, k.id)
        for k in ctx.list_client_keys(session):
            ctx.delete_client_key(session, k.id)
        try:
            keep_names = {SYSTEM_BOT_NAME, SHOWCASE_BOT_NAME}
            bots = session.exec(select(Bot)).all()
            keep_ids = {b.id for b in bots if b.name in keep_names}

            session.exec(delete(ConversationMessage))
            session.exec(delete(ConversationReadState))
            session.exec(delete(HostAction))
            session.exec(delete(Conversation))
            session.exec(delete(IntegrationTool))
            if keep_ids:
                session.exec(delete(Bot).where(~Bot.id.in_(keep_ids)))
            else:
                session.exec(delete(Bot))
            session.exec(delete(GitToken))
            session.exec(delete(AppSetting))
            session.commit()
            ctx._get_or_create_system_bot(session)
            ctx._get_or_create_showcase_bot(session)
        except Exception:
            pass
        try:
            ctx.LOCAL_RUNTIME.stop()
        except Exception:
            pass
        return {"ok": True}

    app.include_router(router)
