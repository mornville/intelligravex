from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from typing import Any, Optional

import httpx
from sqlmodel import Session

from voicebot.config import Settings
from voicebot.crypto import CryptoError, get_crypto_box
from voicebot.llm.openai_compat_llm import OpenAICompatLLM
from voicebot.llm.openai_llm import OpenAILLM
from voicebot.llm.openrouter_llm import OpenRouterLLM
from voicebot.local_runtime import LOCAL_RUNTIME
from voicebot.models import Bot
from voicebot.store import decrypt_provider_key
from voicebot.utils.tokens import ModelPrice, estimate_cost_usd, estimate_messages_tokens, estimate_text_tokens
from voicebot.web.helpers.settings import read_key_from_env_file


def normalize_llm_provider(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p in ("openai", "openrouter", "local"):
        return p
    return "openai"


def provider_display_name(provider: str) -> str:
    if provider == "openrouter":
        return "OpenRouter"
    if provider == "local":
        return "Local"
    return "OpenAI"


def _settings_or_default(settings: Settings | None) -> Settings | None:
    if settings is not None:
        return settings
    try:
        return Settings()
    except Exception:
        return None


def _crypto_from_settings(settings: Settings | None):
    if settings is None:
        return None
    try:
        return get_crypto_box(settings.secret_key)
    except CryptoError:
        return None


def get_openai_api_key(session: Session, *, settings: Settings | None = None) -> str:
    key = os.environ.get("OPENAI_API_KEY") or ""
    if not key:
        s = _settings_or_default(settings)
        crypto = _crypto_from_settings(s)
        if crypto is not None:
            try:
                key = decrypt_provider_key(session, crypto=crypto, provider="openai") or ""
            except Exception:
                key = ""
    if not key:
        key = read_key_from_env_file("OPENAI_API_KEY")
    return (key or "").strip()


def get_openai_api_key_for_bot(session: Session, *, bot: Bot, settings: Settings | None = None) -> str:
    _ = bot
    return get_openai_api_key(session, settings=settings)


def get_openrouter_api_key(session: Session, *, settings: Settings | None = None) -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or ""
    if not key:
        s = _settings_or_default(settings)
        crypto = _crypto_from_settings(s)
        if crypto is not None:
            try:
                key = decrypt_provider_key(session, crypto=crypto, provider="openrouter") or ""
            except Exception:
                key = ""
    if not key:
        key = read_key_from_env_file("OPENROUTER_API_KEY")
    return (key or "").strip()


def get_openrouter_api_key_for_bot(session: Session, *, bot: Bot, settings: Settings | None = None) -> str:
    _ = bot
    return get_openrouter_api_key(session, settings=settings)


def llm_provider_for_bot(bot: Bot) -> str:
    raw = (getattr(bot, "llm_provider", "") or "").strip().lower()
    if raw in ("openai", "openrouter", "local"):
        return raw
    return "openai"


def get_llm_api_key_for_bot(
    session: Session,
    *,
    bot: Bot,
    settings: Settings | None = None,
) -> tuple[str, str]:
    provider = llm_provider_for_bot(bot)
    if provider == "openrouter":
        return provider, get_openrouter_api_key_for_bot(session, bot=bot, settings=settings)
    if provider == "local":
        return provider, ""
    return provider, get_openai_api_key_for_bot(session, bot=bot, settings=settings)


def build_llm_client(*, bot: Bot, api_key: str, model_override: Optional[str] = None):
    provider = llm_provider_for_bot(bot)
    model = (model_override or bot.openai_model or "").strip() or "o4-mini"
    if provider == "openrouter":
        base_url = (os.environ.get("OPENROUTER_BASE_URL") or "").strip() or "https://openrouter.ai/api/v1"
        ref = (os.environ.get("OPENROUTER_REFERER") or "").strip()
        title = (os.environ.get("OPENROUTER_TITLE") or "").strip()
        headers = {}
        if ref:
            headers["HTTP-Referer"] = ref
        if title:
            headers["X-Title"] = title
        return OpenRouterLLM(api_key=api_key, model=model, base_url=base_url, headers=headers)
    if provider == "local":
        base_url = (os.environ.get("IGX_LOCAL_LLM_BASE_URL") or "").strip()
        if not base_url:
            status = LOCAL_RUNTIME.status()
            port = int(status.get("server_port") or 0) or 0
            if port:
                base_url = f"http://127.0.0.1:{port}"
        if not base_url:
            raise RuntimeError("Local runtime not ready.")
        return OpenAICompatLLM(model=model, base_url=base_url, api_key="")
    return OpenAILLM(api_key=api_key, model=model)


def _select_local_model_id(bot: Bot) -> str:
    model_id = (getattr(bot, "openai_model", "") or "").strip()
    if model_id:
        return model_id
    try:
        models = LOCAL_RUNTIME.list_models()
    except Exception:
        models = []
    for m in models:
        if bool(m.get("recommended")):
            return str(m.get("id") or m.get("name") or "").strip()
    if models:
        return str(models[0].get("id") or models[0].get("name") or "").strip()
    return ""


def require_llm_client(
    session: Session,
    *,
    bot: Bot,
    settings: Settings | None = None,
):
    provider, api_key = get_llm_api_key_for_bot(session, bot=bot, settings=settings)
    if provider != "local" and not api_key:
        raise RuntimeError(f"No {provider_display_name(provider)} key configured for this bot.")
    if provider == "local":
        base_url_env = (os.environ.get("IGX_LOCAL_LLM_BASE_URL") or "").strip()
        if not base_url_env and not LOCAL_RUNTIME.is_ready():
            model_id = _select_local_model_id(bot)
            try:
                LOCAL_RUNTIME.start(model_id=model_id)
            except Exception as exc:
                raise RuntimeError(f"Local model not ready. Failed to start local setup: {exc}") from exc
            raise RuntimeError("Local model not ready. Setup started; check Local setup status in the dashboard.")
    llm = build_llm_client(bot=bot, api_key=api_key, model_override=bot.openai_model)
    return provider, api_key, llm


@lru_cache(maxsize=1)
def get_openai_pricing() -> dict[str, ModelPrice]:
    raw = os.environ.get("OPENAI_PRICING_JSON") or ""
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    out: dict[str, ModelPrice] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                continue
            try:
                input_per_1m = v.get("input_per_1m")
                output_per_1m = v.get("output_per_1m")
                if input_per_1m is None or output_per_1m is None:
                    continue
                out[k] = ModelPrice(input_per_1m=float(input_per_1m), output_per_1m=float(output_per_1m))
            except Exception:
                continue
    return out


def refresh_openrouter_models_cache(
    session: Session,
    *,
    cache: dict[str, Any],
    get_openrouter_api_key_fn,
) -> None:
    try:
        now = time.time()
        if (now - float(cache.get("ts") or 0.0)) <= 3600.0:
            return
        cache["ts"] = now
        cache["models"] = []
        cache["pricing"] = {}
        or_key = get_openrouter_api_key_fn(session)
        base_url = (os.environ.get("OPENROUTER_BASE_URL") or "").strip() or "https://openrouter.ai/api/v1"
        headers = {}
        if or_key:
            headers["Authorization"] = f"Bearer {or_key}"
        ref = (os.environ.get("OPENROUTER_REFERER") or "").strip()
        title = (os.environ.get("OPENROUTER_TITLE") or "").strip()
        if ref:
            headers["HTTP-Referer"] = ref
        if title:
            headers["X-Title"] = title
        resp = httpx.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=15.0)
        if resp.status_code >= 400:
            return
        data = resp.json()
        items = data.get("data") or []
        ids: list[str] = []
        pricing_map: dict[str, ModelPrice] = {}
        for m in items:
            if not isinstance(m, dict):
                continue
            mid = m.get("id")
            if not isinstance(mid, str) or not mid.strip():
                continue
            mid = mid.strip()
            ids.append(mid)
            price = m.get("pricing")
            if isinstance(price, dict):
                try:
                    prompt = float(price.get("prompt"))
                    completion = float(price.get("completion"))
                except Exception:
                    prompt = None
                    completion = None
                if prompt is not None and completion is not None:
                    pricing_map[mid] = ModelPrice(
                        input_per_1m=prompt * 1_000_000.0,
                        output_per_1m=completion * 1_000_000.0,
                    )
        cache["models"] = sorted(set(ids))
        cache["pricing"] = pricing_map
    except Exception:
        return


def get_openrouter_pricing(
    session: Session,
    *,
    cache: dict[str, Any],
    get_openrouter_api_key_fn,
) -> dict[str, ModelPrice]:
    refresh_openrouter_models_cache(session, cache=cache, get_openrouter_api_key_fn=get_openrouter_api_key_fn)
    return dict(cache.get("pricing") or {})


def get_model_price(
    session: Session,
    *,
    provider: str,
    model: str,
    cache: dict[str, Any],
    get_openrouter_api_key_fn,
) -> Optional[ModelPrice]:
    if provider == "local":
        return None
    if provider == "openrouter":
        return get_openrouter_pricing(session, cache=cache, get_openrouter_api_key_fn=get_openrouter_api_key_fn).get(model)
    return get_openai_pricing().get(model)


def estimate_llm_cost_for_turn(
    *,
    session: Session,
    bot: Bot,
    provider: str,
    history: list,
    assistant_text: str,
    get_model_price_fn,
) -> tuple[int, int, float]:
    prompt_tokens = estimate_messages_tokens(history, bot.openai_model)
    output_tokens = estimate_text_tokens(assistant_text, bot.openai_model)
    price = get_model_price_fn(session, provider=provider, model=bot.openai_model)
    cost = estimate_cost_usd(model_price=price, input_tokens=prompt_tokens, output_tokens=output_tokens)
    return prompt_tokens, output_tokens, cost


def make_start_message_instruction(bot: Bot) -> str:
    return (
        "Generate a short opening message to start a voice conversation. "
        "Keep it concise and end with a question."
    )
