from __future__ import annotations

import os

from sqlmodel import Session

from voicebot.config import Settings
from voicebot.crypto import CryptoError, get_crypto_box
from voicebot.store import decrypt_provider_key
from voicebot.web.helpers.settings import read_key_from_env_file


def get_openai_api_key_global(session: Session) -> str:
    key = os.environ.get("OPENAI_API_KEY") or ""
    if not key:
        try:
            settings = Settings()
        except Exception:
            settings = None
        if settings is not None:
            try:
                crypto = get_crypto_box(settings.secret_key)
            except CryptoError:
                crypto = None
            if crypto is not None:
                try:
                    key = decrypt_provider_key(session, crypto=crypto, provider="openai") or ""
                except Exception:
                    key = ""
    if not key:
        key = read_key_from_env_file("OPENAI_API_KEY")
    return (key or "").strip()
