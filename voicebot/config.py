from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet
from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_data_dir() -> str:
    return str(Path.home() / ".gravexstudio")


def _load_or_create_secret_key(data_dir: Path) -> Optional[str]:
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "secret.key"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        key = Fernet.generate_key().decode("utf-8")
        path.write_text(key, encoding="utf-8")
        return key
    except Exception:
        return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOICEBOT_", env_file=".env", extra="ignore")

    # OpenAI
    openai_model: str = Field(default="o4-mini", alias="OPENAI_MODEL")
    openai_asr_model: str = Field(default="gpt-4o-mini-transcribe", alias="OPENAI_ASR_MODEL")
    openai_api_key: Optional[str] = None  # optional override (otherwise uses env OPENAI_API_KEY)
    system_prompt: str = Field(
        default="You are a fast, helpful voice assistant. Keep answers concise unless asked."
    )
    language: str = Field(default="en")
    bot_uuid: Optional[str] = None

    # App data (default: ~/.gravexstudio)
    data_dir: str = Field(default_factory=_default_data_dir, alias="DATA_DIR")

    # Storage
    db_url: str = Field(default="", alias="DB_URL")
    secret_key: Optional[str] = Field(default=None, alias="SECRET_KEY")  # Fernet key for encrypting provider secrets
    # Absolute base URL used for download links returned by export helpers.
    # Override via VOICEBOT_DOWNLOAD_BASE_URL (supports full URL or host[:port]).
    download_base_url: str = "127.0.0.1:8000"
    basic_auth_user: Optional[str] = None
    basic_auth_pass: Optional[str] = None

    # Audio I/O
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000

    # VAD
    vad_aggressiveness: int = 2
    vad_silence_ms: int = 800
    vad_min_speech_ms: int = 250
    vad_max_speech_ms: int = 15000

    # Turn-taking (helps prevent TTS audio being re-transcribed)
    mic_mute_during_tts: bool = True
    tts_mic_release_ms: int = 500

    # OpenAI TTS
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"
    openai_tts_speed: float = 1.0

    @field_validator("data_dir", mode="before")
    @classmethod
    def _normalize_data_dir(cls, v):
        raw = str(v or "").strip()
        return str(Path(raw).expanduser()) if raw else _default_data_dir()

    @field_validator("db_url", mode="before")
    @classmethod
    def _default_db_url(cls, v, info):
        raw = str(v or "").strip()
        if raw:
            return raw
        data_dir = Path(str(info.data.get("data_dir") or _default_data_dir())).expanduser()
        return f"sqlite:///{data_dir / 'voicebot.db'}"

    @field_validator("secret_key", mode="before")
    @classmethod
    def _default_secret_key(cls, v, info):
        raw = str(v or "").strip()
        if raw:
            return raw
        data_dir = Path(str(info.data.get("data_dir") or _default_data_dir())).expanduser()
        return _load_or_create_secret_key(data_dir)

    @field_validator("input_device", "output_device", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return v

    def print_diagnostics(self) -> None:
        import os
        import sys

        from voicebot.audio.devices import list_audio_devices

        print("python:", sys.version.replace("\n", " "))
        print("openai_model:", self.openai_model)
        print("openai_asr_model:", self.openai_asr_model)
        print("input_sample_rate:", self.input_sample_rate)
        print("output_sample_rate:", self.output_sample_rate)
        print("input_device:", self.input_device or "(default)")
        print("output_device:", self.output_device or "(default)")
        print("data_dir:", self.data_dir)
        print("db_url:", self.db_url)
        print("VOICEBOT_SECRET_KEY set:", bool(self.secret_key))
        print("OPENAI_API_KEY set:", bool(os.environ.get("OPENAI_API_KEY")))

        print("\nAudio devices:")
        for line in list_audio_devices():
            print(" ", line)
