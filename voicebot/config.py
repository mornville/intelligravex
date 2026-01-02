from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOICEBOT_", env_file=".env", extra="ignore")

    # OpenAI
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_api_key: Optional[str] = None  # optional override (otherwise uses env OPENAI_API_KEY)
    system_prompt: str = Field(
        default="You are a fast, helpful voice assistant. Keep answers concise unless asked."
    )
    language: str = Field(default="en")
    bot_uuid: Optional[str] = None

    # Storage
    db_url: str = "sqlite:///voicebot.db"
    secret_key: Optional[str] = None  # Fernet key for encrypting provider secrets

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

    # Whisper
    whisper_model: str = "small"
    whisper_device: str = "auto"

    # Turn-taking (helps prevent TTS audio being re-transcribed)
    mic_mute_during_tts: bool = True
    tts_mic_release_ms: int = 500

    # TTS performance
    tts_use_gpu: bool = True
    tts_split_sentences: bool = False
    tts_chunk_min_chars: int = 20
    tts_chunk_max_chars: int = 120
    tts_vendor: str = "xtts_local"  # xtts_local | openai_tts

    # XTTS v2
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    speaker_wav: Optional[str] = None
    speaker_id: Optional[str] = None
    tts_language: str = "en"

    # OpenAI TTS (only used when tts_vendor=openai_tts)
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"
    openai_tts_speed: float = 1.0

    # Web scraping (for system tool: web_search)
    scrapingbee_api_key: Optional[str] = Field(default=None, alias="SCRAPINGBEE_API_KEY")

    @field_validator("input_device", "output_device", "speaker_wav", "speaker_id", mode="before")
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
        print("input_sample_rate:", self.input_sample_rate)
        print("output_sample_rate:", self.output_sample_rate)
        print("input_device:", self.input_device or "(default)")
        print("output_device:", self.output_device or "(default)")
        print("whisper_model:", self.whisper_model)
        print("whisper_device:", self.whisper_device)
        print("tts_use_gpu:", self.tts_use_gpu)
        print("tts_vendor:", self.tts_vendor)
        print("db_url:", self.db_url)
        print("VOICEBOT_SECRET_KEY set:", bool(self.secret_key))
        print("OPENAI_API_KEY set:", bool(os.environ.get("OPENAI_API_KEY")))

        try:
            import whisper  # type: ignore

            print("whisper installed:", True, "(module:", whisper.__name__ + ")")
        except Exception as exc:
            print("whisper installed:", False, f"({exc})")

        try:
            from TTS.api import TTS  # type: ignore

            _ = TTS
            print("coqui TTS installed:", True)
        except Exception as exc:
            print("coqui TTS installed:", False, f"({exc})")

        print("\nAudio devices:")
        for line in list_audio_devices():
            print(" ", line)
