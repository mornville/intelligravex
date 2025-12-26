from __future__ import annotations

import datetime as dt
from typing import Optional
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel


class ApiKey(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    provider: str = Field(index=True)  # e.g. "openai"
    name: str = Field(index=True)
    hint: str = Field(default="")  # masked preview for UI, e.g. sk-****abcd
    secret_ciphertext: bytes
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class Bot(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    name: str = Field(index=True)

    # LLM
    openai_model: str = "gpt-4o"
    system_prompt: str = "You are a fast, helpful voice assistant. Keep answers concise unless asked."
    openai_key_id: Optional[UUID] = Field(default=None, foreign_key="apikey.id")

    # ASR
    asr_vendor: str = "whisper_local"
    language: str = "en"
    whisper_model: str = "small"
    whisper_device: str = "auto"

    # TTS
    tts_vendor: str = "xtts_local"
    tts_language: str = "en"
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    speaker_wav: Optional[str] = None
    speaker_id: Optional[str] = None
    tts_split_sentences: bool = False
    tts_chunk_min_chars: int = 20
    tts_chunk_max_chars: int = 120

    # Conversation start message (assistant speaks first)
    start_message_mode: str = "llm"  # "static" | "llm"
    start_message_text: str = ""

    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class Conversation(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    bot_id: UUID = Field(foreign_key="bot.id", index=True)
    test_flag: bool = Field(default=True, index=True)
    # LLM usage + cost (estimated)
    llm_input_tokens_est: int = Field(default=0)
    llm_output_tokens_est: int = Field(default=0)
    cost_usd_est: float = Field(default=0.0)

    # Latency (last turn, ms)
    last_asr_ms: Optional[int] = Field(default=None)
    last_llm_ttfb_ms: Optional[int] = Field(default=None)
    last_llm_total_ms: Optional[int] = Field(default=None)
    last_tts_first_audio_ms: Optional[int] = Field(default=None)
    last_total_ms: Optional[int] = Field(default=None)

    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), index=True)


class ConversationMessage(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    conversation_id: UUID = Field(foreign_key="conversation.id", index=True)
    role: str = Field(index=True)  # system/user/assistant
    content: str
    # Optional per-message metrics (typically set on assistant messages)
    input_tokens_est: Optional[int] = Field(default=None)
    output_tokens_est: Optional[int] = Field(default=None)
    cost_usd_est: Optional[float] = Field(default=None)
    asr_ms: Optional[int] = Field(default=None)
    llm_ttfb_ms: Optional[int] = Field(default=None)
    llm_total_ms: Optional[int] = Field(default=None)
    tts_first_audio_ms: Optional[int] = Field(default=None)
    total_ms: Optional[int] = Field(default=None)
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), index=True)
