from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine


def get_db_url(db_url: Optional[str] = None) -> str:
    if db_url:
        return db_url
    return os.environ.get("VOICEBOT_DB_URL") or "sqlite:///voicebot.db"


def make_engine(db_url: Optional[str] = None):
    url = get_db_url(db_url)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite:") else {}
    return create_engine(url, echo=False, connect_args=connect_args)


def init_db(engine) -> None:
    SQLModel.metadata.create_all(engine)
    _apply_light_migrations(engine)


def _apply_light_migrations(engine) -> None:
    # Minimal, best-effort SQLite migrations for local dev (no Alembic).
    url = str(getattr(engine, "url", ""))
    if not url.startswith("sqlite:"):
        return
    with engine.begin() as conn:
        try:
            rows = conn.execute(text("PRAGMA table_info(conversation)")).fetchall()
        except Exception:
            return
        existing = {r[1] for r in rows}  # name at index=1

        def add_col(name: str, ddl: str) -> None:
            if name in existing:
                return
            conn.execute(text(f"ALTER TABLE conversation ADD COLUMN {name} {ddl}"))

        add_col("llm_input_tokens_est", "INTEGER NOT NULL DEFAULT 0")
        add_col("llm_output_tokens_est", "INTEGER NOT NULL DEFAULT 0")
        add_col("cost_usd_est", "REAL NOT NULL DEFAULT 0.0")
        add_col("last_asr_ms", "INTEGER")
        add_col("last_llm_ttfb_ms", "INTEGER")
        add_col("last_llm_total_ms", "INTEGER")
        add_col("last_tts_first_audio_ms", "INTEGER")
        add_col("last_total_ms", "INTEGER")
        add_col("metadata_json", "TEXT NOT NULL DEFAULT '{}'")
        add_col("external_id", "TEXT")
        add_col("client_key_id", "TEXT")

        # Bot
        rows = conn.execute(text("PRAGMA table_info(bot)")).fetchall()
        existing = {r[1] for r in rows}

        def add_bot_col(name: str, ddl: str) -> None:
            if name in existing:
                return
            conn.execute(text(f"ALTER TABLE bot ADD COLUMN {name} {ddl}"))

        add_bot_col("start_message_mode", "TEXT NOT NULL DEFAULT 'llm'")
        add_bot_col("start_message_text", "TEXT NOT NULL DEFAULT ''")
        add_bot_col("tts_vendor", "TEXT NOT NULL DEFAULT 'xtts_local'")
        add_bot_col("openai_tts_model", "TEXT NOT NULL DEFAULT 'gpt-4o-mini-tts'")
        add_bot_col("openai_tts_voice", "TEXT NOT NULL DEFAULT 'alloy'")
        add_bot_col("openai_tts_speed", "REAL NOT NULL DEFAULT 1.0")
        add_bot_col("web_search_model", "TEXT NOT NULL DEFAULT 'gpt-4o-mini'")
        add_bot_col("codex_model", "TEXT NOT NULL DEFAULT 'gpt-5.1-codex-mini'")
        add_bot_col("disabled_tools_json", "TEXT NOT NULL DEFAULT '[]'")
        try:
            conn.execute(
                text(
                    "UPDATE bot SET codex_model='gpt-5.1-codex-mini' "
                    "WHERE codex_model IS NULL OR codex_model='' OR codex_model='gpt-5-codex-mini'"
                )
            )
        except Exception:
            pass
        try:
            conn.execute(
                text(
                    "UPDATE bot SET openai_model='gpt-5.1-codex-mini' "
                    "WHERE openai_model='gpt-5-codex-mini'"
                )
            )
        except Exception:
            pass

        # ConversationMessage
        rows = conn.execute(text("PRAGMA table_info(conversationmessage)")).fetchall()
        existing = {r[1] for r in rows}

        def add_msg_col(name: str, ddl: str) -> None:
            if name in existing:
                return
            conn.execute(text(f"ALTER TABLE conversationmessage ADD COLUMN {name} {ddl}"))

        add_msg_col("input_tokens_est", "INTEGER")
        add_msg_col("output_tokens_est", "INTEGER")
        add_msg_col("cost_usd_est", "REAL")
        add_msg_col("asr_ms", "INTEGER")
        add_msg_col("llm_ttfb_ms", "INTEGER")
        add_msg_col("llm_total_ms", "INTEGER")
        add_msg_col("tts_first_audio_ms", "INTEGER")
        add_msg_col("total_ms", "INTEGER")

        # IntegrationTool
        try:
            rows = conn.execute(text("PRAGMA table_info(integrationtool)")).fetchall()
        except Exception:
            rows = []
        existing = {r[1] for r in rows} if rows else set()

        def add_tool_col(name: str, ddl: str) -> None:
            if not existing or name in existing:
                return
            conn.execute(text(f"ALTER TABLE integrationtool ADD COLUMN {name} {ddl}"))

        add_tool_col("static_reply_template", "TEXT NOT NULL DEFAULT ''")
        add_tool_col("headers_template_json", "TEXT NOT NULL DEFAULT '{}'")
        add_tool_col("args_required_json", "TEXT NOT NULL DEFAULT '[]'")
        add_tool_col(
            "parameters_schema_json",
            "TEXT NOT NULL DEFAULT '{\"type\":\"object\",\"properties\":{},\"additionalProperties\":true}'",
        )
        add_tool_col("response_schema_json", "TEXT NOT NULL DEFAULT ''")
        add_tool_col("codex_prompt", "TEXT NOT NULL DEFAULT ''")
        add_tool_col("use_codex_response", "INTEGER NOT NULL DEFAULT 0")
        add_tool_col("enabled", "INTEGER NOT NULL DEFAULT 1")
        add_tool_col("pagination_json", "TEXT NOT NULL DEFAULT ''")

        # ClientKey
        try:
            rows = conn.execute(text("PRAGMA table_info(clientkey)")).fetchall()
        except Exception:
            rows = []
        if not rows:
            return
        existing = {r[1] for r in rows}

        def add_client_key_col(name: str, ddl: str) -> None:
            if name in existing:
                return
            conn.execute(text(f"ALTER TABLE clientkey ADD COLUMN {name} {ddl}"))

        add_client_key_col("allowed_origins", "TEXT NOT NULL DEFAULT ''")
        add_client_key_col("allowed_bot_ids_json", "TEXT NOT NULL DEFAULT '[]'")


def get_session(engine) -> Session:
    return Session(engine)
