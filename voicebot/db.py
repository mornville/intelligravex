from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine, select

from voicebot.models import Bot, Conversation, ConversationMessage, HostAction


def get_db_url(db_url: Optional[str] = None) -> str:
    if db_url:
        return _normalize_db_url(db_url)
    return _normalize_db_url(os.environ.get("VOICEBOT_DB_URL") or "sqlite:///voicebot.db")


def _normalize_db_url(url: str) -> str:
    raw = str(url or "").strip()
    if raw.startswith("sqlite:///"):
        path = raw[len("sqlite:///") :]
        path = os.path.expanduser(path)
        if path:
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
        return f"sqlite:///{path}"
    return raw


def make_engine(db_url: Optional[str] = None):
    url = get_db_url(db_url)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite:") else {}
    return create_engine(url, echo=False, connect_args=connect_args)


def init_db(engine) -> None:
    logger = logging.getLogger("voicebot.db")
    start = time.monotonic()
    logger.info("init_db: start")
    try:
        url = str(getattr(engine, "url", ""))
        if url.startswith("sqlite:"):
            with engine.begin() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.execute(text("PRAGMA synchronous=NORMAL"))
    except Exception:
        # Best-effort; do not block app startup.
        pass
    SQLModel.metadata.create_all(engine)
    logger.info("init_db: create_all done (%.2fs)", time.monotonic() - start)
    mig_start = time.monotonic()
    _apply_light_migrations(engine)
    logger.info("init_db: migrations done (%.2fs)", time.monotonic() - mig_start)
    logger.info("init_db: complete (%.2fs)", time.monotonic() - start)


def _apply_light_migrations(engine) -> None:
    # Minimal, best-effort SQLite migrations for local dev (no Alembic).
    logger = logging.getLogger("voicebot.db")
    start = time.monotonic()
    url = str(getattr(engine, "url", ""))
    if not url.startswith("sqlite:"):
        logger.info("init_db: _apply_light_migrations skipped (non-sqlite) (%.2fs)", time.monotonic() - start)
        return
    with engine.begin() as conn:
        try:
            rows = conn.execute(text("PRAGMA table_info(conversation)")).fetchall()
        except Exception:
            logger.info("init_db: _apply_light_migrations skipped (no conversation table) (%.2fs)", time.monotonic() - start)
            return
        existing = {r[1] for r in rows}  # name at index=1

        added_last_message_cols = ("last_message_at" not in existing) or ("last_message_preview" not in existing)

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
        add_col("last_message_at", "TEXT")
        add_col("last_message_preview", "TEXT NOT NULL DEFAULT ''")
        add_col("metadata_json", "TEXT NOT NULL DEFAULT '{}'")
        add_col("external_id", "TEXT")
        add_col("client_key_id", "TEXT")
        add_col("is_group", "INTEGER NOT NULL DEFAULT 0")
        add_col("group_title", "TEXT NOT NULL DEFAULT ''")
        add_col("group_bots_json", "TEXT NOT NULL DEFAULT '[]'")

        if added_last_message_cols:
            try:
                conn.execute(
                    text(
                        """
                        UPDATE conversation
                        SET last_message_at = (
                            SELECT created_at
                            FROM conversationmessage
                            WHERE conversationmessage.conversation_id = conversation.id
                              AND role IN ('user','assistant')
                            ORDER BY created_at DESC, id DESC
                            LIMIT 1
                        ),
                        last_message_preview = (
                            SELECT substr(
                                replace(replace(trim(content), char(10), ' '), char(13), ' '),
                                1,
                                200
                            )
                            FROM conversationmessage
                            WHERE conversationmessage.conversation_id = conversation.id
                              AND role IN ('user','assistant')
                            ORDER BY created_at DESC, id DESC
                            LIMIT 1
                        )
                        WHERE last_message_at IS NULL
                        """
                    )
                )
            except Exception:
                logger.info("init_db: last_message backfill skipped (best-effort)")

        # Bot
        rows = conn.execute(text("PRAGMA table_info(bot)")).fetchall()
        existing = {r[1] for r in rows}

        def add_bot_col(name: str, ddl: str) -> None:
            if name in existing:
                return
            conn.execute(text(f"ALTER TABLE bot ADD COLUMN {name} {ddl}"))

        add_bot_col("start_message_mode", "TEXT NOT NULL DEFAULT 'llm'")
        add_bot_col("start_message_text", "TEXT NOT NULL DEFAULT ''")
        add_bot_col("llm_provider", "TEXT NOT NULL DEFAULT 'openai'")
        add_bot_col("openai_asr_model", "TEXT NOT NULL DEFAULT 'gpt-4o-mini-transcribe'")
        add_bot_col("openai_tts_model", "TEXT NOT NULL DEFAULT 'gpt-4o-mini-tts'")
        add_bot_col("openai_tts_voice", "TEXT NOT NULL DEFAULT 'alloy'")
        add_bot_col("openai_tts_speed", "REAL NOT NULL DEFAULT 1.0")
        add_bot_col("web_search_model", "TEXT NOT NULL DEFAULT 'gpt-4o-mini'")
        add_bot_col("codex_model", "TEXT NOT NULL DEFAULT 'gpt-5.1-codex-mini'")
        add_bot_col("summary_model", "TEXT NOT NULL DEFAULT 'gpt-5-nano'")
        add_bot_col("history_window_turns", "INTEGER NOT NULL DEFAULT 16")
        add_bot_col("enable_data_agent", "INTEGER NOT NULL DEFAULT 0")
        add_bot_col("data_agent_api_spec_text", "TEXT NOT NULL DEFAULT ''")
        add_bot_col("data_agent_auth_json", "TEXT NOT NULL DEFAULT '{}'")  # stored as plaintext JSON by request
        add_bot_col("data_agent_system_prompt", "TEXT NOT NULL DEFAULT ''")
        add_bot_col("data_agent_return_result_directly", "INTEGER NOT NULL DEFAULT 0")
        add_bot_col("data_agent_prewarm_on_start", "INTEGER NOT NULL DEFAULT 0")
        add_bot_col("data_agent_prewarm_prompt", "TEXT NOT NULL DEFAULT ''")
        add_bot_col("data_agent_model", "TEXT NOT NULL DEFAULT 'gpt-5.2'")
        add_bot_col("data_agent_reasoning_effort", "TEXT NOT NULL DEFAULT 'high'")
        add_bot_col("enable_host_actions", "INTEGER NOT NULL DEFAULT 0")
        add_bot_col("enable_host_shell", "INTEGER NOT NULL DEFAULT 0")
        add_bot_col("require_host_action_approval", "INTEGER NOT NULL DEFAULT 0")
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
                    "UPDATE bot SET openai_asr_model='gpt-4o-mini-transcribe' "
                    "WHERE openai_asr_model IS NULL OR openai_asr_model=''"
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
        add_msg_col("sender_bot_id", "TEXT")
        add_msg_col("sender_name", "TEXT")
        add_msg_col("citations_json", "TEXT NOT NULL DEFAULT '[]'")

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
        add_tool_col("postprocess_python", "TEXT NOT NULL DEFAULT ''")
        add_tool_col("return_result_directly", "INTEGER NOT NULL DEFAULT 0")
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
    logger.info("init_db: _apply_light_migrations complete (%.2fs)", time.monotonic() - start)


def get_session(engine) -> Session:
    return Session(engine)


def _seed_demo_group(engine) -> None:
    # Demo PM/SDE seed intentionally disabled.
    logging.getLogger("voicebot.db").info("init_db: demo seed disabled")
