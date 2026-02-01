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

from voicebot.models import Bot, Conversation, ConversationMessage


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
    SQLModel.metadata.create_all(engine)
    logger.info("init_db: create_all done (%.2fs)", time.monotonic() - start)
    mig_start = time.monotonic()
    _apply_light_migrations(engine)
    logger.info("init_db: migrations done (%.2fs)", time.monotonic() - mig_start)
    _seed_demo_group(engine)
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
        add_col("is_group", "INTEGER NOT NULL DEFAULT 0")
        add_col("group_title", "TEXT NOT NULL DEFAULT ''")
        add_col("group_bots_json", "TEXT NOT NULL DEFAULT '[]'")

        # Bot
        rows = conn.execute(text("PRAGMA table_info(bot)")).fetchall()
        existing = {r[1] for r in rows}

        def add_bot_col(name: str, ddl: str) -> None:
            if name in existing:
                return
            conn.execute(text(f"ALTER TABLE bot ADD COLUMN {name} {ddl}"))

        add_bot_col("start_message_mode", "TEXT NOT NULL DEFAULT 'llm'")
        add_bot_col("start_message_text", "TEXT NOT NULL DEFAULT ''")
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
    logger = logging.getLogger("voicebot.db")
    demo_title = "Demo: PM → SDE1 → SDE2"
    with Session(engine) as session:
        existing = session.exec(
            select(Conversation).where(Conversation.is_group == True, Conversation.group_title == demo_title)  # noqa: E712
        ).first()
        if existing:
            logger.info("init_db: demo seed exists; checking prompts")
            # Update demo bot prompts if needed.
            def ensure_prompt(name: str, system_prompt: str, enable_data_agent: bool) -> None:
                bot = session.exec(select(Bot).where(Bot.name == name)).first()
                if not bot:
                    return
                existing_prompt = bot.system_prompt or ""
                needs_update = (
                    "dispatch_assistants" in existing_prompt
                    or "facilitator" in existing_prompt.lower()
                    or "mention @sde2" in existing_prompt.lower()
                    or "mention @pm" in existing_prompt.lower()
                    or "<no_reply>" not in existing_prompt
                )
                if needs_update:
                    bot.system_prompt = system_prompt
                    bot.updated_at = dt.datetime.now(dt.timezone.utc)
                    session.add(bot)
                    session.commit()
                    session.refresh(bot)
            pm_prompt = (
                "You are the PM in a multi-assistant group chat. Clarify goals, define success criteria, and break work "
                "into actionable steps. To ask another assistant for work, mention them with @slug in your message "
                "(example: @sde1 please gather sources). If you have nothing to add, respond with <no_reply>."
            )
            sde1_prompt = (
                "You are SDE1. Implement tasks using the Data Agent and available tools. State your plan, execute, and "
                "report results clearly. If requirements are unclear, ask the user or another assistant using @slug. "
                "If you have nothing to add, respond with <no_reply>."
            )
            sde2_prompt = (
                "You are SDE2. Review other assistant output against requirements. Identify gaps or confirm completion. "
                "Ask questions using @slug when needed. If you have nothing to add, respond with <no_reply>."
            )
            ensure_prompt("PM", pm_prompt, enable_data_agent=False)
            ensure_prompt("SDE1", sde1_prompt, enable_data_agent=True)
            ensure_prompt("SDE2", sde2_prompt, enable_data_agent=False)
            return

        def get_or_create_bot(name: str, system_prompt: str, enable_data_agent: bool) -> Bot:
            bot = session.exec(select(Bot).where(Bot.name == name)).first()
            if bot:
                existing = bot.system_prompt or ""
                needs_update = (
                    "dispatch_assistants" in existing
                    or "facilitator" in existing.lower()
                    or "mention @sde2" in existing.lower()
                    or "mention @pm" in existing.lower()
                    or "<no_reply>" not in existing
                )
                if needs_update:
                    bot.system_prompt = system_prompt
                    bot.updated_at = dt.datetime.now(dt.timezone.utc)
                    session.add(bot)
                    session.commit()
                    session.refresh(bot)
                return bot
            bot = Bot(
                name=name,
                system_prompt=system_prompt,
                enable_data_agent=enable_data_agent,
                start_message_mode="static",
                start_message_text=f"Hi! I'm {name}.",
                created_at=dt.datetime.now(dt.timezone.utc),
                updated_at=dt.datetime.now(dt.timezone.utc),
            )
            session.add(bot)
            session.commit()
            session.refresh(bot)
            return bot

        pm_prompt = (
            "You are the PM in a multi-assistant group chat. Clarify goals, define success criteria, and break work "
            "into actionable steps. To ask another assistant for work, mention them with @slug in your message "
            "(example: @sde1 please gather sources). If you have nothing to add, respond with <no_reply>."
        )
        sde1_prompt = (
            "You are SDE1. Implement tasks using the Data Agent and available tools. State your plan, execute, and "
            "report results clearly. If requirements are unclear, ask the user or another assistant using @slug. "
            "If you have nothing to add, respond with <no_reply>."
        )
        sde2_prompt = (
            "You are SDE2. Review other assistant output against requirements. Identify gaps or confirm completion. "
            "Ask questions using @slug when needed. If you have nothing to add, respond with <no_reply>."
        )

        pm_bot = get_or_create_bot("PM", pm_prompt, enable_data_agent=False)
        sde1_bot = get_or_create_bot("SDE1", sde1_prompt, enable_data_agent=True)
        sde2_bot = get_or_create_bot("SDE2", sde2_prompt, enable_data_agent=False)

        group_bots = [
            {"id": str(pm_bot.id), "name": pm_bot.name, "slug": "pm"},
            {"id": str(sde1_bot.id), "name": sde1_bot.name, "slug": "sde1"},
            {"id": str(sde2_bot.id), "name": sde2_bot.name, "slug": "sde2"},
        ]

        now = dt.datetime.now(dt.timezone.utc)
        conv = Conversation(
            bot_id=pm_bot.id,
            test_flag=True,
            is_group=True,
            group_title=demo_title,
            group_bots_json=json.dumps(group_bots, ensure_ascii=False),
            metadata_json=json.dumps({"demo_seed": "pm_sde"}, ensure_ascii=False),
            created_at=now,
            updated_at=now,
        )
        session.add(conv)
        session.commit()
        session.refresh(conv)

        individual_map: dict[str, str] = {}
        for b in group_bots:
            bid = str(b["id"])
            now = dt.datetime.now(dt.timezone.utc)
            child = Conversation(
                bot_id=UUID(bid),
                test_flag=True,
                is_group=False,
                metadata_json=json.dumps(
                    {
                        "group_parent_id": str(conv.id),
                        "group_bot_id": bid,
                        "group_bot_name": str(b.get("name") or ""),
                    },
                    ensure_ascii=False,
                ),
                created_at=now,
                updated_at=now,
            )
            session.add(child)
            session.commit()
            session.refresh(child)
            individual_map[bid] = str(child.id)

        timeline = [
            ("user", None, "User", "I want a weekly competitor digest for AI voice bots. Pull updates from 5 sources, summarize, and save a markdown report."),
            (
                "assistant",
                pm_bot.id,
                "PM",
                "Got it. Use case: weekly competitor digest. Success: a markdown report saved locally with links + summaries.\n"
                "@sde1: implement a workflow that pulls 5 sources, summarizes them, and writes report.md. Ask me if you need a source list.",
            ),
            (
                "assistant",
                sde1_bot.id,
                "SDE1",
                "On it. I’ll use the Data Agent to generate a script and a report template, then share results for review.",
            ),
            (
                "tool",
                sde1_bot.id,
                "SDE1",
                json.dumps(
                    {
                        "tool": "give_command_to_data_agent",
                        "arguments": {
                            "cmd": "python - <<'PY'\nprint('Created weekly_digest.py and report.md template')\nPY"
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
            (
                "tool",
                sde1_bot.id,
                "SDE1",
                json.dumps(
                    {
                        "tool": "give_command_to_data_agent",
                        "result": {
                            "ok": True,
                            "result_text": "weekly_digest.py and report.md written in workspace",
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
            (
                "assistant",
                sde1_bot.id,
                "SDE1",
                "Implementation complete: weekly_digest.py + report.md template saved to workspace with placeholders for 5 sources.\n"
                "@sde2: review against the PM’s requirements.",
            ),
            (
                "assistant",
                sde2_bot.id,
                "SDE2",
                "Reviewed: matches requirements (5 sources, summaries, markdown report). No changes needed. @pm good to ship.",
            ),
            (
                "assistant",
                pm_bot.id,
                "PM",
                "All done. You can run weekly_digest.py to generate report.md each week. Let me know if you want a cron schedule.",
            ),
        ]

        created_at = now + dt.timedelta(seconds=2)
        for i, (role, sender_bot_id, sender_name, content) in enumerate(timeline):
            session.add(
                ConversationMessage(
                    conversation_id=conv.id,
                    role=role,
                    content=content,
                    sender_bot_id=sender_bot_id,
                    sender_name=sender_name,
                    created_at=created_at + dt.timedelta(seconds=i * 2),
                )
            )
            if sender_bot_id and role in ("assistant", "tool"):
                if role == "tool":
                    try:
                        obj = json.loads(content or "")
                    except Exception:
                        obj = None
                    if not isinstance(obj, dict) or "arguments" not in obj:
                        continue
                target = individual_map.get(str(sender_bot_id))
                if target:
                    session.add(
                        ConversationMessage(
                            conversation_id=UUID(target),
                            role=role,
                            content=content,
                            sender_bot_id=sender_bot_id,
                            sender_name=sender_name,
                            created_at=created_at + dt.timedelta(seconds=i * 2),
                        )
                    )

        conv.updated_at = created_at + dt.timedelta(seconds=len(timeline) * 2)
        conv.metadata_json = json.dumps(
            {
                "demo_seed": "pm_sde",
                "group_individual_conversations": individual_map,
            },
            ensure_ascii=False,
        )
        session.add(conv)
        session.commit()
        logger.info("init_db: demo seed created (PM/SDE1/SDE2 group)")
