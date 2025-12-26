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


def get_session(engine) -> Session:
    return Session(engine)
