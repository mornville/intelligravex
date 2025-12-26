from __future__ import annotations

import os
from typing import Optional

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


def get_session(engine) -> Session:
    return Session(engine)

