from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.engine import Engine

from voicebot.config import Settings


@dataclass
class AppState:
    settings: Settings
    engine: Engine
    logger: logging.Logger
    data_agent_kickoff_locks: dict[UUID, asyncio.Lock]
    download_base_url: str
    basic_user: str
    basic_pass: str
    basic_auth_enabled: bool
    ui_options: dict[str, Any]
    ui_dir: Path
    ui_index: Path
    openai_models_cache: dict[str, Any]
    openrouter_models_cache: dict[str, Any]
