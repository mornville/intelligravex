from __future__ import annotations

import logging

DEFAULT_DATA_AGENT_SYSTEM_PROMPT = (
    "You are given a task (what_to_do), API spec, authorization tokens, and conversation context. "
    "Call any API if needed, satisfy what_to_do, and respond back with a simple response."
)

DEFAULT_DATA_AGENT_IMAGE = "igx-data-agent:local"

logger = logging.getLogger("voicebot.data_agent")
