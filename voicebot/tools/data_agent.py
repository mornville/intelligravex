from __future__ import annotations

from typing import Any


def give_command_to_data_agent_tool_def() -> dict[str, Any]:
    """
    OpenAI Responses API tool schema for give_command_to_data_agent.

    This tool delegates a subtask to a Codex CLI "Isolated Workspace" running in an isolated runtime
    (currently Docker). The backend persists a Codex session per conversation so the agent can resume.
    """
    return {
        "type": "function",
        "name": "give_command_to_data_agent",
        "description": (
            "Delegate a data/API task to the Isolated Workspace (Codex CLI) running in an isolated runtime for this conversation. "
            "Use this when you need to call external APIs or perform structured data extraction outside the main chat model."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "what_to_do": {
                    "type": "string",
                    "description": "Clear instruction for the Isolated Workspace. Be specific about inputs/outputs.",
                },
                "wait_reply": {
                    "type": "string",
                    "description": "Short filler message to say while the Isolated Workspace runs.",
                },
            },
            "required": ["what_to_do", "wait_reply"],
            "additionalProperties": True,
        },
        "strict": False,
    }
