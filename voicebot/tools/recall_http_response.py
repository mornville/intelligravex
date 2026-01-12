from __future__ import annotations

from typing import Any


def recall_http_response_tool_def() -> dict[str, Any]:
    """
    System tool: re-run the Codex one-shot post-processing over a previously saved
    HTTP integration response for this conversation.
    """
    return {
        "type": "function",
        "name": "recall_http_response",
        "description": (
            "Recall a previously saved HTTP integration response for this conversation and extract new fields. "
            "Use this when the user asks follow-up questions about data returned earlier (e.g., 'for those providers...')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_tool_name": {
                    "type": "string",
                    "description": "Integration tool name whose previous response you want to recall (e.g. 'search_individuals').",
                },
                "source_req_id": {
                    "type": "string",
                    "description": "Optional: exact previous tool req_id to recall (if known). If omitted, uses the latest run.",
                },
                "fields_required": {
                    "type": "string",
                    "description": "Fields required from the saved response to answer the user now.",
                },
                "why_api_was_called": {
                    "type": "string",
                    "description": "Why you need this recall/extraction (user intent).",
                },
                "wait_reply": {
                    "type": "string",
                    "description": "Short filler message to say while recalling (e.g. 'Got it—checking that now.').",
                },
            },
            "required": ["source_tool_name", "fields_required", "why_api_was_called", "wait_reply"],
            "additionalProperties": True,
        },
        "strict": False,
    }
