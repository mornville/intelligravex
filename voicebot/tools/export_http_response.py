from __future__ import annotations

from typing import Any


def export_http_response_tool_def() -> dict[str, Any]:
    """
    System tool: generate a downloadable file from a previously saved HTTP integration response.

    The file is created by running a Codex-generated Python script over the saved JSON payload.
    """
    return {
        "type": "function",
        "name": "export_http_response",
        "description": (
            "Create a downloadable export file (CSV/JSON) from a previously saved HTTP integration response "
            "in this conversation. Use this when a client asks to download results from an earlier tool call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_tool_name": {
                    "type": "string",
                    "description": "Integration tool name whose previous response you want to export (e.g. 'search_individuals').",
                },
                "source_req_id": {
                    "type": "string",
                    "description": "Optional: exact previous tool req_id to export (if known). If omitted, uses the latest run.",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["csv", "json"],
                    "description": "Desired export file format.",
                    "default": "csv",
                },
                "export_request": {
                    "type": "string",
                    "description": (
                        "What the exported file should contain: columns/fields, ordering, grouping, and any "
                        "aggregation needed. The upstream API response is already filtered."
                    ),
                },
                "file_name_hint": {
                    "type": "string",
                    "description": "Optional preferred file name (no path). Backend may adjust/override.",
                },
                "wait_reply": {
                    "type": "string",
                    "description": "Short filler message to say while exporting (e.g. 'Got it—preparing a download.').",
                },
            },
            "required": ["source_tool_name", "export_request"],
            "additionalProperties": True,
        },
        "strict": False,
    }

