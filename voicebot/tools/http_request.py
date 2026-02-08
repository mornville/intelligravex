from __future__ import annotations

from typing import Any


def http_request_tool_def() -> dict[str, Any]:
    """
    Generic HTTP tool that lets the LLM specify URL/method/headers/body and
    request specific fields to extract from the response.
    """
    return {
        "type": "function",
        "name": "http_request",
        "description": (
            "Call an external HTTP API. Provide the URL/method/headers/body, and specify "
            "fields_required (or a response_mapper_json) to extract only the needed values. "
            "The tool returns extracted fields; raw responses are not returned unless you map them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "wait_reply": {
                    "type": "string",
                    "description": "Short filler message to say while the tool runs.",
                },
                "args": {
                    "type": "object",
                    "description": "Arguments used to call the HTTP API (required).",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Full URL for the request (may include query params).",
                        },
                        "method": {
                            "type": "string",
                            "description": "HTTP method (GET/POST/PUT/PATCH/DELETE/HEAD/OPTIONS).",
                        },
                        "headers": {
                            "anyOf": [{"type": "object"}, {"type": "string"}],
                            "description": "Request headers as JSON or a JSON string.",
                        },
                        "body": {
                            "anyOf": [
                                {"type": "object"},
                                {"type": "array", "items": {}},
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "boolean"},
                            ],
                            "description": "Request body as JSON or raw string.",
                        },
                        "query": {
                            "anyOf": [{"type": "object"}, {"type": "string"}],
                            "description": "Optional query params as JSON or JSON string.",
                        },
                        "fields_required": {
                            "anyOf": [{"type": "string"}, {"type": "array", "items": {}}],
                            "description": (
                                "List of response fields needed for the answer (comma or newline separated). "
                                "Used to auto-build a response mapper if none is provided."
                            ),
                        },
                        "why_api_was_called": {
                            "type": "string",
                            "description": "Short reason for this API call (user intent).",
                        },
                        "response_mapper_json": {
                            "anyOf": [{"type": "object"}, {"type": "string"}],
                            "description": (
                                "Optional response mapper (JSON) that maps output keys to templates like "
                                "{{response.data.id}}. Overrides fields_required when provided."
                            ),
                        },
                    },
                    "required": ["url", "fields_required"],
                    "additionalProperties": True,
                },
            },
            "required": ["args"],
            "additionalProperties": True,
        },
        "strict": False,
    }
