from __future__ import annotations


def set_metadata_tool_def() -> dict:
    """
    OpenAI Responses API tool schema for set_metadata.

    Requirement: parameters must be an object schema with a `properties` map
    (Responses rejects object schemas that only use `additionalProperties`).
    """
    return {
        "type": "function",
        "name": "set_metadata",
        "description": (
            "Set or update conversation variables/metadata as key/value pairs. "
            "Include a 'next_reply' string to say to the user after updating variables."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "next_reply": {
                    "type": "string",
                    "description": "What the assistant should say next (no second LLM call).",
                },
                "checkpoint": {
                    "type": "string",
                    "description": "Optional checkpoint/state name for the conversation.",
                },
            },
            "required": ["next_reply"],
            # Allow arbitrary keys like ssn, dob, etc at the top-level.
            "additionalProperties": True,
        },
        "strict": False,
    }


def set_variable_tool_def() -> dict:
    """Alias for set_metadata (user-friendly name used in prompts)."""
    d = set_metadata_tool_def()
    d["name"] = "set_variable"
    d["description"] = (
        "Set or update conversation variables/metadata as key/value pairs. "
        "Include a 'next_reply' string to say to the user after updating variables."
    )
    return d
