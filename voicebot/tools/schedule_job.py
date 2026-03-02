from __future__ import annotations


def schedule_job_tool_def() -> dict:
    """
    Universal scheduler tool.

    Time requirements:
    - Always send UTC values.
    - `time_utc` must be 24h `HH:MM`.
    - `run_at_utc` must be ISO-8601 with `Z` (e.g. `2026-03-02T14:30:00Z`).
    """
    return {
        "type": "function",
        "name": "schedule_job",
        "description": (
            "Create, update, list, pause/resume, or delete recurring jobs for this assistant conversation. "
            "Always pass schedule times in UTC. If user provided local time, convert it to UTC before calling. "
            "For update/delete/disable/enable actions, first list jobs to find the correct job_id when needed. "
            "Use all=true (or *_all actions) for bulk operations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create",
                        "update",
                        "delete",
                        "pause",
                        "resume",
                        "disable",
                        "enable",
                        "delete_all",
                        "disable_all",
                        "enable_all",
                        "list",
                    ],
                    "description": "CRUD action for scheduled jobs.",
                },
                "all": {
                    "type": "boolean",
                    "description": "If true, apply pause/resume/delete to all jobs in scope.",
                },
                "all_conversations": {
                    "type": "boolean",
                    "description": "If true, list/apply jobs across all bot conversations. Default is current conversation.",
                },
                "job_id": {
                    "type": "string",
                    "description": "Existing job UUID for update/delete/pause/resume.",
                },
                "assistant_id": {
                    "type": "string",
                    "description": "Assistant UUID. If omitted, backend uses current bot.",
                },
                "conversation_uuid": {
                    "type": "string",
                    "description": "Conversation UUID to run the job in. If omitted, backend uses current conversation.",
                },
                "input_message": {
                    "type": "string",
                    "description": "Original user message that requested scheduling.",
                },
                "what_to_do": {
                    "type": "string",
                    "description": "Instruction to execute on each run (as if user typed it).",
                },
                "cadence": {
                    "type": "string",
                    "enum": ["once", "daily", "weekly"],
                    "description": "Recurrence cadence.",
                },
                "time_utc": {
                    "type": "string",
                    "description": "UTC 24-hour time in HH:MM. Required for daily/weekly.",
                },
                "weekday_utc": {
                    "type": "string",
                    "enum": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                    "description": "Required for weekly cadence.",
                },
                "run_at_utc": {
                    "type": "string",
                    "description": "One-time UTC run timestamp in ISO format with Z, e.g. 2026-03-02T14:30:00Z.",
                },
            },
            "required": ["action"],
            "additionalProperties": True,
        },
        "strict": False,
    }
