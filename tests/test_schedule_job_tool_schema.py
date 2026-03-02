from __future__ import annotations

from voicebot.tools.schedule_job import schedule_job_tool_def


def test_schedule_job_tool_schema_is_valid_for_responses_api() -> None:
    tool = schedule_job_tool_def()
    assert tool["type"] == "function"
    assert tool["name"] == "schedule_job"
    params = tool["parameters"]
    assert params["type"] == "object"
    assert isinstance(params.get("properties"), dict)
    assert "action" in params["properties"]
    assert "required" in params and "action" in params["required"]
    actions = params["properties"]["action"]["enum"]
    assert "list" in actions
    assert "update" in actions
    assert "delete" in actions
    assert "disable" in actions
    assert "enable" in actions
    assert "disable_all" in actions
    assert "enable_all" in actions
    assert "delete_all" in actions
    assert "all" in params["properties"]
    assert "all_conversations" in params["properties"]
    assert "next_reply" not in params["properties"]
