from __future__ import annotations

from voicebot.tools.http_request import http_request_tool_def


def test_http_request_tool_schema_is_valid_for_responses_api() -> None:
    tool = http_request_tool_def()
    assert tool["type"] == "function"
    assert tool["name"] == "http_request"
    params = tool["parameters"]
    assert params["type"] == "object"
    assert isinstance(params.get("properties"), dict)
    assert "args" in params["properties"]
