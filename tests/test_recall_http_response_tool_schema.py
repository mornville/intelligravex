from __future__ import annotations


from voicebot.tools.recall_http_response import recall_http_response_tool_def


def test_recall_http_response_tool_schema_is_valid_for_responses_api() -> None:
    tool = recall_http_response_tool_def()
    assert tool["type"] == "function"
    assert tool["name"] == "recall_http_response"
    params = tool["parameters"]
    assert params["type"] == "object"
    assert isinstance(params.get("properties"), dict)
    for k in ("source_tool_name", "fields_required", "why_api_was_called"):
        assert k in params["properties"]
    assert "required" in params
    assert "source_tool_name" in params["required"]
    assert "fields_required" in params["required"]
    assert "why_api_was_called" in params["required"]

