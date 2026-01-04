from __future__ import annotations


from voicebot.tools.export_http_response import export_http_response_tool_def


def test_export_http_response_tool_schema_is_valid_for_responses_api() -> None:
    tool = export_http_response_tool_def()
    assert tool["type"] == "function"
    assert tool["name"] == "export_http_response"
    params = tool["parameters"]
    assert params["type"] == "object"
    assert isinstance(params.get("properties"), dict)
    for k in ("source_tool_name", "export_request", "output_format"):
        assert k in params["properties"]
    assert "required" in params
    assert "source_tool_name" in params["required"]
    assert "export_request" in params["required"]

