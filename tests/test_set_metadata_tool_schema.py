from __future__ import annotations


from voicebot.tools.set_metadata import set_metadata_tool_def


def test_set_metadata_tool_schema_is_valid_for_responses_api() -> None:
    tool = set_metadata_tool_def()
    assert tool["type"] == "function"
    assert tool["name"] == "set_metadata"
    params = tool["parameters"]
    assert params["type"] == "object"
    # Responses API rejects object schemas without a `properties` map.
    assert isinstance(params.get("properties"), dict)
    assert "next_reply" in params["properties"]
    assert "required" in params and "next_reply" in params["required"]

