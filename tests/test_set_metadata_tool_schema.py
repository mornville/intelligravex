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


def test_tool_stream_parser_emits_toolcall_on_done_event() -> None:
    from voicebot.llm.openai_llm import _parse_stream_events

    events = [
        {"type": "response.function_call_arguments.delta", "delta": '{"dob":"07072000",'},
        {"type": "response.function_call_arguments.delta", "delta": '"next_reply":"What do you want for your birthday?"}'},
        {
            "type": "response.function_call_arguments.done",
            "name": "set_metadata",
            "arguments": '{"dob":"07072000","next_reply":"What do you want for your birthday?"}',
        },
    ]
    out = list(_parse_stream_events(events, tool_name_hint="set_metadata", now_fn=lambda: 1.0))
    assert len(out) == 1
    tc = out[0]
    assert getattr(tc, "name") == "set_metadata"
    assert "next_reply" in getattr(tc, "arguments_json")
