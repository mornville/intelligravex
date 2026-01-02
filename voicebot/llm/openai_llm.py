from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, List, Literal, Optional, Sequence, Tuple

import time


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


def _to_responses_input(messages: Sequence[Message]) -> List[Dict]:
    # Responses API expects typed content blocks:
    # - user/system: `input_text`
    # - assistant: `output_text`
    items: List[Dict] = []
    for m in messages:
        if not m.content:
            continue
        content_type = "output_text" if m.role == "assistant" else "input_text"
        items.append({"role": m.role, "content": [{"type": content_type, "text": m.content}]})
    return items


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments_json: str
    first_event_ts: Optional[float] = None


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_stream_events(
    events: Iterable[Any],
    *,
    tool_name_hint: Optional[str] = None,
    now_fn=time.time,
) -> Generator[str | ToolCall, None, None]:
    """
    Parses Responses streaming events.

    Event types used:
    - response.output_text.delta
    - response.function_call_arguments.delta/done
    """
    fc_name: Optional[str] = None
    fc_args: List[str] = []
    fc_first_ts: Optional[float] = None

    for event in events:
        et = _get(event, "type")
        if et == "response.output_text.delta":
            delta = _get(event, "delta")
            if isinstance(delta, str) and delta:
                yield delta
            continue

        if et == "response.function_call_arguments.delta":
            delta = _get(event, "delta")
            if isinstance(delta, str) and delta:
                if fc_first_ts is None:
                    fc_first_ts = now_fn()
                fc_args.append(delta)
            continue

        if et == "response.function_call_arguments.done":
            name = _get(event, "name")
            if isinstance(name, str) and name:
                fc_name = name
            arguments = _get(event, "arguments")
            if isinstance(arguments, str) and arguments:
                args_json = arguments
            else:
                args_json = "".join(fc_args)
            if fc_first_ts is None:
                fc_first_ts = now_fn()
            tool_name = fc_name or tool_name_hint or "set_metadata"
            if args_json:
                yield ToolCall(name=tool_name, arguments_json=args_json, first_event_ts=fc_first_ts)
                # Reset for potential additional tool calls in the same stream.
                fc_name = None
                fc_args = []
                fc_first_ts = None
            continue

        # Some SDK versions may include tool info on output_item events.
        if et in ("response.output_item.added", "response.output_item.done"):
            item = _get(event, "item")
            item_type = _get(item, "type")
            if item_type in ("function_call", "tool_call"):
                name = _get(item, "name")
                if isinstance(name, str) and name:
                    fc_name = name
                arguments = _get(item, "arguments")
                if isinstance(arguments, str) and arguments:
                    if fc_first_ts is None:
                        fc_first_ts = now_fn()
                    fc_args = [arguments]
            continue


class OpenAILLM:
    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("openai sdk not installed; pip install openai") from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("No OpenAI API key found (set OPENAI_API_KEY or configure a bot key).")

        self._client = OpenAI(api_key=key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def build_request_payload(
        self,
        *,
        messages: Sequence[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Returns the exact payload shape we send to the Responses API via the SDK.
        Useful for debugging / tracing.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "input": _to_responses_input(messages),
        }
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True
        return payload

    def stream_text(self, *, messages: Sequence[Message]) -> Generator[str, None, None]:
        stream = self._client.responses.create(
            model=self._model,
            input=_to_responses_input(messages),
            stream=True,
        )
        for event in stream:
            if isinstance(event, dict):
                event_type = event.get("type")
                delta = event.get("delta")
            else:
                event_type = getattr(event, "type", None)
                delta = getattr(event, "delta", None)
            if event_type == "response.output_text.delta":
                if isinstance(delta, str) and delta:
                    yield delta

    def stream_text_or_tool(
        self,
        *,
        messages: Sequence[Message],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Generator[str | ToolCall, None, None]:
        """
        Streams assistant text deltas. If the model emits function/tool calls,
        yields ToolCall events (may be multiple in one stream).
        """
        stream = self._client.responses.create(
            model=self._model,
            input=_to_responses_input(messages),
            tools=tools or None,
            stream=True,
        )
        tool_name_hint = None
        if tools and len(tools) == 1:
            n = tools[0].get("name")
            if isinstance(n, str) and n:
                tool_name_hint = n

        yield from _parse_stream_events(stream, tool_name_hint=tool_name_hint)

    def complete_text(self, *, messages: Sequence[Message]) -> str:
        resp = self._client.responses.create(
            model=self._model,
            input=_to_responses_input(messages),
        )
        return (getattr(resp, "output_text", "") or "").strip()
