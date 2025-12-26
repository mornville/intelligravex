from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Literal, Optional, Sequence


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
        Streams assistant text deltas. If the model emits a function/tool call,
        yields a single ToolCall at the end (and typically no text deltas).
        """
        stream = self._client.responses.create(
            model=self._model,
            input=_to_responses_input(messages),
            tools=tools or None,
            stream=True,
        )

        tool_name: Optional[str] = None
        args_buf: List[str] = []

        def _get(obj: Any, key: str, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        for event in stream:
            et = _get(event, "type")
            if et == "response.output_text.delta":
                delta = _get(event, "delta")
                if isinstance(delta, str) and delta:
                    yield delta
                continue

            # Best-effort tool call capture (SDK types differ across versions).
            if isinstance(et, str) and ("function_call" in et or "tool_call" in et):
                name = _get(event, "name") or _get(_get(event, "item"), "name")
                if isinstance(name, str) and name:
                    tool_name = name

                arguments = _get(event, "arguments") or _get(_get(event, "item"), "arguments")
                if isinstance(arguments, str) and arguments:
                    args_buf = [arguments]
                    continue

                delta = _get(event, "delta")
                if isinstance(delta, str) and delta:
                    args_buf.append(delta)
                    continue

                item = _get(event, "item")
                if item is not None:
                    name = _get(item, "name")
                    if isinstance(name, str) and name:
                        tool_name = name
                    arguments = _get(item, "arguments")
                    if isinstance(arguments, str) and arguments:
                        args_buf = [arguments]
                        continue

        if tool_name and args_buf:
            yield ToolCall(name=tool_name, arguments_json="".join(args_buf))

    def complete_text(self, *, messages: Sequence[Message]) -> str:
        resp = self._client.responses.create(
            model=self._model,
            input=_to_responses_input(messages),
        )
        return (getattr(resp, "output_text", "") or "").strip()
