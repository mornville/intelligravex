from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Generator, List, Literal, Sequence


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

    def complete_text(self, *, messages: Sequence[Message]) -> str:
        resp = self._client.responses.create(
            model=self._model,
            input=_to_responses_input(messages),
        )
        return (getattr(resp, "output_text", "") or "").strip()
