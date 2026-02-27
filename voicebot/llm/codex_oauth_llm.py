from __future__ import annotations

import json
import os
from typing import Any, Generator, Iterable, Optional, Sequence

import httpx

from voicebot.llm.openai_llm import CitationEvent, Message, ToolCall, _parse_stream_events, _to_responses_input


DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex/responses"


def _normalize_codex_base_url(raw: str | None) -> str:
    base = (raw or "").strip()
    if not base:
        return DEFAULT_CODEX_BASE_URL
    base = base.rstrip("/")
    if base.endswith("/codex/responses"):
        return base
    if base.endswith("/codex"):
        return f"{base}/responses"
    if base.endswith("/backend-api"):
        return f"{base}/codex/responses"
    return base


class CodexOAuthLLM:
    def __init__(
        self,
        *,
        model: str,
        access_token: str,
        base_url: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        token = (access_token or "").strip()
        if not token:
            raise RuntimeError("No ChatGPT OAuth token found.")
        self._base_url = _normalize_codex_base_url(base_url or os.environ.get("CHATGPT_OAUTH_BASE_URL"))
        self._model = (model or "").strip()
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "OpenAI-Beta": "responses=codex",
        }
        self._client = httpx.Client(timeout=timeout_s)

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
        instructions_parts: list[str] = []
        non_system: list[Message] = []
        for m in messages:
            if m.role == "system":
                if m.content:
                    instructions_parts.append(m.content)
            else:
                non_system.append(m)
        instructions = "\n\n".join(instructions_parts).strip()
        payload: dict[str, Any] = {
            "model": self._model,
            "instructions": instructions or "You are a helpful assistant.",
            "input": _to_responses_input(non_system),
            "store": False,
        }
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True
        return payload

    def _post(self, *, payload: dict[str, Any], stream: bool) -> httpx.Response:
        if stream:
            return self._client.stream("POST", self._base_url, json=payload, headers=self._headers)
        return self._client.post(self._base_url, json=payload, headers=self._headers)

    def _iter_sse_events(self, *, payload: dict[str, Any]) -> Iterable[Any]:
        with self._post(payload=payload, stream=True) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise RuntimeError(f"Codex backend error {resp.status_code}: {resp.text}")
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data or data == "[DONE]":
                    if data == "[DONE]":
                        break
                    continue
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                yield obj

    def stream_text(self, *, messages: Sequence[Message]) -> Generator[str, None, None]:
        for ev in self.stream_text_or_tool(messages=messages, tools=None):
            if isinstance(ev, (ToolCall, CitationEvent)):
                continue
            yield str(ev)

    def stream_text_or_tool(
        self,
        *,
        messages: Sequence[Message],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Generator[str | ToolCall | CitationEvent, None, None]:
        payload = self.build_request_payload(messages=messages, tools=tools, stream=True)
        for ev in _parse_stream_events(self._iter_sse_events(payload=payload)):
            yield ev

    def complete_text(self, *, messages: Sequence[Message]) -> str:
        parts: list[str] = []
        for ev in self.stream_text_or_tool(messages=messages, tools=None):
            if isinstance(ev, (ToolCall, CitationEvent)):
                continue
            text = str(ev)
            if text:
                parts.append(text)
        return "".join(parts).strip()
