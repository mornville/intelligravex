from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence

import httpx

from voicebot.llm.openai_llm import CitationEvent, Message, ToolCall


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _to_chat_messages(messages: Sequence[Message]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        if not m.content:
            continue
        out.append({"role": m.role, "content": m.content})
    return out


def _to_openrouter_tools(tools: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") != "function":
            # OpenRouter's OpenAI-compatible endpoint supports function tools.
            continue
        name = t.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        fn = {
            "name": name,
            "description": str(t.get("description") or "").strip(),
            "parameters": t.get("parameters") or {"type": "object", "properties": {}},
        }
        out.append({"type": "function", "function": fn})
    return out or None


class OpenRouterLLM:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        referer: str | None = None,
        title: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("No OpenRouter API key found (set OPENROUTER_API_KEY or configure a bot key).")

        base = (base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL).strip()
        self._base_url = base.rstrip("/")
        self._model = (model or "").strip()
        self._headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        ref = (referer or os.environ.get("OPENROUTER_REFERER") or "").strip()
        title_hdr = (title or os.environ.get("OPENROUTER_TITLE") or "").strip()
        if ref:
            self._headers["HTTP-Referer"] = ref
        if title_hdr:
            self._headers["X-Title"] = title_hdr
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
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": _to_chat_messages(messages),
        }
        converted_tools = _to_openrouter_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools
        if stream:
            payload["stream"] = True
        return payload

    def _post(self, *, payload: dict[str, Any], stream: bool) -> httpx.Response:
        url = f"{self._base_url}/chat/completions"
        if stream:
            return self._client.stream("POST", url, json=payload, headers=self._headers)
        return self._client.post(url, json=payload, headers=self._headers)

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
        tool_calls: dict[int, dict[str, Any]] = {}
        with self._post(payload=payload, stream=True) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] if isinstance(choices, list) else {}
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield content
                tc_list = delta.get("tool_calls")
                if isinstance(tc_list, list):
                    for tc in tc_list:
                        if not isinstance(tc, dict):
                            continue
                        idx = tc.get("index")
                        try:
                            idx_i = int(idx) if idx is not None else 0
                        except Exception:
                            idx_i = 0
                        entry = tool_calls.setdefault(
                            idx_i, {"name": None, "args": [], "first_event_ts": None}
                        )
                        if entry["first_event_ts"] is None:
                            entry["first_event_ts"] = time.time()
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        name = fn.get("name")
                        if isinstance(name, str) and name:
                            entry["name"] = name
                        args = fn.get("arguments")
                        if isinstance(args, str) and args:
                            entry["args"].append(args)

        if tool_calls:
            for idx in sorted(tool_calls.keys()):
                entry = tool_calls[idx]
                name = entry.get("name") or "set_metadata"
                args_json = "".join(entry.get("args") or []) or "{}"
                yield ToolCall(
                    name=name,
                    arguments_json=args_json,
                    first_event_ts=entry.get("first_event_ts"),
                )

    def complete_text(self, *, messages: Sequence[Message]) -> str:
        payload = self.build_request_payload(messages=messages, tools=None, stream=False)
        resp = self._post(payload=payload, stream=False)
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = msg.get("content") if isinstance(msg, dict) else None
        return str(content or "").strip()

    def complete_vision(self, *, prompt: str, image_url: str) -> str:
        raise RuntimeError("Vision requests are not implemented for OpenRouter in this client.")
