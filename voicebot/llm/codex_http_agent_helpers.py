from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional


ProgressFn = Callable[[str], None]
NextStep = Literal["STOP", "CONTINUE"]
_STATUS_PREFIX = "__IGX_STATUS__="


@dataclass(frozen=True)
class CodexAgentResult:
    ok: bool
    output_dir: str
    debug_json_path: str
    input_json_path: str
    schema_json_path: str
    filtered_json_path: str
    validated_json_path: str
    result_text_path: str
    result_text: str
    stop_reason: str
    continue_reason: str
    next_step: NextStep
    error: Optional[str] = None


def _to_responses_input(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role") or "user"
        text = (m.get("content") or "").strip()
        if not text:
            continue
        items.append({"role": role, "content": [{"type": "input_text", "text": text}]})
    return items


def _coerce_json_schema(obj: Any) -> Optional[dict[str, Any]]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str) and obj.strip():
        try:
            parsed = json.loads(obj)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _schema_matches_payload_top(schema: dict[str, Any], payload: Any) -> bool:
    """
    Best-effort, shallow validation:
    - If schema declares top-level type object/array, ensure payload matches.
    - Otherwise return True.
    """
    t = schema.get("type")
    types: list[str] = []
    if isinstance(t, str):
        types = [t]
    elif isinstance(t, list):
        types = [x for x in t if isinstance(x, str)]
    if not types:
        return True
    if "object" in types:
        return isinstance(payload, dict)
    if "array" in types:
        return isinstance(payload, list)
    return True


def _parse_status_line(stdout: str) -> Optional[dict[str, Any]]:
    if not stdout:
        return None
    lines = stdout.splitlines()
    for line in reversed(lines):
        s = (line or "").strip()
        if not s.startswith(_STATUS_PREFIX):
            continue
        raw = s[len(_STATUS_PREFIX) :].strip()
        try:
            obj = json.loads(raw)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def _json_tree_schema(value: Any, *, max_depth: int = 8, max_list_items: int = 25) -> Any:
    """
    Produces a compact, tree-structured "schema-like" representation:
    - objects: {key: subtree, ...} plus "__type__":"object"
    - arrays: {"__type__":"array","items":subtree}
    - primitives: {"__type__":"string|number|boolean|null"}
    """
    if max_depth <= 0:
        return {"__type__": "unknown"}
    if value is None:
        return {"__type__": "null"}
    if isinstance(value, bool):
        return {"__type__": "boolean"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"__type__": "number"}
    if isinstance(value, str):
        return {"__type__": "string"}
    if isinstance(value, dict):
        out: dict[str, Any] = {"__type__": "object"}
        for k in sorted([str(x) for x in value.keys()])[:800]:
            try:
                v = value.get(k)
            except Exception:
                v = None
            out[str(k)] = _json_tree_schema(v, max_depth=max_depth - 1, max_list_items=max_list_items)
        return out
    if isinstance(value, list):
        items: list[Any] = []
        for item in value[:max_list_items]:
            items.append(_json_tree_schema(item, max_depth=max_depth - 1, max_list_items=max_list_items))
        merged: Any = None
        for item in items:
            merged = item if merged is None else _merge_tree_schema(merged, item)
        return {"__type__": "array", "items": merged or {"__type__": "unknown"}}
    return {"__type__": "unknown"}


def _merge_tree_schema(a: Any, b: Any) -> Any:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return a
    ta = a.get("__type__")
    tb = b.get("__type__")
    if ta != tb:
        return a
    if ta == "object":
        out: dict[str, Any] = {"__type__": "object"}
        keys = set(a.keys()) | set(b.keys())
        for k in sorted([str(x) for x in keys if x != "__type__"])[:800]:
            if k in a and k not in b:
                out[k] = a.get(k)
            elif k in b and k not in a:
                out[k] = b.get(k)
            else:
                out[k] = _merge_tree_schema(a.get(k), b.get(k))
        return out
    if ta == "array":
        return {
            "__type__": "array",
            "items": _merge_tree_schema(a.get("items"), b.get("items")),
        }
    return a


def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _run_python_script(
    *,
    script_path: str,
    cwd: str,
    timeout_s: float = 20.0,
    argv: Optional[list[str]] = None,
    env_extra: Optional[dict[str, str]] = None,
) -> tuple[int, str, str]:
    env = dict(os.environ)
    for k in (
        "OPENAI_API_KEY",
        "OPENAI_KEY",
        "OPENAI_ORG",
        "OPENAI_ORGANIZATION",
        "OPENAI_PROJECT",
        "OPENAI_BASE_URL",
    ):
        env.pop(k, None)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if env_extra:
        for k, v in env_extra.items():
            if isinstance(k, str) and k.strip():
                env[str(k)] = str(v)
    proc = subprocess.run(
        [sys.executable, script_path] + ([str(x) for x in (argv or [])]),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return int(proc.returncode), proc.stdout or "", proc.stderr or ""


def _call_codex_json(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    schema_name: str,
    progress_fn: Optional[ProgressFn] = None,
) -> dict[str, Any]:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"openai sdk not installed: {exc}") from exc

    client = OpenAI(api_key=api_key)
    # Progress messaging should be handled by the caller (stage-specific). Avoid emitting a generic
    # "Thinking…" here which can overwrite more helpful UI updates.
    resp = client.responses.create(
        model=model,
        input=_to_responses_input(messages),
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "STOP_REASON": {"type": "string"},
                        "CONTINUE_REASON": {"type": "string"},
                        "next_step": {"type": "string", "enum": ["STOP", "CONTINUE"]},
                        "script": {"type": "string"},
                    },
                    "required": ["STOP_REASON", "CONTINUE_REASON", "next_step", "script"],
                },
            }
        },
    )
    raw = (getattr(resp, "output_text", "") or "").strip()
    try:
        obj = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Codex did not return valid JSON: {exc}; raw={raw[:500]}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError("Codex JSON must be an object.")
    return obj
