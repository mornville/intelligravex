from __future__ import annotations

import json
import re
from typing import Any

from voicebot.models import IntegrationTool

_TEMPLATE_VAR_RE = re.compile(r"{{\s*([^}]+?)\s*}}")


def get_json_path(obj: Any, path: str) -> Any:
    if not path:
        return obj
    if obj is None:
        return None
    parts = [p for p in str(path).split(".") if p]
    cur = obj
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
        elif isinstance(cur, list):
            try:
                idx = int(p)
            except Exception:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            return None
    return cur


def set_json_path(obj: Any, path: str, value: Any) -> bool:
    if not path:
        return False
    parts = [p for p in str(path).split(".") if p]
    if not parts:
        return False
    cur = obj
    for i, p in enumerate(parts):
        last = i == len(parts) - 1
        if isinstance(cur, dict):
            if last:
                cur[p] = value
                return True
            if p not in cur or not isinstance(cur[p], (dict, list)):
                cur[p] = {}
            cur = cur[p]
            continue
        if isinstance(cur, list):
            try:
                idx = int(p)
            except Exception:
                return False
            if idx < 0:
                return False
            if idx >= len(cur):
                cur.extend([{}] * (idx - len(cur) + 1))
            if last:
                cur[idx] = value
                return True
            if not isinstance(cur[idx], (dict, list)):
                cur[idx] = {}
            cur = cur[idx]
            continue
        return False
    return False


def apply_schema_defaults(schema: Any, value: Any) -> Any:
    if schema is None:
        return value
    if not isinstance(schema, dict):
        return value
    if value is None:
        if "default" in schema:
            return schema.get("default")
        if schema.get("type") == "object":
            return {}
        if schema.get("type") == "array":
            return []
        return value
    if schema.get("type") == "object" and isinstance(value, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            out = dict(value)
            for k, v in props.items():
                if k not in out:
                    out[k] = apply_schema_defaults(v, None)
                else:
                    out[k] = apply_schema_defaults(v, out[k])
            return out
    if schema.get("type") == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if item_schema is None:
            return value
        return [apply_schema_defaults(item_schema, v) for v in value]
    return value


def http_error_response(*, url: str, status_code: int | None, body: str | None, message: str | None) -> dict:
    return {
        "ok": False,
        "error": {
            "message": (message or "HTTP request failed"),
            "status_code": status_code,
            "url": url,
            "body": body,
        },
    }


def extract_required_tool_args(tool: IntegrationTool) -> list[str]:
    raw = str(getattr(tool, "args_required_json", "") or "").strip() or "[]"
    try:
        args = json.loads(raw)
    except Exception:
        args = []
    required: list[str] = []
    if isinstance(args, list):
        for a in args:
            s = str(a or "").strip()
            if s:
                required.append(s)

    # Also infer required args from templated content.
    def scan(text: str) -> set[str]:
        found: set[str] = set()
        for m in _TEMPLATE_VAR_RE.finditer(text):
            expr = m.group(1) or ""
            expr = expr.strip()
            if not expr:
                continue
            # Only accept "args.*" references for required args.
            if expr.startswith("args."):
                parts = expr.split(".")
                if len(parts) >= 2:
                    key = parts[1].strip()
                    if key:
                        found.add(key)
        return found

    hints = set()
    for candidate in (
        getattr(tool, "request_body_template", "") or "",
        getattr(tool, "headers_template_json", "") or "",
        getattr(tool, "response_mapper_json", "") or "",
        getattr(tool, "static_reply_template", "") or "",
    ):
        hints |= scan(str(candidate))

    for k in sorted(hints):
        if k not in required:
            required.append(k)

    return required


def parse_required_args_json(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out: list[str] = []
    for v in obj:
        s = str(v or "").strip()
        if s:
            out.append(s)
    return out


def parse_parameters_schema_json(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def missing_required_args(required: list[str], args: dict) -> list[str]:
    missing = []
    for key in required:
        if key not in args or args.get(key) in (None, ""):
            missing.append(key)
    return missing


def normalize_content_type_header_value(v: str) -> str:
    vv = str(v or "").strip()
    if not vv:
        return ""
    lower = vv.lower()
    if lower in ("json", "application/json"):
        return "application/json"
    if lower in ("form", "application/x-www-form-urlencoded"):
        return "application/x-www-form-urlencoded"
    return vv


def normalize_headers_for_json(headers: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (headers or {}).items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, str):
            out[k] = v
            continue
        if k.lower() == "content-type":
            out[k] = normalize_content_type_header_value(v)
        else:
            out[k] = v
    return out


def integration_error_user_message(*, tool_name: str, err: dict) -> str:
    msg = ""
    if isinstance(err, dict):
        msg = str(err.get("message") or "").strip()
    if not msg:
        msg = f"{tool_name} failed"
    return msg


def should_followup_llm_for_tool(*, tool: IntegrationTool | None, static_rendered: str) -> bool:
    if tool is None:
        return True
    if bool(getattr(tool, "return_result_directly", False)):
        return True
    return not bool(static_rendered)


def safe_json_list(raw: str) -> list:
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def parse_follow_up_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"true", "1", "yes", "y"}
