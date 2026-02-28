from __future__ import annotations

import json
from typing import Any

from sqlmodel import Session

from voicebot.models import Bot, IntegrationTool
from voicebot.store import list_integration_tools
from voicebot.tools.data_agent import give_command_to_data_agent_tool_def
from voicebot.tools.http_request import http_request_tool_def
from voicebot.tools.set_metadata import set_metadata_tool_def, set_variable_tool_def
from voicebot.tools.web_search import web_search_tool_def
from voicebot.utils.template import safe_json_loads
from voicebot.web.helpers.host_actions import (
    capture_screenshot_tool_def,
    summarize_screenshot_tool_def,
)
from voicebot.web.helpers.integration_utils import (
    parse_parameters_schema_json,
    parse_required_args_json,
)
from voicebot.web.helpers.llm import llm_provider_for_bot


def set_metadata_tool() -> dict:
    return set_metadata_tool_def()


def set_variable_tool() -> dict:
    return set_variable_tool_def()


def web_search_tool() -> dict:
    return web_search_tool_def()


def http_request_tool() -> dict:
    return http_request_tool_def()


def host_action_tool() -> dict:
    return {
        "type": "function",
        "name": "request_host_action",
        "description": "Use this when a task requires running local shell commands or AppleScript.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["run_shell", "run_applescript", "run_powershell"],
                    "description": "Type of host action.",
                },
                "command": {
                    "type": "string",
                    "description": "Command string for run_shell, or fallback script text for run_powershell.",
                },
                "script": {
                    "type": "string",
                    "description": "Script body for run_applescript (macOS) or run_powershell (Windows).",
                },
                "follow_up": {
                    "type": "boolean",
                    "description": "If true, force a follow-up assistant reply after the action runs.",
                },
                "next_reply": {
                    "type": "string",
                    "description": "Keep this empty",
                },
            },
            "required": ["action"],
        },
        "strict": False,
    }


def disabled_tool_names(bot: Bot) -> set[str]:
    raw = (getattr(bot, "disabled_tools_json", "") or "[]").strip() or "[]"
    try:
        obj = json.loads(raw)
    except Exception:
        obj = []
    out: set[str] = set()
    if isinstance(obj, list):
        for x in obj:
            s = str(x or "").strip()
            if s:
                out.add(s)
    out.discard("set_metadata")
    out.discard("set_variable")
    return out


def system_tools_defs(*, bot: Bot) -> list[dict[str, Any]]:
    tools = [
        set_metadata_tool(),
        http_request_tool(),
    ]
    if llm_provider_for_bot(bot) in ("openai", "chatgpt"):
        tools.insert(1, web_search_tool())
    if bool(getattr(bot, "enable_data_agent", False)):
        tools.append(give_command_to_data_agent_tool_def())
        tools.append(summarize_screenshot_tool_def())
    if bool(getattr(bot, "enable_host_actions", False)):
        tools.append(host_action_tool())
        if bool(getattr(bot, "enable_host_shell", False)):
            tools.append(capture_screenshot_tool_def())
    return tools


def system_tools_public_list(*, bot: Bot, disabled: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in system_tools_defs(bot=bot):
        name = str(d.get("name") or d.get("type") or "")
        if not name:
            continue
        desc = str(d.get("description") or "")
        if not desc and str(d.get("type") or "") == "web_search":
            desc = "Search the web for recent information with citations."
        can_disable = name not in ("set_metadata", "set_variable")
        out.append(
            {
                "name": name,
                "description": desc,
                "enabled": name not in disabled,
                "can_disable": can_disable,
            }
        )
    return [x for x in out if x.get("name")]


def integration_tool_def(t: IntegrationTool) -> dict[str, Any]:
    required_args = parse_required_args_json(getattr(t, "args_required_json", "[]"))
    explicit_schema = parse_parameters_schema_json(getattr(t, "parameters_schema_json", ""))
    if explicit_schema:
        args_schema = explicit_schema
    else:
        args_schema = {
            "type": "object",
            "properties": {k: {"type": "string"} for k in required_args},
            "required": required_args,
            "additionalProperties": True,
        }

    def _append_required_args_to_schema(schema: dict[str, Any], required: list[str]) -> dict[str, Any]:
        if not required:
            return schema
        props = schema.get("properties")
        if not isinstance(props, dict) and schema.get("type") in (None, "object"):
            props = {}
        if isinstance(props, dict):
            merged = dict(schema)
            merged["type"] = "object"
            merged_props = dict(props)
            for k in required:
                merged_props.setdefault(k, {"type": "string"})
            merged["properties"] = merged_props
            req = merged.get("required")
            if not isinstance(req, list):
                req = []
            for k in required:
                if k not in req:
                    req.append(k)
            merged["required"] = req
            if "additionalProperties" not in merged:
                merged["additionalProperties"] = True
            return merged
        return {
            "allOf": [
                schema,
                {
                    "type": "object",
                    "properties": {k: {"type": "string"} for k in required},
                    "required": required,
                    "additionalProperties": True,
                },
            ]
        }

    args_schema = _append_required_args_to_schema(args_schema, required_args)

    use_codex_response = bool(getattr(t, "use_codex_response", False))
    if use_codex_response:
        intent_schema = {
            "type": "object",
            "properties": {
                "fields_required": {
                    "type": "string",
                    "description": "Fields required from the HTTP response to craft the final user-facing response.",
                },
                "why_api_was_called": {
                    "type": "string",
                    "description": "User intent or business reason for calling this API.",
                },
                "what_to_search_for": {"type": "string"},
                "why_to_search_for": {"type": "string"},
            },
            "required": ["fields_required", "why_api_was_called"],
            "additionalProperties": True,
        }

        args_schema = _append_required_args_to_schema(args_schema, list(intent_schema["required"]))
        props = args_schema.get("properties")
        if isinstance(props, dict):
            merged = dict(args_schema)
            merged_props = dict(props)
            for k, v in intent_schema["properties"].items():
                merged_props.setdefault(k, v)
            merged["properties"] = merged_props
            args_schema = merged
        else:
            args_schema = {"allOf": [args_schema, intent_schema]}

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "wait_reply": {
                "type": "string",
                "description": "Short filler message to say while the tool runs (not used for set_metadata).",
            },
            "args": {
                "description": "Arguments used to call the integration (required).",
                **args_schema,
            },
        },
        "additionalProperties": True,
    }
    if use_codex_response:
        schema["required"] = ["args", "wait_reply"]
    else:
        schema["properties"]["next_reply"] = {
            "type": "string",
            "description": (
                "What the assistant should say next (no second LLM call). "
                "Variables like {{.firstName}} are allowed."
            ),
        }
        schema["required"] = ["args", "next_reply", "wait_reply"]

    try:
        pag = safe_json_loads(getattr(t, "pagination_json", "") or "") or None
    except Exception:
        pag = None
    if isinstance(pag, dict) and str(pag.get("items_path") or "").strip():
        max_items_cap = pag.get("max_items_cap")
        try:
            max_items_cap_i = int(max_items_cap) if max_items_cap is not None else 5000
        except Exception:
            max_items_cap_i = 5000
        if max_items_cap_i < 1:
            max_items_cap_i = 5000
        if max_items_cap_i > 50000:
            max_items_cap_i = 50000
        props = schema.get("properties")
        if isinstance(props, dict):
            args_props = props.get("args")
            if isinstance(args_props, dict):
                args_props.setdefault("properties", {})
                if isinstance(args_props.get("properties"), dict):
                    args_props["properties"].setdefault(
                        "max_items",
                        {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": max_items_cap_i,
                            "description": (
                                f"Optional: stop pagination after collecting this many items (max {max_items_cap_i}). "
                                "Backend will fetch multiple pages if needed."
                            ),
                        },
                    )
    return {
        "type": "function",
        "name": t.name,
        "description": (t.description or "").strip()
        + " This tool calls an external HTTP API and maps selected response fields into conversation metadata. "
        + (
            "Return your spoken/text response in next_reply (you can use metadata variables like {{.firstName}})."
            if not use_codex_response
            else "If Codex mode is enabled, the backend runs a Codex agent to generate a result string, and the main chat model will rephrase it."
        ),
        "parameters": schema,
        "strict": False,
    }


def build_tools_for_bot(session: Session, bot_id) -> list[dict[str, Any]]:
    from voicebot.store import get_bot

    bot = get_bot(session, bot_id)
    disabled = disabled_tool_names(bot)
    tools: list[dict[str, Any]] = []
    for d in system_tools_defs(bot=bot):
        tool_id = str(d.get("name") or d.get("type") or "")
        if tool_id and tool_id in disabled:
            continue
        tools.append(d)
    for t in list_integration_tools(session, bot_id=bot_id):
        if not bool(getattr(t, "enabled", True)):
            continue
        if str(getattr(t, "name", "") or "").strip() in disabled:
            continue
        try:
            tools.append(integration_tool_def(t))
        except Exception:
            continue
    return tools
