from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import parse_qsl

import httpx

from voicebot.models import IntegrationTool


def apply_response_mapper(
    ctx,
    *,
    mapper_json: str,
    response_json: Any,
    meta: dict,
    tool_args: dict,
) -> dict:
    mapper = ctx.safe_json_loads(mapper_json or "{}") or {}
    if not isinstance(mapper, dict):
        return {}
    out: dict = {}
    ctx_obj = {"response": response_json, "meta": meta, "args": tool_args, "params": tool_args}
    for k, tmpl in mapper.items():
        if not isinstance(k, str):
            continue
        if isinstance(tmpl, (dict, list)):
            out[k] = tmpl
            continue
        if tmpl is None:
            out[k] = None
            continue
        out[k] = ctx.eval_template_value(str(tmpl), ctx=ctx_obj)
    return out


def render_with_meta(ctx, text: str, meta: dict) -> str:
    return ctx.render_template(text, ctx={"meta": meta})


def render_static_reply(
    ctx,
    *,
    template_text: str,
    meta: dict,
    response_json: Any,
    tool_args: dict,
) -> str:
    return ctx.render_jinja_template(
        template_text,
        ctx={"meta": meta, "response": response_json, "args": tool_args, "params": tool_args},
    )


def coerce_json_object(value: Any) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
        except Exception:
            return {}
        return obj if isinstance(obj, dict) else {}
    return {}


def render_templates_in_obj(ctx, value: Any, *, ctx_obj: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: render_templates_in_obj(ctx, v, ctx_obj=ctx_obj) for k, v in value.items() if k is not None}
    if isinstance(value, list):
        return [render_templates_in_obj(ctx, v, ctx_obj=ctx_obj) for v in value]
    if isinstance(value, str):
        return ctx.render_template(value, ctx=ctx_obj)
    return value


def parse_query_params(ctx, value: Any, *, ctx_obj: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return render_templates_in_obj(ctx, value, ctx_obj=ctx_obj)
    if isinstance(value, str):
        rendered = ctx.render_template(value, ctx=ctx_obj).strip()
        if not rendered:
            return {}
        if rendered.startswith("{") and rendered.endswith("}"):
            try:
                obj = json.loads(rendered)
            except Exception:
                obj = None
            if isinstance(obj, dict):
                return render_templates_in_obj(ctx, obj, ctx_obj=ctx_obj)
        pairs = parse_qsl(rendered, keep_blank_values=True)
        out: dict[str, Any] = {}
        for k, v in pairs:
            if not k:
                continue
            out[k] = v
        return out
    return {}


def parse_fields_required(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, list):
        for item in value:
            s = str(item or "").strip()
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        for part in re.split(r"[\\n,]+", value):
            s = part.strip()
            if s:
                out.append(s)
    return out


def build_response_mapper_from_fields(fields_required: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in fields_required:
        s = str(item or "").strip()
        if not s:
            continue
        key = s
        expr = s
        for sep in ("=>", "=", ":"):
            if sep in s:
                left, right = s.split(sep, 1)
                if left.strip():
                    key = left.strip()
                    expr = right.strip() or expr
                break
        if "{{" in expr and "}}" in expr:
            out[key] = expr
            continue
        if not (
            expr.startswith("response.")
            or expr.startswith("meta.")
            or expr.startswith("args.")
            or expr.startswith("params.")
        ):
            expr = f"response.{expr}"
        out[key] = f"{{{{{expr}}}}}"
    return out


def execute_http_request_tool(ctx, *, tool_args: dict, meta: dict) -> dict:
    url = str(tool_args.get("url") or "").strip()
    if not url:
        return {"ok": False, "error": {"message": "Missing required tool arg: url"}}

    ctx_obj = {"meta": meta, "args": tool_args, "params": tool_args, "env": dict(os.environ)}
    url = ctx.render_template(url, ctx=ctx_obj)
    method = str(tool_args.get("method") or "GET").strip().upper() or "GET"

    headers_raw = tool_args.get("headers")
    headers_obj = coerce_json_object(headers_raw)
    if not headers_obj and isinstance(headers_raw, str):
        rendered_headers = ctx.render_template(headers_raw, ctx=ctx_obj)
        headers_obj = coerce_json_object(rendered_headers)
    headers_obj = render_templates_in_obj(ctx, headers_obj, ctx_obj=ctx_obj)
    headers_obj = ctx._normalize_headers_for_json(headers_obj)

    query_params = parse_query_params(ctx, tool_args.get("query"), ctx_obj=ctx_obj)

    body_raw = tool_args.get("body")
    body_obj: Any = None
    if isinstance(body_raw, (dict, list)):
        body_obj = render_templates_in_obj(ctx, body_raw, ctx_obj=ctx_obj)
    elif isinstance(body_raw, str):
        rendered_body = ctx.render_template(body_raw, ctx=ctx_obj).strip()
        if rendered_body:
            if rendered_body.startswith("{") or rendered_body.startswith("["):
                try:
                    body_obj = json.loads(rendered_body)
                except Exception:
                    body_obj = rendered_body
            else:
                body_obj = rendered_body
    else:
        body_obj = body_raw

    fields_required = parse_fields_required(tool_args.get("fields_required"))
    mapper_obj = coerce_json_object(tool_args.get("response_mapper_json"))
    if not mapper_obj and not fields_required:
        return {"ok": False, "error": {"message": "Missing required tool arg: fields_required"}}
    if not mapper_obj:
        mapper_obj = build_response_mapper_from_fields(fields_required)
    mapper_json = json.dumps(mapper_obj, ensure_ascii=False)

    timeout = httpx.Timeout(60.0, connect=20.0)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            req_kwargs = {"headers": headers_obj or None, "params": query_params or None}
            if method in ("GET", "HEAD"):
                resp = client.request(method, url, **req_kwargs)
            else:
                if isinstance(body_obj, (dict, list)):
                    resp = client.request(method, url, json=body_obj, **req_kwargs)
                elif body_obj is None:
                    resp = client.request(method, url, **req_kwargs)
                else:
                    resp = client.request(method, url, content=str(body_obj), **req_kwargs)
        if resp.status_code >= 400:
            err = ctx._http_error_response(
                url=str(resp.request.url),
                status_code=resp.status_code,
                body=(resp.text or None),
                message=resp.reason_phrase,
            )
            return {"ok": False, "error": err.get("__http_error__") or {"message": "HTTP error"}}
        try:
            response_json = resp.json()
        except Exception:
            response_json = {"raw": resp.text or ""}
        mapped = apply_response_mapper(
            ctx,
            mapper_json=mapper_json,
            response_json=response_json,
            meta=meta,
            tool_args=tool_args,
        )
        return {
            "ok": True,
            "status_code": int(resp.status_code),
            "url": str(resp.request.url),
            "data": mapped,
        }
    except httpx.RequestError as exc:
        err = ctx._http_error_response(url=url, status_code=None, body=None, message=str(exc))
        return {"ok": False, "error": err.get("__http_error__")}


def execute_integration_http(
    ctx,
    *,
    tool: IntegrationTool,
    meta: dict,
    tool_args: dict,
) -> dict:
    pagination_raw = (getattr(tool, "pagination_json", "") or "").strip()
    pagination_cfg: dict[str, Any] | None = None
    pagination_cfg_error: str | None = None
    if pagination_raw:
        try:
            obj = json.loads(pagination_raw)
            if isinstance(obj, dict):
                pagination_cfg = obj
            else:
                pagination_cfg_error = "pagination_json must be a JSON object."
        except Exception as exc:
            pagination_cfg_error = f"invalid pagination_json: {exc}"

    args0 = dict(tool_args or {})
    try:
        schema_obj = json.loads(getattr(tool, "parameters_schema_json", "") or "null")
    except Exception:
        schema_obj = None
    if isinstance(schema_obj, dict):
        args0 = ctx._apply_schema_defaults(schema_obj, args0)
    tool_args = args0

    required_args = ctx._parse_required_args_json(getattr(tool, "args_required_json", "[]"))
    missing = ctx._missing_required_args(required_args, tool_args or {})
    if bool(getattr(tool, "use_codex_response", False)):
        args0 = tool_args or {}
        if not str(args0.get("fields_required") or "").strip() and not str(args0.get("what_to_search_for") or "").strip():
            missing.append("fields_required")
        if not str(args0.get("why_api_was_called") or "").strip() and not str(args0.get("why_to_search_for") or "").strip():
            missing.append("why_api_was_called")
    if missing:
        return {
            "__tool_args_error__": {
                "missing": sorted(set(missing)),
                "message": f"Missing required tool args: {', '.join(sorted(set(missing)))}",
            }
        }

    def _single_request(*, loop_args: dict[str, Any]) -> tuple[Any, str]:
        ctx_obj = {"meta": meta, "args": loop_args, "params": loop_args, "env": dict(os.environ)}
        url = ctx.render_template(tool.url, ctx=ctx_obj)
        method = (tool.method or "GET").upper()

        headers_obj: dict[str, str] = {}
        headers_template = tool.headers_template_json or ""
        if headers_template.strip():
            rendered_headers = ctx.render_template(headers_template, ctx=ctx_obj)
            try:
                h = json.loads(rendered_headers)
                if isinstance(h, dict):
                    for k, v in h.items():
                        if isinstance(k, str) and isinstance(v, str) and k.strip():
                            headers_obj[k] = v
            except Exception:
                headers_obj = {}
        headers_obj = ctx._normalize_headers_for_json(headers_obj)

        body_template = tool.request_body_template or ""
        body_obj = None
        if body_template.strip():
            rendered_body = ctx.render_template(body_template, ctx=ctx_obj)
            try:
                body_obj = json.loads(rendered_body)
            except Exception:
                body_obj = rendered_body

        loop_request_args = dict(loop_args)
        for k in ("fields_required", "why_api_was_called", "what_to_search_for", "why_to_search_for", "max_items"):
            loop_request_args.pop(k, None)

        timeout = httpx.Timeout(60.0, connect=20.0)
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                if method == "GET":
                    resp = client.request(method, url, headers=headers_obj or None)
                else:
                    if isinstance(body_obj, (dict, list)):
                        resp = client.request(method, url, json=body_obj, headers=headers_obj or None)
                    elif body_obj is None:
                        resp = client.request(method, url, json=(loop_request_args or {}), headers=headers_obj or None)
                    else:
                        resp = client.request(method, url, content=str(body_obj), headers=headers_obj or None)
            if resp.status_code >= 400:
                return (
                    ctx._http_error_response(
                        url=str(resp.request.url),
                        status_code=resp.status_code,
                        body=(resp.text or None),
                        message=resp.reason_phrase,
                    ),
                    url,
                )
            try:
                return resp.json(), url
            except Exception:
                return {"raw": resp.text}, url
        except httpx.RequestError as exc:
            return ctx._http_error_response(url=url, status_code=None, body=None, message=str(exc)), url

    if not isinstance(pagination_cfg, dict) or not pagination_cfg:
        resp_json, _ = _single_request(loop_args=dict(tool_args or {}))
        if pagination_cfg_error and isinstance(resp_json, dict) and "__http_error__" not in resp_json:
            resp_json["__igx_pagination__"] = {"error": pagination_cfg_error}
        return resp_json

    mode = str(pagination_cfg.get("mode") or "page_limit").strip()
    if mode not in ("page_limit", "offset_limit"):
        resp_json, _ = _single_request(loop_args=dict(tool_args or {}))
        if isinstance(resp_json, dict) and "__http_error__" not in resp_json:
            resp_json["__igx_pagination__"] = {"error": f"unsupported pagination mode: {mode}"}
        return resp_json

    items_path = str(pagination_cfg.get("items_path") or "").strip()
    if not items_path:
        resp_json, _ = _single_request(loop_args=dict(tool_args or {}))
        if isinstance(resp_json, dict) and "__http_error__" not in resp_json:
            resp_json["__igx_pagination__"] = {"error": "pagination_json missing items_path."}
        return resp_json

    page_arg = str(pagination_cfg.get("page_arg") or "page").strip() or "page"
    limit_arg = str(pagination_cfg.get("limit_arg") or "limit").strip() or "limit"
    offset_arg = str(pagination_cfg.get("offset_arg") or "offset").strip() or "offset"
    max_pages = int(pagination_cfg.get("max_pages") or 5)
    if max_pages < 1:
        max_pages = 1
    if max_pages > 50:
        max_pages = 50

    max_items_cap = int(pagination_cfg.get("max_items_cap") or 5000)
    if max_items_cap < 1:
        max_items_cap = 5000
    if max_items_cap > 50000:
        max_items_cap = 50000

    requested_max_items = tool_args.get("max_items")
    if requested_max_items is None:
        requested_max_items = pagination_cfg.get("max_items_default")
    try:
        max_items = int(requested_max_items) if requested_max_items is not None else None
    except Exception:
        max_items = None
    if max_items is not None:
        if max_items < 1:
            max_items = None
        else:
            max_items = min(max_items, max_items_cap)

    def _read_int(v: Any, default: int) -> int:
        try:
            x = int(v)
            return x
        except Exception:
            return default

    limit_val = _read_int(tool_args.get(limit_arg), int(pagination_cfg.get("limit_default") or 100))
    if limit_val < 1:
        limit_val = 100

    start_page = _read_int(tool_args.get(page_arg), 1)
    if start_page < 1:
        start_page = 1

    start_offset = _read_int(tool_args.get(offset_arg), 0)
    if start_offset < 0:
        start_offset = 0

    base_resp: Any = None
    aggregated: list[Any] = []
    fetched = 0

    for i in range(max_pages):
        loop_args = dict(tool_args or {})
        loop_args[limit_arg] = limit_val
        if mode == "page_limit":
            loop_args[page_arg] = start_page + i
        else:
            loop_args[offset_arg] = start_offset + (i * limit_val)

        resp_json, _url = _single_request(loop_args=loop_args)
        if isinstance(resp_json, dict) and resp_json.get("__http_error__"):
            return resp_json

        if base_resp is None:
            base_resp = resp_json if not isinstance(resp_json, dict) else dict(resp_json)

        page_items = ctx._get_json_path(resp_json, items_path)
        if not isinstance(page_items, list):
            if isinstance(base_resp, dict):
                base_resp["__igx_pagination__"] = {
                    "mode": mode,
                    "items_path": items_path,
                    "limit": limit_val,
                    "pages_fetched": fetched,
                    "items_returned": len(aggregated),
                    "max_items": max_items,
                    "max_pages": max_pages,
                    "error": f"items_path not a list: {items_path}",
                }
            return base_resp

        fetched += 1
        aggregated.extend(page_items)

        if max_items is not None and len(aggregated) >= max_items:
            aggregated = aggregated[:max_items]
            break

        if len(page_items) < limit_val:
            break

    if base_resp is None:
        resp_json, _ = _single_request(loop_args=dict(tool_args or {}))
        return resp_json

    if not ctx._set_json_path(base_resp, items_path, aggregated):
        if isinstance(base_resp, dict):
            base_resp["__igx_pagination__"] = {
                "mode": mode,
                "items_path": items_path,
                "limit": limit_val,
                "pages_fetched": fetched,
                "items_returned": len(aggregated),
                "max_items": max_items,
                "max_pages": max_pages,
                "error": f"failed to set items_path: {items_path}",
            }
        return base_resp

    if isinstance(base_resp, dict):
        base_resp["__igx_pagination__"] = {
            "mode": mode,
            "items_path": items_path,
            "limit": limit_val,
            "pages_fetched": fetched,
            "items_returned": len(aggregated),
            "max_items": max_items,
            "max_pages": max_pages,
        }

    return base_resp
