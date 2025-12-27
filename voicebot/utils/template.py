from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional


_VAR_RE = re.compile(r"{{\s*([^}]+?)\s*}}")


def render_template(text: str, *, ctx: dict[str, Any]) -> str:
    """
    Very small template renderer for strings like:
      - {{.first_name}}              (ctx["meta"] or ctx root)
      - {{meta.user.first_name}}
      - {{response.data.first_name}}
      - {{args.user_id}}

    Missing values render as empty string.
    """
    if not text or "{{" not in text:
        return text

    def repl(m: re.Match) -> str:
        expr = (m.group(1) or "").strip()
        if not expr:
            return ""
        try:
            v = resolve_expr(expr, ctx=ctx)
        except Exception:
            return ""
        return "" if v is None else str(v)

    return _VAR_RE.sub(repl, text)


def eval_template_value(template: str, *, ctx: dict[str, Any]) -> Any:
    """
    Evaluate a template intended to map to JSON values.
    - If the template is exactly one {{expr}}, returns the raw resolved value.
    - Otherwise returns the rendered string.
    """
    if not template:
        return None
    s = template.strip()
    m = _VAR_RE.fullmatch(s)
    if m:
        expr = (m.group(1) or "").strip()
        return resolve_expr(expr, ctx=ctx)
    return render_template(template, ctx=ctx)


def resolve_expr(expr: str, *, ctx: dict[str, Any]) -> Any:
    # Support shorthand {{.path}} as metadata lookup.
    if expr.startswith("."):
        meta = ctx.get("meta") if isinstance(ctx, dict) else None
        if isinstance(meta, dict):
            return resolve_path(meta, expr[1:])
        return resolve_path(ctx, expr[1:])  # fallback

    # First segment selects a root object.
    parts = expr.split(".", 1)
    root_name = parts[0]
    rest = parts[1] if len(parts) == 2 else ""
    root = ctx.get(root_name) if isinstance(ctx, dict) else None
    if root is None and root_name == "meta":
        root = ctx.get("meta")
    if rest:
        return resolve_path(root, rest)
    return root


_SEG_RE = re.compile(r"([A-Za-z0-9_-]+)|\[(\d+)\]")


def resolve_path(obj: Any, path: str) -> Any:
    """
    Resolve a dotted path with optional [index] segments.
    Examples:
      - user.first_name
      - data.items[0].id
    """
    if path == "" or obj is None:
        return obj
    cur = obj
    for token in _iter_path_tokens(path):
        if cur is None:
            return None
        if isinstance(token, int):
            if isinstance(cur, (list, tuple)) and 0 <= token < len(cur):
                cur = cur[token]
            else:
                return None
        else:
            if isinstance(cur, dict):
                cur = cur.get(token)
            else:
                cur = getattr(cur, token, None)
    return cur


def _iter_path_tokens(path: str) -> Iterable[str | int]:
    for seg in path.split("."):
        seg = seg.strip()
        if not seg:
            continue
        for m in _SEG_RE.finditer(seg):
            key = m.group(1)
            # NOTE: regex must have two capturing groups; group(2) is index.
            idx = m.group(2) if m.re.groups >= 2 else None
            if key is not None:
                yield key
            elif idx is not None:
                yield int(idx)


def safe_json_loads(s: str) -> Optional[dict]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def render_jinja_template(text: str, *, ctx: dict[str, Any]) -> str:
    """
    Render a Jinja2 template (supports if/for/etc).

    Convenience: occurrences of `.foo` in expressions are rewritten to `meta.foo`,
    so you can write `{{.firstName}}` and `{% if .firstName %}`.
    """
    if not text:
        return ""
    try:
        from jinja2.sandbox import SandboxedEnvironment  # type: ignore
        from jinja2 import Undefined  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("jinja2 is required to render static_reply templates") from exc

    # Replace `.foo` tokens (not part of numeric literals or other identifiers) with `meta.foo`.
    # This lets users write `{{.firstName}}` or `{% if .vip %}` in Jinja templates.
    src = re.sub(r"(?<![A-Za-z0-9_])\.([A-Za-z_][A-Za-z0-9_]*)", r"meta.\1", text)

    env = SandboxedEnvironment(autoescape=False, undefined=Undefined)
    try:
        tmpl = env.from_string(src)
        return str(tmpl.render(**(ctx or {})))
    except Exception:
        # Fail closed (empty) rather than crashing the conversation.
        return ""
