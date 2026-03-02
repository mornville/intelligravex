from __future__ import annotations

import json
from typing import Any

from voicebot.models import Bot


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _to_int(value: Any) -> int:
    try:
        n = int(value)
    except Exception:
        return 0
    return n if n > 0 else 0


def _auth_obj(raw: str) -> dict[str, Any]:
    try:
        obj = json.loads((raw or "").strip() or "{}")
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    return obj


def _connected_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("connected"), bool):
            return bool(value.get("connected"))
        return bool(_safe_str(value.get("account_email")) or _safe_str(value.get("oauth_account")))
    return False


def _bool_text(value: bool) -> str:
    return "yes" if value else "no"


def build_connected_apps_prompt_context(bot: Bot, *, conversation_meta: dict[str, Any] | None = None) -> str:
    auth = _auth_obj(getattr(bot, "data_agent_auth_json", "") or "{}")
    connected_apps = _as_dict(auth.get("connected_apps"))
    git_integrations = _as_dict(auth.get("git_integrations"))
    runtime_meta = _as_dict(_as_dict(conversation_meta).get("data_agent"))

    lines: list[str] = []
    lines.append("Connected apps and Isolated Workspace context (sanitized):")

    data_agent_enabled = bool(getattr(bot, "enable_data_agent", False))
    lines.append(f"- Isolated Workspace (data agent) enabled for this assistant: {_bool_text(data_agent_enabled)}.")
    if data_agent_enabled:
        lines.append(
            "- For repo/integration/file tasks requiring external credentials, prefer `give_command_to_data_agent`."
        )
        container_name = _safe_str(runtime_meta.get("container_name"))
        container_id = _safe_str(runtime_meta.get("container_id"))
        workspace_dir = _safe_str(runtime_meta.get("workspace_dir"))
        ide_port = _to_int(runtime_meta.get("ide_port"))
        port_pairs: list[tuple[int, int]] = []
        for item in _as_list(runtime_meta.get("ports")):
            if not isinstance(item, dict):
                continue
            host = _to_int(item.get("host"))
            container = _to_int(item.get("container"))
            if not host or not container:
                continue
            port_pairs.append((host, container))
        dedup_pairs = sorted(set(port_pairs), key=lambda p: p[0])
        runtime_bits: list[str] = []
        if container_name:
            runtime_bits.append(f"container_name={container_name}")
        if container_id:
            runtime_bits.append(f"container_id={container_id}")
        if workspace_dir:
            runtime_bits.append(f"workspace_dir={workspace_dir}")
        if runtime_bits:
            lines.append(f"- Active Isolated Workspace runtime: {'; '.join(runtime_bits)}.")
        if dedup_pairs:
            mapping = ", ".join(f"{h}->{c}" for h, c in dedup_pairs)
            lines.append(f"- Allowed published ports (host->container): {mapping}.")
        else:
            lines.append("- Allowed published ports (host->container): not assigned yet.")
        if ide_port:
            lines.append(f"- IDE host port: {ide_port} (reserved for OpenVSCode server).")
        lines.append(
            "- For host access, run services on one of the allowed container ports above and bind to 0.0.0.0."
        )
        lines.append("- Ports not listed above are not published to the host.")
    else:
        lines.append("- If user asks for repo/integration/file automation, ask them to enable Isolated Workspace.")

    ssh_key_available = bool(
        _safe_str(auth.get("ssh_private_key"))
        or _safe_str(auth.get("ssh_key"))
        or _safe_str(auth.get("ssh_private_key_pem"))
        or _safe_str(auth.get("ssh_private_key_b64"))
        or _safe_str(auth.get("ssh_private_key_base64"))
        or _safe_str(auth.get("ssh_private_key_path"))
        or _safe_str(auth.get("ssh_key_path"))
    )
    lines.append(f"- Workspace SSH key available: {_bool_text(ssh_key_available)}.")

    selected_provider = _safe_str(auth.get("git_provider")).lower()
    selected_provider = selected_provider if selected_provider in {"github", "gitlab"} else "github"

    def _git_line(provider: str, label: str) -> str:
        cfg = _as_dict(git_integrations.get(provider))
        auth_mode = _safe_str(cfg.get("auth_mode")).lower()
        oauth_account = _safe_str(cfg.get("oauth_account"))
        token_present = bool(
            _safe_str(cfg.get("token"))
            or _safe_str(auth.get(f"{provider}_token"))
            or _safe_str(auth.get(f"{provider.upper()}_TOKEN"))
        )
        if not auth_mode:
            if oauth_account:
                auth_mode = "oauth"
            elif token_present:
                auth_mode = "pat"
            elif ssh_key_available:
                auth_mode = "ssh"
            else:
                auth_mode = "unknown"
        repo_url = _safe_str(cfg.get("repo_url"))
        if not repo_url and selected_provider == provider:
            repo_url = _safe_str(
                auth.get("preferred_repo_url")
                or auth.get("git_preferred_repo_url")
                or auth.get("git_repo_url")
                or auth.get("preferred_repo")
            )
        connected = _connected_flag(connected_apps.get(provider)) or token_present or bool(oauth_account)
        return (
            f"- {label}: {'connected' if connected else 'not connected'}; "
            f"auth: {auth_mode}; repo linked: {_bool_text(bool(repo_url))}."
        )

    lines.append(_git_line("github", "GitHub"))
    lines.append(_git_line("gitlab", "GitLab"))

    jira = _as_dict(connected_apps.get("jira")) or _as_dict(auth.get("jira_integration"))
    jira_domain = _safe_str(jira.get("domain") or jira.get("site") or jira.get("base_url"))
    jira_email = _safe_str(jira.get("email"))
    jira_project = _safe_str(jira.get("default_project_key") or jira.get("project_key"))
    jira_issue_type = _safe_str(jira.get("default_issue_type") or jira.get("issue_type"))
    jira_connected = bool(jira_domain and (jira_email or _safe_str(jira.get("api_token")) or _safe_str(jira.get("token"))))
    jira_bits: list[str] = [f"- Jira: {'connected' if jira_connected else 'not connected'}."]
    if jira_domain:
        jira_bits.append(f"domain={jira_domain}")
    if jira_project:
        jira_bits.append(f"default_project={jira_project}")
    if jira_issue_type:
        jira_bits.append(f"default_issue_type={jira_issue_type}")
    lines.append("; ".join(jira_bits))

    gmail = _as_dict(connected_apps.get("gmail")) or _as_dict(auth.get("gmail_integration"))
    gmail_connected = _connected_flag(gmail) or bool(_safe_str(auth.get("gmail_refresh_token")) or _safe_str(auth.get("gmail_access_token")))
    gmail_email = _safe_str(gmail.get("account_email") or auth.get("gmail_account_email"))
    scope = _safe_str(gmail.get("scope") or auth.get("gmail_scope"))
    scope_caps: list[str] = []
    if "gmail.send" in scope:
        scope_caps.append("send")
    if "gmail.readonly" in scope or "gmail.modify" in scope or "gmail.metadata" in scope:
        scope_caps.append("read")
    caps_text = ", ".join(scope_caps) if scope_caps else "none"
    gmail_line = f"- Gmail: {'connected' if gmail_connected else 'not connected'}; capabilities={caps_text}."
    if gmail_email:
        gmail_line += f" account={gmail_email}."
    lines.append(gmail_line)

    db_creds = _as_list(connected_apps.get("database_credentials") or connected_apps.get("db_credentials"))
    if not db_creds:
        db_creds = _as_list(auth.get("db_credentials") or auth.get("database_credentials"))
    db_profiles: list[str] = []
    for item in db_creds:
        if not isinstance(item, dict):
            continue
        nickname = _safe_str(item.get("nickname") or item.get("name"))
        engine = _safe_str(item.get("engine") or item.get("driver"))
        host = _safe_str(item.get("host"))
        database = _safe_str(item.get("database") or item.get("db"))
        bits = [nickname or "unnamed", engine or "unknown-engine"]
        if host:
            bits.append(f"host={host}")
        if database:
            bits.append(f"db={database}")
        db_profiles.append(" / ".join(bits))
    if db_profiles:
        preview = "; ".join(db_profiles[:3])
        if len(db_profiles) > 3:
            preview += f"; +{len(db_profiles) - 3} more"
        lines.append(f"- Database credentials: {len(db_profiles)} profile(s). {preview}.")
    else:
        lines.append("- Database credentials: none configured.")

    lines.append("- Never reveal or print secrets/tokens/passwords from connected app config.")
    lines.append("- If a requested action needs a tool that is unavailable, explain what to enable next.")
    return "\n".join(lines).strip()
