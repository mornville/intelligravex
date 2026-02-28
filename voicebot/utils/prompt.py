from __future__ import annotations

import datetime as dt
import platform
import sys

HOST_ACTION_APPROVAL_NOTICE = "You have generated the command, and user will be given the command for approval, instead of telling user you have done the task, tell that you have asked user for approval, and wait for user to approve the command, then you can execute the command and tell user you have done the task. If user does not approve the command, do not execute the command and tell user that you have canceled the task."


def _append_notice(base: str, notice: str) -> str:
    if not notice or notice in base:
        return base
    if base.strip():
        return f"{base.rstrip()}\n\n{notice}"
    return notice


def host_action_runtime_notice(*, host_actions_enabled: bool) -> str:
    if not host_actions_enabled:
        return ""
    if sys.platform == "darwin":
        os_name = "macOS"
        supported_actions = "run_shell, run_applescript"
        screenshot_status = "supported"
    elif sys.platform == "win32":
        os_name = "Windows"
        supported_actions = "run_shell, run_powershell"
        screenshot_status = "supported"
    else:
        os_name = platform.system() or sys.platform
        supported_actions = "run_shell"
        screenshot_status = "not supported"
    return (
        f"Host action runtime: this app is running on {os_name}. "
        f"When using request_host_action, supported action types are: {supported_actions}. "
        f"capture_screenshot is {screenshot_status} on this host."
    )


def append_host_action_approval_notice(
    prompt: str,
    *,
    require_approval: bool,
    host_actions_enabled: bool = False,
) -> str:
    base = str(prompt or "")
    base = _append_notice(base, host_action_runtime_notice(host_actions_enabled=host_actions_enabled))
    if require_approval:
        base = _append_notice(base, HOST_ACTION_APPROVAL_NOTICE)
    return base


def system_prompt_with_runtime(
    prompt: str,
    *,
    require_approval: bool,
    host_actions_enabled: bool = False,
) -> str:
    ts = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    prompt_with_notice = append_host_action_approval_notice(
        prompt,
        require_approval=require_approval,
        host_actions_enabled=host_actions_enabled,
    )
    return f"Current Date Time(UTC): {ts}\n\n{prompt_with_notice}"
