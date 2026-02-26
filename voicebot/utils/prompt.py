from __future__ import annotations

import datetime as dt

HOST_ACTION_APPROVAL_NOTICE = "You have generated the command, and user will be given the command for approval, instead of telling user you have done the task, tell that you have asked user for approval, and wait for user to approve the command, then you can execute the command and tell user you have done the task. If user does not approve the command, do not execute the command and tell user that you have canceled the task."


def append_host_action_approval_notice(prompt: str, *, require_approval: bool) -> str:
    if not require_approval:
        return prompt
    base = str(prompt or "")
    if HOST_ACTION_APPROVAL_NOTICE in base:
        return base
    if base.strip():
        return f"{base.rstrip()}\n\n{HOST_ACTION_APPROVAL_NOTICE}"
    return HOST_ACTION_APPROVAL_NOTICE


def system_prompt_with_runtime(prompt: str, *, require_approval: bool) -> str:
    ts = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    prompt_with_notice = append_host_action_approval_notice(prompt, require_approval=require_approval)
    return f"Current Date Time(UTC): {ts}\n\n{prompt_with_notice}"
