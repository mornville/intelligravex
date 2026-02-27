from __future__ import annotations

from . import bots, chatgpt_oauth, chatgpt_proxy, conversations, core, data_agent, downloads, files, groups, host_actions, ide_proxy, keys, public, public_widget, status, streaming, ws


def register_all(app, ctx) -> None:
    core.register(app, ctx)
    public_widget.register(app, ctx)
    status.register(app, ctx)
    downloads.register(app, ctx)
    data_agent.register(app, ctx)
    bots.register(app, ctx)
    chatgpt_oauth.register(app, ctx)
    chatgpt_proxy.register(app, ctx)
    keys.register(app, ctx)
    conversations.register(app, ctx)
    groups.register(app, ctx)
    public.register(app, ctx)
    files.register(app, ctx)
    host_actions.register(app, ctx)
    ide_proxy.register(app, ctx)
    streaming.register(app, ctx)
    ws.register(app, ctx)
