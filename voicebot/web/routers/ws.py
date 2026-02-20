from __future__ import annotations


def register(app, ctx) -> None:
    app.add_api_websocket_route("/ws/bots/{bot_id}/talk", ctx.talk_ws)
    app.add_api_websocket_route("/public/v1/ws/bots/{bot_id}/chat", ctx.public_chat_ws)
