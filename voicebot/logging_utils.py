from __future__ import annotations

import logging

from rich.logging import RichHandler


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    # Silence extremely chatty websocket frame logs even in debug mode.
    for name in (
        "websockets",
        "websockets.client",
        "websockets.server",
        "websockets.protocol",
        "websockets.connection",
    ):
        logging.getLogger(name).setLevel(logging.INFO)
