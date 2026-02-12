from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

import typer

from voicebot.config import Settings
from voicebot.dialog.session import VoiceBotSession
from voicebot.logging_utils import configure_logging

app = typer.Typer(add_completion=False, help="GravexStudio continuous AI voice bot.")


@app.command()
def run(
    bot: Optional[str] = typer.Option(
        None, "--bot", help="Bot UUID from the DB (overrides prompt/language/model/voice)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="OpenAI model (default: from env/config)."
    ),
    input_device: Optional[str] = typer.Option(
        None, "--input-device", help="Input device index/name (see `voicebot devices`)."
    ),
    output_device: Optional[str] = typer.Option(
        None, "--output-device", help="Output device index/name (see `voicebot devices`)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Run the continuous mic->ASR->LLM->TTS loop.
    """
    level = logging.DEBUG if verbose else logging.INFO
    configure_logging(level=level)
    base = Settings()

    bot_uuid = bot or base.bot_uuid
    if bot_uuid:
        from voicebot.crypto import get_crypto_box
        from voicebot.db import init_db, make_engine
        from voicebot.store import decrypt_provider_key, get_bot
        from sqlmodel import Session

        engine = make_engine(base.db_url)
        init_db(engine)
        with Session(engine) as session:
            bot_row = get_bot(session, UUID(bot_uuid))
            crypto = get_crypto_box(base.secret_key)
            api_key = decrypt_provider_key(session, crypto=crypto, provider="openai")

        settings = base.model_copy(
            update={
                "openai_model": bot_row.openai_model,
                "openai_asr_model": bot_row.openai_asr_model,
                "openai_api_key": api_key,
                "system_prompt": bot_row.system_prompt,
                "language": bot_row.language,
                "openai_tts_model": bot_row.openai_tts_model,
                "openai_tts_voice": bot_row.openai_tts_voice,
                "openai_tts_speed": bot_row.openai_tts_speed,
                "input_device": input_device or base.input_device,
                "output_device": output_device or base.output_device,
            }
        )
    else:
        settings = base.model_copy(
            update={
                "openai_model": model or base.openai_model,
                "input_device": input_device or base.input_device,
                "output_device": output_device or base.output_device,
            }
        )
    VoiceBotSession(settings).run_forever()


@app.command()
def doctor() -> None:
    """
    Print environment and dependency checks.
    """
    configure_logging(level=logging.INFO)
    Settings().print_diagnostics()


@app.command()
def devices() -> None:
    """
    List available audio devices.
    """
    from voicebot.audio.devices import list_audio_devices

    for line in list_audio_devices():
        print(line)


@app.command()
def web(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host (use 127.0.0.1 for localhost-only)."),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on file changes (dev only)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Run the bot configuration UI + test API server.
    """
    level = logging.DEBUG if verbose else logging.INFO
    configure_logging(level=level)
    logging.getLogger("voicebot.cli").info("web: starting uvicorn (host=%s port=%s reload=%s)", host, port, reload)
    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("Missing web dependencies. Install: pip install -e '.[web]'") from exc

    from voicebot.web.app import create_app

    try:
        fastapi_app = create_app()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Failed to initialize FastAPI app.") from exc

    async def _asgi_app(scope, receive, send):
        await fastapi_app(scope, receive, send)

    config = uvicorn.Config(
        _asgi_app,
        host=host,
        port=port,
        reload=reload,
        proxy_headers=False,
        server_header=False,
        date_header=False,
    )
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    app()
