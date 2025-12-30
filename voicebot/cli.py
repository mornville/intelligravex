from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

import typer

from voicebot.config import Settings
from voicebot.dialog.session import VoiceBotSession
from voicebot.logging_utils import configure_logging

app = typer.Typer(add_completion=False, help="Intelligravex continuous AI voice bot.")


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
        from voicebot.store import decrypt_openai_key, get_bot
        from sqlmodel import Session

        engine = make_engine(base.db_url)
        init_db(engine)
        with Session(engine) as session:
            bot_row = get_bot(session, UUID(bot_uuid))
            if bot_row.openai_key_id:
                crypto = get_crypto_box(base.secret_key)
                api_key = decrypt_openai_key(session, crypto=crypto, bot=bot_row)
            else:
                api_key = None

        settings = base.model_copy(
            update={
                "openai_model": bot_row.openai_model,
                "openai_api_key": api_key,
                "system_prompt": bot_row.system_prompt,
                "language": bot_row.language,
                "whisper_model": bot_row.whisper_model,
                "whisper_device": bot_row.whisper_device,
                "tts_vendor": bot_row.tts_vendor,
                "xtts_model": bot_row.xtts_model,
                "speaker_wav": bot_row.speaker_wav,
                "speaker_id": bot_row.speaker_id,
                "tts_language": bot_row.tts_language,
                "openai_tts_model": bot_row.openai_tts_model,
                "openai_tts_voice": bot_row.openai_tts_voice,
                "openai_tts_speed": bot_row.openai_tts_speed,
                "tts_split_sentences": bot_row.tts_split_sentences,
                "tts_chunk_min_chars": bot_row.tts_chunk_min_chars,
                "tts_chunk_max_chars": bot_row.tts_chunk_max_chars,
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


@app.command("tts-speakers")
def tts_speakers() -> None:
    """
    List available speaker IDs for the configured TTS model.
    """
    settings = Settings()
    try:
        from TTS.api import TTS  # type: ignore
    except Exception as exc:
        raise typer.Exit(code=1) from exc

    tts = TTS(model_name=settings.xtts_model)
    speakers = getattr(tts, "speakers", None) or []
    languages = getattr(tts, "languages", None) or []
    if languages:
        print("languages:", ", ".join(languages))
    if not speakers:
        print("No speakers exposed by this model.")
        raise typer.Exit(code=0)
    for s in speakers:
        print(s)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on file changes (dev only)."),
) -> None:
    """
    Run the bot configuration UI + test API server.
    """
    configure_logging(level=logging.INFO)
    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("Missing web dependencies. Install: pip install -e '.[web]'") from exc

    uvicorn.run("voicebot.web.app:create_app", host=host, port=port, reload=reload, factory=True)


if __name__ == "__main__":
    app()
