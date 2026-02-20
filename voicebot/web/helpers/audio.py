from __future__ import annotations

import base64
import json
import threading
from functools import lru_cache
from typing import Callable, Generator, Optional

from voicebot.asr.openai_asr import OpenAIASR
from voicebot.tts.openai_tts import OpenAITTS


def ndjson(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def wav_bytes(audio, sample_rate: int) -> bytes:
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def decode_wav_bytes_to_pcm16_16k(wav_bytes: bytes) -> bytes:
    import io

    import numpy as np
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if isinstance(data, np.ndarray) and data.ndim > 1:
        data = data[:, 0]
    audio_f32 = np.asarray(data, dtype=np.float32)
    if sr != 16000:
        ratio = 16000.0 / float(sr)
        n_out = int(round(len(audio_f32) * ratio))
        if n_out <= 0:
            return b""
        x_old = np.linspace(0.0, 1.0, num=len(audio_f32), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        audio_f32 = np.interp(x_new, x_old, audio_f32).astype(np.float32)
    audio_i16 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
    return audio_i16.tobytes()


@lru_cache(maxsize=16)
def get_asr(api_key: str, model_name: str, language: str) -> OpenAIASR:
    lang = None if (language or "").lower() in ("", "auto") else language
    return OpenAIASR(api_key=api_key, model=model_name, language=lang)


@lru_cache(maxsize=16)
def get_openai_tts_handle(
    api_key: str,
    model: str,
    voice: str,
    speed: float,
) -> tuple[OpenAITTS, threading.Lock]:
    return (
        OpenAITTS(api_key=api_key, model=model, voice=voice, speed=speed),
        threading.Lock(),
    )


def get_tts_synth_fn(bot, api_key: Optional[str]) -> Callable[[str], tuple[bytes, int]]:
    if not api_key:
        raise RuntimeError("No OpenAI API key configured for OpenAI TTS.")
    model = (getattr(bot, "openai_tts_model", None) or "gpt-4o-mini-tts").strip()
    voice = (getattr(bot, "openai_tts_voice", None) or "alloy").strip()
    speed_raw = getattr(bot, "openai_tts_speed", None)
    try:
        speed = float(speed_raw) if speed_raw is not None else 1.0
    except Exception:
        speed = 1.0

    tts, lock = get_openai_tts_handle(api_key, model, voice, speed)

    def synth(text: str) -> tuple[bytes, int]:
        with lock:
            wav = tts.synthesize_wav_bytes(text)
        return wav, OpenAITTS.DEFAULT_SAMPLE_RATE

    return synth


def estimate_wav_seconds(wav_bytes: bytes, sr: int) -> float:
    if not wav_bytes:
        return 0.0
    if sr <= 0:
        return 0.0
    sample_bytes = 2
    try:
        n_samples = len(wav_bytes) // sample_bytes
        return float(n_samples) / float(sr)
    except Exception:
        return 0.0


def iter_tts_chunks(delta_q: "queue.Queue[Optional[str]]") -> Generator[str, None, None]:
    buf = ""
    while True:
        part = delta_q.get()
        if part is None:
            if buf:
                yield buf
            return
        buf += part
        if len(buf) < 120:
            continue
        # Split on sentence boundaries where possible
        cut = max(buf.rfind("."), buf.rfind("!"), buf.rfind("?"))
        if cut <= 0:
            cut = buf.rfind(" ")
        if cut <= 0:
            continue
        chunk, buf = buf[: cut + 1], buf[cut + 1 :].lstrip()
        if chunk.strip():
            yield chunk.strip()
