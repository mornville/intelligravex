from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from voicebot.tts.xtts import TtsAudio


@dataclass(frozen=True)
class OpenAITtsConfig:
    model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"
    speed: float = 1.0


class OpenAITTS:
    """
    Thin wrapper around OpenAI Text-to-Speech.

    Uses OpenAI Python SDK `client.audio.speech.create(...)` with:
    - `input` (text)
    - `model`
    - `voice`
    - `response_format="wav"`
    - `speed` (optional)
    """

    # OpenAI TTS WAV output is typically 24kHz; used as a fallback if callers skip decoding the WAV.
    DEFAULT_SAMPLE_RATE = 24000

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        speed: float = 1.0,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("openai sdk not installed; pip install openai") from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("No OpenAI API key found (set OPENAI_API_KEY or configure a bot key).")

        self._client = OpenAI(api_key=key)
        self._model = (model or "gpt-4o-mini-tts").strip()
        self._voice = (voice or "alloy").strip()
        self._speed = float(speed) if speed else 1.0

    def synthesize_wav_bytes(self, text: str) -> bytes:
        text = (text or "").strip()
        if not text:
            return b""
        resp = self._client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format="wav",
            speed=self._speed,
        )
        return resp.read()

    def synthesize(self, text: str) -> TtsAudio:
        wav = self.synthesize_wav_bytes(text)
        if not wav:
            return TtsAudio(audio=np.zeros((0,), dtype=np.float32), sample_rate=self.DEFAULT_SAMPLE_RATE)

        try:
            import soundfile as sf
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("soundfile is required to decode WAV bytes") from exc

        audio, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = audio[:, 0]
        audio_f32 = np.asarray(audio, dtype=np.float32)
        return TtsAudio(audio=audio_f32, sample_rate=int(sr or self.DEFAULT_SAMPLE_RATE))
