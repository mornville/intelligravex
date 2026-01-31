from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AsrResult:
    text: str
    language: Optional[str] = None


class OpenAIASR:
    def __init__(self, *, api_key: Optional[str], model: str, language: Optional[str]) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("openai sdk not installed; pip install openai") from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("No OpenAI API key found (set OPENAI_API_KEY or add a key in Studio).")
        self._client = OpenAI(api_key=key)
        self._model = (model or "gpt-4o-mini-transcribe").strip()
        self._language = (language or "").strip() or None

    @staticmethod
    def _pcm16_to_wav_bytes(pcm16: bytes, sample_rate: int) -> bytes:
        if not pcm16:
            return b""
        audio_i16 = np.frombuffer(pcm16, dtype=np.int16)
        buf = io.BytesIO()
        try:
            import soundfile as sf
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("soundfile is required to encode WAV bytes") from exc
        sf.write(buf, audio_i16, samplerate=sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def transcribe_pcm16(self, *, pcm16: bytes, sample_rate: int) -> AsrResult:
        wav = self._pcm16_to_wav_bytes(pcm16, sample_rate)
        if not wav:
            return AsrResult(text="")
        bio = io.BytesIO(wav)
        bio.name = "audio.wav"
        kwargs = {"model": self._model, "file": bio}
        if self._language:
            kwargs["language"] = self._language
        resp = self._client.audio.transcriptions.create(**kwargs)
        text = ""
        if isinstance(resp, dict):
            text = str(resp.get("text") or "").strip()
        else:
            text = str(getattr(resp, "text", "") or "").strip()
        return AsrResult(text=text, language=self._language)
