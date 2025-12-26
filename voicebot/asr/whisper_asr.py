from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from voicebot.audio.pcm import pcm16_bytes_to_float32_mono

log = logging.getLogger(__name__)


@dataclass
class AsrResult:
    text: str
    language: Optional[str] = None


class WhisperASR:
    def __init__(self, *, model_name: str, device: str, language: Optional[str]) -> None:
        try:
            import whisper  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Whisper is not installed. Install extras: pip install -e '.[asr]'"
            ) from exc

        self._whisper = whisper
        resolved_device = self._resolve_device(device)
        log.info("Whisper device: %s", resolved_device)
        self._model = whisper.load_model(model_name, device=resolved_device)
        self._language = language

    @staticmethod
    def _resolve_device(device: str) -> str:
        d = (device or "auto").lower()
        if d != "auto":
            return d
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            # macOS Apple Silicon
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def transcribe_pcm16(self, *, pcm16: bytes, sample_rate: int) -> AsrResult:
        if sample_rate != 16000:
            raise ValueError("This app expects mic sample_rate=16000 to avoid resampling.")

        audio_f32: np.ndarray = pcm16_bytes_to_float32_mono(pcm16)
        # Whisper accepts numpy arrays directly (no ffmpeg needed).
        result = self._model.transcribe(
            audio_f32,
            language=self._language,
            task="transcribe",
            temperature=0.0,
            fp16=False,
            verbose=False,
        )
        text = (result.get("text") or "").strip()
        lang = result.get("language")
        return AsrResult(text=text, language=lang)
