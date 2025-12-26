from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class TtsAudio:
    audio: np.ndarray
    sample_rate: int


class XTTSv2:
    def __init__(
        self,
        *,
        model_name: str,
        use_gpu: bool,
    ) -> None:
        # Ensure MPS shims are present before importing Coqui TTS.
        try:
            from voicebot.compat.torch_mps import ensure_torch_mps_compat

            ensure_torch_mps_compat()
        except Exception:
            pass
        try:
            from TTS.api import TTS  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Coqui TTS is not installed. Install extras: pip install -e '.[tts]'"
            ) from exc

        # NOTE: Coqui's `gpu=True` path assumes CUDA. On macOS we can still speed up by moving
        # the underlying torch module to MPS.
        self._tts = TTS(model_name=model_name, gpu=False, progress_bar=False)
        self._device = self._resolve_device(use_gpu)
        try:
            self._tts.synthesizer.tts_model.to(self._device)
            log.info("XTTS device: %s", self._device)
        except Exception:
            log.exception("Failed to move XTTS model to device=%s; falling back to CPU", self._device)
            self._device = "cpu"
            self._tts.synthesizer.tts_model.to(self._device)
        self._model_name = model_name
        self._speakers = list(getattr(self._tts, "speakers", None) or [])
        self._languages = list(getattr(self._tts, "languages", None) or [])

    @staticmethod
    def _resolve_device(requested: bool) -> str:
        if not requested:
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _default_speaker_id(self) -> Optional[str]:
        if not self._speakers:
            return None
        return self._speakers[0]

    def meta(self) -> dict:
        return {"speakers": list(self._speakers), "languages": list(self._languages)}

    def synthesize(
        self,
        text: str,
        *,
        speaker_wav: Optional[str],
        speaker_id: Optional[str],
        language: str,
        split_sentences: bool,
    ) -> TtsAudio:
        speaker = (speaker_id or "").strip() or self._default_speaker_id()
        wav = self._tts.tts(
            text=text,
            speaker=speaker,
            speaker_wav=(speaker_wav or "").strip() or None,
            language=(language or "").strip() or None,
            split_sentences=bool(split_sentences),
        )
        audio = np.asarray(wav, dtype=np.float32)
        sr = int(getattr(self._tts.synthesizer, "output_sample_rate", 24000))
        return TtsAudio(audio=audio, sample_rate=sr)
