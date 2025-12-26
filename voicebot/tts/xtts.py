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
        speaker_wav: Optional[str],
        speaker_id: Optional[str],
        use_gpu: bool,
        split_sentences: bool,
        language: str,
    ) -> None:
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
        self._speaker_wav = speaker_wav
        self._speaker_id = speaker_id or self._default_speaker_id()
        self._language = language
        self._split_sentences = split_sentences

        if self._speaker_wav:
            log.info("XTTS voice: speaker_wav=%s", self._speaker_wav)
        elif self._speaker_id:
            log.info("XTTS voice: speaker_id=%s", self._speaker_id)
        else:
            log.warning("XTTS voice: no speaker_wav or speaker_id; synthesis may fail for XTTS.")

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
        speakers = getattr(self._tts, "speakers", None)
        if not speakers:
            return None
        return speakers[0]

    def synthesize(self, text: str) -> TtsAudio:
        wav = self._tts.tts(
            text=text,
            speaker=self._speaker_id,
            speaker_wav=self._speaker_wav,
            language=self._language,
            split_sentences=self._split_sentences,
        )
        audio = np.asarray(wav, dtype=np.float32)
        sr = int(getattr(self._tts.synthesizer, "output_sample_rate", 24000))
        return TtsAudio(audio=audio, sample_rate=sr)
