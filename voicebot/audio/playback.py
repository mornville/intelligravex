from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class AudioPlayer:
    device: Optional[Union[int, str]] = None

    def play_blocking(self, audio: np.ndarray, sample_rate: int) -> None:
        import sounddevice as sd

        sd.play(audio, samplerate=sample_rate, device=self.device, blocking=True)

    def stop(self) -> None:
        import sounddevice as sd

        sd.stop()
