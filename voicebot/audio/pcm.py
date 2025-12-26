from __future__ import annotations

import numpy as np


def pcm16_bytes_to_float32_mono(pcm16: bytes) -> np.ndarray:
    """
    Convert little-endian 16-bit PCM bytes to float32 mono in [-1, 1].
    """
    audio_i16 = np.frombuffer(pcm16, dtype=np.int16)
    audio_f32 = (audio_i16.astype(np.float32)) / 32768.0
    return audio_f32

