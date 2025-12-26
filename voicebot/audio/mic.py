from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Callable, Generator, Optional, Union


@dataclass(frozen=True)
class MicFrame:
    pcm16: bytes
    sample_rate: int
    frame_ms: int


class Microphone:
    def __init__(
        self,
        *,
        sample_rate: int,
        device: Optional[Union[int, str]],
        frame_ms: int = 30,
        queue_max_frames: int = 200,
        should_drop: Optional[Callable[[], bool]] = None,
    ) -> None:
        if frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30 for webrtcvad compatibility")
        self.sample_rate = sample_rate
        self.device = device
        self.frame_ms = frame_ms
        self._should_drop = should_drop
        self._frames: "queue.Queue[MicFrame]" = queue.Queue(maxsize=queue_max_frames)
        self._stream = None

    def __enter__(self) -> "Microphone":
        import sounddevice as sd
        import numpy as np

        frames_per_block = int(self.sample_rate * self.frame_ms / 1000)

        def callback(indata: np.ndarray, _frames: int, _time, status) -> None:
            if self._should_drop is not None and self._should_drop():
                return
            if status:
                # Drop frames on overflow/underflow; log upstream if needed.
                pass
            if indata.ndim == 2:
                mono = indata[:, 0]
            else:
                mono = indata
            pcm16 = mono.astype(np.int16, copy=False).tobytes()
            try:
                self._frames.put_nowait(
                    MicFrame(pcm16=pcm16, sample_rate=self.sample_rate, frame_ms=self.frame_ms)
                )
            except queue.Full:
                # If consumer is slow, drop frames to keep latency bounded.
                pass

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=self.device,
            channels=1,
            dtype="int16",
            blocksize=frames_per_block,
            callback=callback,
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
        self._stream = None

    def frames(self) -> Generator[MicFrame, None, None]:
        while True:
            yield self._frames.get()

    def drain(self) -> int:
        """
        Drop any queued audio frames (useful after TTS playback to avoid echo being transcribed).
        """
        dropped = 0
        while True:
            try:
                self._frames.get_nowait()
                dropped += 1
            except queue.Empty:
                return dropped
