from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable

import webrtcvad

from voicebot.audio.mic import MicFrame


@dataclass(frozen=True)
class Utterance:
    pcm16: bytes
    sample_rate: int


class VadSegmenter:
    """
    Frame-based utterance segmenter using webrtcvad.
    """

    def __init__(
        self,
        *,
        aggressiveness: int,
        silence_ms: int,
        min_speech_ms: int,
        max_speech_ms: int,
    ) -> None:
        self._vad = webrtcvad.Vad(aggressiveness)
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms

    def utterances(self, frames: Iterable[MicFrame]) -> Generator[Utterance, None, None]:
        in_speech = False
        speech_ms = 0
        silence_ms = 0
        buf: list[bytes] = []
        sample_rate = 16000
        frame_ms = 30

        for frame in frames:
            sample_rate = frame.sample_rate
            frame_ms = frame.frame_ms
            is_speech = self._vad.is_speech(frame.pcm16, sample_rate)

            if not in_speech:
                if is_speech:
                    in_speech = True
                    speech_ms = frame_ms
                    silence_ms = 0
                    buf = [frame.pcm16]
                continue

            # in_speech
            buf.append(frame.pcm16)
            speech_ms += frame_ms
            if is_speech:
                silence_ms = 0
            else:
                silence_ms += frame_ms

            should_end = False
            if silence_ms >= self.silence_ms:
                should_end = True
            if speech_ms >= self.max_speech_ms:
                should_end = True

            if should_end:
                if speech_ms >= self.min_speech_ms:
                    yield Utterance(pcm16=b"".join(buf), sample_rate=sample_rate)
                in_speech = False
                speech_ms = 0
                silence_ms = 0
                buf = []

        # flush at end
        if in_speech and speech_ms >= self.min_speech_ms:
            yield Utterance(pcm16=b"".join(buf), sample_rate=sample_rate)

