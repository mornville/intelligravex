from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from voicebot.asr.whisper_asr import WhisperASR
from voicebot.audio.devices import parse_device
from voicebot.audio.mic import Microphone
from voicebot.audio.playback import AudioPlayer
from voicebot.audio.vad import VadSegmenter
from voicebot.config import Settings
from voicebot.llm.openai_llm import Message, OpenAILLM
from voicebot.tts.xtts import XTTSv2
from voicebot.utils.text import SentenceChunker

log = logging.getLogger(__name__)


@dataclass
class _SpeakJob:
    text: str


class VoiceBotSession:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._drop_mic_audio = threading.Event()
        self._mic: Optional[Microphone] = None

        self._asr = WhisperASR(
            model_name=settings.whisper_model,
            device=settings.whisper_device,
            language=settings.language,
        )
        self._llm = OpenAILLM(model=settings.openai_model, api_key=settings.openai_api_key)
        self._tts = XTTSv2(
            model_name=settings.xtts_model,
            speaker_wav=settings.speaker_wav,
            speaker_id=settings.speaker_id,
            use_gpu=settings.tts_use_gpu,
            split_sentences=settings.tts_split_sentences,
            language=settings.tts_language,
        )
        self._player = AudioPlayer(device=parse_device(settings.output_device))
        self._vad = VadSegmenter(
            aggressiveness=settings.vad_aggressiveness,
            silence_ms=settings.vad_silence_ms,
            min_speech_ms=settings.vad_min_speech_ms,
            max_speech_ms=settings.vad_max_speech_ms,
        )

        self._history: List[Message] = [Message(role="system", content=settings.system_prompt)]

    def run_forever(self) -> None:
        if self.settings.input_sample_rate != 16000:
            raise ValueError("Set VOICEBOT_INPUT_SAMPLE_RATE=16000 (Whisper path is non-resampling).")

        log.info("Listening. Press Ctrl+C to stop.")

        with Microphone(
            sample_rate=self.settings.input_sample_rate,
            device=parse_device(self.settings.input_device),
            frame_ms=30,
            should_drop=(self._drop_mic_audio.is_set if self.settings.mic_mute_during_tts else None),
        ) as mic:
            self._mic = mic
            try:
                for utt in self._vad.utterances(mic.frames()):
                    self._handle_utterance(utt.pcm16, utt.sample_rate)
            except KeyboardInterrupt:
                log.info("Stopping.")
            finally:
                self._mic = None

    def _handle_utterance(self, pcm16: bytes, sample_rate: int) -> None:
        asr = self._asr.transcribe_pcm16(pcm16=pcm16, sample_rate=sample_rate)
        if not asr.text:
            return

        log.info("User: %s", asr.text)
        self._history.append(Message(role="user", content=asr.text))

        assistant_text = self._speak_streaming_reply()
        if assistant_text:
            self._history.append(Message(role="assistant", content=assistant_text))

    def _speak_streaming_reply(self) -> str:
        chunker = SentenceChunker(
            min_chars=self.settings.tts_chunk_min_chars,
            max_chars=self.settings.tts_chunk_max_chars,
        )
        assistant_accum: List[str] = []

        q_text: "queue.Queue[Optional[_SpeakJob]]" = queue.Queue()
        from voicebot.tts.xtts import TtsAudio

        q_audio: "queue.Queue[Optional[TtsAudio]]" = queue.Queue()

        if self.settings.mic_mute_during_tts:
            # Ensure we don't buffer TTS audio into the mic queue while generating/speaking.
            self._drop_mic_audio.set()

        def tts_worker() -> None:
            while True:
                job = q_text.get()
                if job is None:
                    q_audio.put(None)
                    return
                try:
                    audio = self._tts.synthesize(job.text)
                    q_audio.put(audio)
                except Exception:
                    log.exception("TTS synthesis failed")

        def player_worker() -> None:
            while True:
                item = q_audio.get()
                if item is None:
                    return
                try:
                    self._player.play_blocking(item.audio, item.sample_rate)
                except Exception:
                    log.exception("Audio playback failed")

        t_tts = threading.Thread(target=tts_worker, name="tts-synth", daemon=True)
        t_play = threading.Thread(target=player_worker, name="tts-play", daemon=True)
        t_tts.start()
        t_play.start()

        try:
            for delta in self._llm.stream_text(messages=self._history):
                assistant_accum.append(delta)
                for chunk in chunker.push(delta):
                    q_text.put(_SpeakJob(text=chunk))
        finally:
            tail = chunker.flush()
            if tail:
                q_text.put(_SpeakJob(text=tail))
            q_text.put(None)
            t_tts.join()
            t_play.join()

            if self.settings.mic_mute_during_tts:
                # Cooldown to let room echo die down a bit, then drop any queued frames.
                time.sleep(max(0, self.settings.tts_mic_release_ms) / 1000.0)
                self._drop_mic_audio.clear()
                if self._mic is not None:
                    dropped = self._mic.drain()
                    if dropped:
                        log.debug("Drained %d mic frames after TTS", dropped)

        full = "".join(assistant_accum).strip()
        if full:
            log.info("Assistant: %s", full)
        return full
