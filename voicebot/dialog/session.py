from __future__ import annotations

import logging
import os
import threading
import time
from typing import List, Optional

from voicebot.asr.whisper_asr import WhisperASR
from voicebot.audio.devices import parse_device
from voicebot.audio.mic import Microphone
from voicebot.audio.playback import AudioPlayer
from voicebot.audio.vad import VadSegmenter
from voicebot.config import Settings
from voicebot.llm.openai_llm import Message, OpenAILLM
from voicebot.tts.openai_tts import OpenAITTS
from voicebot.tts.xtts import XTTSv2

log = logging.getLogger(__name__)


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
        tts_vendor = (settings.tts_vendor or "xtts_local").strip().lower()
        if tts_vendor == "openai_tts":
            self._tts = OpenAITTS(
                api_key=settings.openai_api_key or os.environ.get("OPENAI_API_KEY"),
                model=settings.openai_tts_model,
                voice=settings.openai_tts_voice,
                speed=settings.openai_tts_speed,
            )
        else:
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
        assistant_accum: List[str] = []
        try:
            for delta in self._llm.stream_text(messages=self._history):
                assistant_accum.append(delta)
        finally:
            pass

        full = "".join(assistant_accum).strip()
        if full:
            log.info("Assistant: %s", full)
            if self.settings.mic_mute_during_tts:
                self._drop_mic_audio.set()
            try:
                audio = self._tts.synthesize(full)
                self._player.play_blocking(audio.audio, audio.sample_rate)
            except Exception:
                log.exception("TTS synthesis/playback failed")
            finally:
                if self.settings.mic_mute_during_tts:
                    time.sleep(max(0, self.settings.tts_mic_release_ms) / 1000.0)
                    self._drop_mic_audio.clear()
                    if self._mic is not None:
                        dropped = self._mic.drain()
                        if dropped:
                            log.debug("Drained %d mic frames after TTS", dropped)
        return full
