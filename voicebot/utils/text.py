from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from typing import Generator


_SENTENCE_END_RE = re.compile(r"([.!?]+)(?:\s+|$)")
_SOFT_BREAK_RE = re.compile(r"([,:;]+)(?:\s+|$)")


@dataclass
class SentenceChunker:
    """
    Incrementally splits a stream of text deltas into speakable chunks.
    """

    buffer: str = ""
    min_chars: int = 40
    max_chars: int = 180

    def push(self, delta: str) -> Generator[str, None, None]:
        self.buffer += delta
        while True:
            cut = self._find_cut(self.buffer)
            if cut is None:
                return
            chunk = self.buffer[:cut].strip()
            self.buffer = self.buffer[cut:]
            if chunk:
                yield chunk

    def flush(self) -> str:
        chunk = self.buffer.strip()
        self.buffer = ""
        return chunk

    def _find_cut(self, text: str) -> Optional[int]:
        # Cut at the end of a sentence (punctuation + whitespace), or at a newline.
        nl = text.find("\n")
        if nl != -1:
            return nl + 1

        m = _SENTENCE_END_RE.search(text)
        if not m:
            # If we don't have a complete sentence yet, cut a smaller chunk to reduce time-to-first-audio.
            if len(text) >= self.max_chars:
                return self._cut_at_space(text, self.max_chars)
            if len(text) >= self.min_chars:
                soft = _SOFT_BREAK_RE.search(text)
                if soft:
                    return soft.end()
            return None
        return m.end()

    @staticmethod
    def _cut_at_space(text: str, max_chars: int) -> int:
        # Prefer cutting at a whitespace boundary to avoid chopping words mid-way.
        if len(text) <= max_chars:
            return len(text)
        cut = text.rfind(" ", 0, max_chars + 1)
        if cut == -1 or cut < max_chars // 2:
            return max_chars
        return cut + 1
