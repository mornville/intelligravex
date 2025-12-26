from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from voicebot.llm.openai_llm import Message


def _try_get_tiktoken():
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None
    return tiktoken


def estimate_text_tokens(text: str, model: str) -> int:
    s = (text or "").strip()
    if not s:
        return 0

    tiktoken = _try_get_tiktoken()
    if tiktoken is None:
        # Rough fallback: ~4 chars/token in English. Clamp to at least 1 token.
        return max(1, int(round(len(s) / 4.0)))

    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return int(len(enc.encode(s)))


def estimate_messages_tokens(messages: Iterable[Message], model: str) -> int:
    # This is an estimate; Responses API has its own internal formatting overhead.
    # Add a small per-message overhead (role + wrapper).
    per_msg_overhead = 4
    total = 0
    for m in messages:
        total += per_msg_overhead
        total += estimate_text_tokens(m.content or "", model)
    return int(total)


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float
    output_per_1m: float


def estimate_cost_usd(*, model_price: Optional[ModelPrice], input_tokens: int, output_tokens: int) -> float:
    if model_price is None:
        return 0.0
    return float(input_tokens) / 1_000_000.0 * model_price.input_per_1m + float(output_tokens) / 1_000_000.0 * model_price.output_per_1m

