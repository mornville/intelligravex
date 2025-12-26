from voicebot.utils.text import SentenceChunker


def test_sentence_chunker_splits_on_sentence_end() -> None:
    c = SentenceChunker()
    chunks = []
    chunks.extend(list(c.push("Hello world.")))
    chunks.extend(list(c.push(" How are you?")))
    tail = c.flush()
    if tail:
        chunks.append(tail)
    assert chunks == ["Hello world.", "How are you?"]


def test_sentence_chunker_flushes_remainder() -> None:
    c = SentenceChunker()
    chunks = list(c.push("Hello"))
    assert chunks == []
    assert c.flush() == "Hello"

