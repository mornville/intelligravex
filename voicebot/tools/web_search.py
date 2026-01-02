from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any, Iterable, Optional

import httpx
import numpy as np


def web_search_tool_def() -> dict[str, Any]:
    """
    OpenAI Responses API tool schema for web_search.

    The model should call this tool to fetch a web page, create embeddings for the cleaned page text,
    and return top matching chunks for each query.
    """
    return {
        "type": "function",
        "name": "web_search",
        "description": (
            "Fetch and search a web page. Provide a full URL in complete_url and a comma-separated list "
            "of (ideally 9) short search queries in vector_search_queries. The system will scrape the page, "
            "clean it to text, embed chunks, and return top matching chunks per query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "complete_url": {
                    "type": "string",
                    "description": "A complete URL to fetch (include scheme, e.g. https://...).",
                },
                "vector_search_queries": {
                    "type": "string",
                    "description": "Comma-separated queries (prefer 9): q1,q2,q3,...",
                },
                "what_to_search": {
                    "type": "string",
                    "description": "Optional: brief intent for what you are trying to find on the page.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Optional: number of top chunks to return per query.",
                    "default": 5,
                },
            },
            "required": ["complete_url", "vector_search_queries"],
            "additionalProperties": True,
        },
        "strict": False,
    }


class _HTMLToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignore_depth = 0
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        t = (tag or "").lower()
        if t in ("script", "style", "noscript", "svg", "canvas"):
            self._ignore_depth += 1
            return
        if self._ignore_depth:
            return
        if t in ("p", "br", "div", "li", "ul", "ol", "section", "article", "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = (tag or "").lower()
        if t in ("script", "style", "noscript", "svg", "canvas"):
            if self._ignore_depth:
                self._ignore_depth -= 1
            return
        if self._ignore_depth:
            return
        if t in ("p", "div", "li", "section", "article"):
            self._buf.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignore_depth:
            return
        if data:
            self._buf.append(data)

    def text(self) -> str:
        raw = "".join(self._buf)
        # Normalize whitespace but keep paragraph-ish newlines.
        raw = unescape(raw)
        raw = raw.replace("\r", "\n")
        raw = re.sub(r"[ \t\f\v]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _clean_html_to_text(html: str) -> str:
    parser = _HTMLToText()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:
        # Fall back to best-effort stripping of tags.
        s = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\\1>", " ", html or "")
        s = re.sub(r"(?is)<[^>]+>", " ", s)
        s = unescape(s)
        s = re.sub(r"\\s+", " ", s)
        return s.strip()
    return parser.text()


def _chunk_text(
    text: str,
    *,
    chunk_chars: int = 1600,
    overlap_chars: int = 200,
    max_chunks: int = 200,
) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    if chunk_chars <= 0:
        return [t]
    overlap = max(0, min(overlap_chars, chunk_chars - 1))
    out: list[str] = []
    start = 0
    while start < len(t) and len(out) < max_chunks:
        end = min(len(t), start + chunk_chars)
        chunk = t[start:end].strip()
        if chunk:
            out.append(chunk)
        if end >= len(t):
            break
        start = end - overlap
    return out


def _cosine_topk(
    *,
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    if doc_vecs.size == 0:
        return []
    q = query_vec.astype(np.float32)
    d = doc_vecs.astype(np.float32)
    qn = np.linalg.norm(q)
    dn = np.linalg.norm(d, axis=1)
    denom = (dn * (qn if qn != 0 else 1.0)) + 1e-12
    scores = (d @ q) / denom
    k = max(1, min(int(k), int(scores.shape[0])))
    # Argpartition for speed, then exact sort for the shortlist.
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


def _parse_queries(raw: str) -> list[str]:
    s = (raw or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _batched(it: Iterable[str], n: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass(frozen=True)
class WebSearchConfig:
    embedding_model: str = "text-embedding-3-small"
    max_clean_chars: int = 200_000
    chunk_chars: int = 1600
    chunk_overlap_chars: int = 200
    max_chunks: int = 200
    top_k: int = 5
    user_agent: str = "IntelligravexVoiceBot/1.0"
    timeout_s: float = 30.0


def web_search(
    *,
    complete_url: str,
    vector_search_queries: str,
    openai_api_key: str,
    scrapingbee_api_key: str,
    what_to_search: str = "",
    top_k: Optional[int] = None,
    config: Optional[WebSearchConfig] = None,
) -> dict[str, Any]:
    """
    Fetches a page via ScrapingBee, cleans it to text, embeds chunks, and runs vector search.
    Returns a JSON-serializable dict suitable as a tool result.
    """
    cfg = config or WebSearchConfig()
    url = (complete_url or "").strip()
    if not url:
        return {"ok": False, "error": {"message": "complete_url is required"}}
    if not (url.startswith("http://") or url.startswith("https://")):
        return {"ok": False, "error": {"message": "complete_url must start with http:// or https://"}}

    queries = _parse_queries(vector_search_queries)
    if not queries:
        return {"ok": False, "error": {"message": "vector_search_queries must be a comma-separated non-empty list"}}
    queries = queries[:9]

    k = int(top_k if top_k is not None else cfg.top_k)
    k = max(1, min(k, 10))

    if not scrapingbee_api_key.strip():
        return {"ok": False, "error": {"message": "Missing ScrapingBee API key (set SCRAPINGBEE_API_KEY)"}}

    # 1) Fetch via ScrapingBee.
    with httpx.Client(
        timeout=httpx.Timeout(cfg.timeout_s, connect=min(10.0, cfg.timeout_s)),
        follow_redirects=True,
        headers={"User-Agent": cfg.user_agent},
    ) as client:
        resp = client.get(
            "https://app.scrapingbee.com/api/v1/",
            params={
                "api_key": scrapingbee_api_key,
                "url": url,
                "render_js": "false",
            },
        )
    if resp.status_code >= 400:
        return {
            "ok": False,
            "error": {
                "message": f"ScrapingBee returned HTTP {resp.status_code}",
                "status_code": resp.status_code,
            },
        }

    html = resp.text or ""
    if not html.strip():
        return {"ok": False, "error": {"message": "Empty HTML response"}}

    # 2) Clean HTML.
    page_text = _clean_html_to_text(html)
    truncated = False
    if cfg.max_clean_chars and len(page_text) > cfg.max_clean_chars:
        page_text = page_text[: cfg.max_clean_chars]
        truncated = True

    chunks = _chunk_text(
        page_text,
        chunk_chars=cfg.chunk_chars,
        overlap_chars=cfg.chunk_overlap_chars,
        max_chunks=cfg.max_chunks,
    )
    if not chunks:
        return {"ok": False, "error": {"message": "No text content found after cleaning/chunking"}}

    # 3) Embeddings.
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": {"message": f"OpenAI SDK not installed: {exc}"}}

    oai = OpenAI(api_key=openai_api_key)

    chunk_embeddings: list[list[float]] = []
    for batch in _batched(chunks, 96):
        emb = oai.embeddings.create(model=cfg.embedding_model, input=batch)
        for item in getattr(emb, "data", []) or []:
            vec = getattr(item, "embedding", None)
            if isinstance(vec, list) and vec:
                chunk_embeddings.append(vec)

    if len(chunk_embeddings) != len(chunks):
        return {
            "ok": False,
            "error": {
                "message": "Embedding count mismatch",
                "chunks": len(chunks),
                "embeddings": len(chunk_embeddings),
            },
        }

    doc_vecs = np.asarray(chunk_embeddings, dtype=np.float32)

    results: list[dict[str, Any]] = []
    for q in queries:
        q_emb = oai.embeddings.create(model=cfg.embedding_model, input=q)
        q_vec_raw = None
        data = getattr(q_emb, "data", []) or []
        if data:
            q_vec_raw = getattr(data[0], "embedding", None)
        if not isinstance(q_vec_raw, list) or not q_vec_raw:
            continue
        q_vec = np.asarray(q_vec_raw, dtype=np.float32)
        top = _cosine_topk(query_vec=q_vec, doc_vecs=doc_vecs, k=k)
        matches: list[dict[str, Any]] = []
        for idx, score in top:
            chunk_text = chunks[idx]
            matches.append(
                {
                    "chunk_index": idx,
                    "score": score,
                    "text": (chunk_text[:900] + ("…" if len(chunk_text) > 900 else "")),
                }
            )
        results.append({"query": q, "matches": matches})

    return {
        "ok": True,
        "url": url,
        "what_to_search": (what_to_search or "").strip(),
        "queries": queries,
        "top_k": k,
        "page": {
            "chars_total": len(page_text),
            "truncated": truncated,
        },
        "chunks": {
            "count": len(chunks),
            "chunk_chars": cfg.chunk_chars,
            "overlap_chars": cfg.chunk_overlap_chars,
        },
        "results": results,
    }

