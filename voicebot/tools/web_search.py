from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any, Iterable, Optional
from typing import Callable

import httpx
import numpy as np


def web_search_tool_def() -> dict[str, Any]:
    """
    OpenAI Responses API tool schema for web_search.

    High-level behavior:
    1) Use `search_term` to fetch Google results via ScrapingBee
    2) Ask an LLM (configured model) to pick 1-4 results (guided by `why`)
    3) For each selected page, scrape+clean+chunk, embed, and run vector search using `vector_search_queries`
    4) Ask the same LLM to produce a final summary using the extracted snippets (guided by `why`)
    """
    return {
        "type": "function",
        "name": "web_search",
        "description": (
            "Search the web for a goal. Provide `search_term` (Google query), a comma-separated list of "
            "(ideally 9) short `vector_search_queries`, and `why` (what you need and how you will use the info). "
            "The system will Google the term, have an LLM select 1-4 relevant results, scrape those pages, run "
            "vector search on the page content, and return a final summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {"type": "string", "description": "Google search term/query."},
                "vector_search_queries": {
                    "type": "string",
                    "description": "Comma-separated queries (prefer 9): q1,q2,q3,...",
                },
                "why": {"type": "string", "description": "Why you are searching (selection + summary guidance)."},
                "wait_reply": {
                    "type": "string",
                    "description": "Short filler message to say while searching (e.g. 'Got it—looking that up now.').",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Optional: number of top chunks to return per query per page.",
                    "default": 5,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Optional: max Google results to consider before LLM filtering.",
                    "default": 10,
                },
            },
            "required": ["search_term", "vector_search_queries", "why"],
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


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    s = (text or "").strip()
    if not s:
        return None
    # Best-effort: find the first {...} block.
    m = re.search(r"\\{.*\\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _to_responses_input(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role") or "user"
        text = m.get("content") or ""
        if not text.strip():
            continue
        items.append({"role": role, "content": [{"type": "input_text", "text": text}]})
    return items


def _llm_complete_text(*, api_key: str, model: str, messages: list[dict[str, str]]) -> str:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)
    resp = client.responses.create(model=model, input=_to_responses_input(messages))
    return (getattr(resp, "output_text", "") or "").strip()


def _scrape_via_scrapingbee(
    *,
    scrapingbee_api_key: str,
    url: str,
    timeout_s: float,
    user_agent: str,
    extra_params: Optional[dict[str, str]] = None,
) -> tuple[Optional[str], dict[str, Any]]:
    with httpx.Client(
        timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
        follow_redirects=True,
        headers={"User-Agent": user_agent},
    ) as client:
        resp = client.get(
            "https://app.scrapingbee.com/api/v1/",
            params={
                "api_key": scrapingbee_api_key,
                "url": url,
                "render_js": "false",
                **(extra_params or {}),
            },
        )
    if resp.status_code >= 400:
        return None, {
            "message": f"ScrapingBee returned HTTP {resp.status_code}",
            "status_code": resp.status_code,
            "body": (resp.text or "")[:2000],
        }
    return (resp.text or ""), {}


def _google_search_url(search_term: str, *, num: int = 10) -> str:
    q = httpx.QueryParams({"q": search_term, "num": str(max(1, min(num, 20))), "hl": "en"})
    return f"https://www.google.com/search?{q}"


def _parse_google_results(html: str, *, max_results: int) -> list[dict[str, str]]:
    # Best-effort parsing that works on common /url?q=... links.
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    h = html or ""
    for m in re.finditer(r'href=\"/url\\?q=([^&\\\"]+)', h):
        raw = unescape(m.group(1))
        try:
            url = httpx.URL(raw).human_repr()
        except Exception:
            continue
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        if "google.com" in url:
            continue
        if url in seen:
            continue
        seen.add(url)
        # Try to capture title/snippet close to the link.
        window = h[m.end() : m.end() + 2500]
        title = ""
        snippet = ""
        mt = re.search(r"(?is)<h3[^>]*>(.*?)</h3>", window)
        if mt:
            title = _clean_html_to_text(mt.group(1))
        ms = re.search(r"(?is)<div[^>]+class=\\\"VwiC3b[^\\\"]*\\\"[^>]*>(.*?)</div>", window)
        if ms:
            snippet = _clean_html_to_text(ms.group(1))
        out.append({"url": url, "title": title.strip(), "snippet": snippet.strip()})
        if len(out) >= max_results:
            break
    return out


def _google_search_via_scrapingbee(
    *,
    scrapingbee_api_key: str,
    search_term: str,
    timeout_s: float,
    user_agent: str,
    max_results: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """
    Uses ScrapingBee's Google Search API, which returns structured JSON results and
    avoids scraping the Google HTML SERP directly (more reliable).
    """
    with httpx.Client(
        timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
        follow_redirects=True,
        headers={"User-Agent": user_agent},
    ) as client:
        resp = client.get(
            "https://app.scrapingbee.com/api/v1/store/google",
            params={
                "api_key": scrapingbee_api_key,
                "search": search_term,
                "language": "en",
                "page": "1",
            },
        )
    if resp.status_code >= 400:
        return [], {
            "message": f"ScrapingBee Google API returned HTTP {resp.status_code}",
            "status_code": resp.status_code,
            "body": (resp.text or "")[:2000],
        }
    try:
        obj = resp.json()
    except Exception:
        return [], {
            "message": "ScrapingBee Google API returned non-JSON response",
            "status_code": resp.status_code,
            "body": (resp.text or "")[:2000],
        }
    if not isinstance(obj, dict):
        return [], {"message": "Unexpected Google API response shape"}

    organic = obj.get("organic_results")
    if not isinstance(organic, list):
        organic = []
    out: list[dict[str, str]] = []
    for it in organic:
        if not isinstance(it, dict):
            continue
        url = str(it.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        title = str(it.get("title") or "").strip()
        snippet = str(it.get("description") or it.get("snippet") or "").strip()
        out.append({"url": url, "title": title, "snippet": snippet})
        if len(out) >= max_results:
            break
    return out, {}


def web_search(
    *,
    search_term: str,
    vector_search_queries: str,
    why: str,
    openai_api_key: str,
    scrapingbee_api_key: str,
    model: str,
    progress_fn: Optional[Callable[[str], None]] = None,
    top_k: Optional[int] = None,
    max_results: Optional[int] = None,
    config: Optional[WebSearchConfig] = None,
) -> str:
    """
    Fetches a page via ScrapingBee, cleans it to text, embeds chunks, and runs vector search.
    Returns the final summary text.
    """
    cfg = config or WebSearchConfig()
    st = (search_term or "").strip()
    if not st:
        raise ValueError("search_term is required")
    why_text = (why or "").strip()
    if not why_text:
        raise ValueError("why is required")

    queries = _parse_queries(vector_search_queries)
    if not queries:
        raise ValueError("vector_search_queries must be a comma-separated non-empty list")
    queries = queries[:9]

    k = int(top_k if top_k is not None else cfg.top_k)
    k = max(1, min(k, 10))
    max_r = int(max_results if max_results is not None else 10)
    max_r = max(3, min(max_r, 20))

    if not scrapingbee_api_key.strip():
        raise RuntimeError("Missing ScrapingBee API key (set SCRAPINGBEE_API_KEY)")
    if not openai_api_key.strip():
        raise RuntimeError("Missing OpenAI API key")
    chosen_model = (model or "").strip()
    if not chosen_model:
        raise RuntimeError("Missing model for filtering/summarization")

    if progress_fn:
        try:
            progress_fn("Searching Google…")
        except Exception:
            pass

    # 1) Google search via ScrapingBee (structured API).
    candidates, google_err = _google_search_via_scrapingbee(
        scrapingbee_api_key=scrapingbee_api_key,
        search_term=st,
        timeout_s=cfg.timeout_s,
        user_agent=cfg.user_agent,
        max_results=max_r,
    )
    google_url = ""
    if google_err:
        # Fallback: scrape Google HTML (less reliable; may be blocked without special flags).
        google_url = _google_search_url(st, num=max_r)
        google_html, html_err = _scrape_via_scrapingbee(
            scrapingbee_api_key=scrapingbee_api_key,
            url=google_url,
            timeout_s=cfg.timeout_s,
            user_agent=cfg.user_agent,
            extra_params={"custom_google": "true"},
        )
        if not google_html:
            raise RuntimeError(f"Google search failed: {google_err}; fallback: {html_err}")
        candidates = _parse_google_results(google_html, max_results=max_r)
    if not candidates:
        raise RuntimeError("No Google results parsed")

    if progress_fn:
        try:
            progress_fn("Selecting best sources…")
        except Exception:
            pass

    # 2) LLM selects 1-4 results based on `why`.
    filter_prompt = (
        "You are filtering Google results.\n"
        "Pick the most relevant results for the user's goal.\n"
        "Constraints:\n"
        "- Select at least 1 and at most 4 urls\n"
        "- Prefer authoritative / primary sources when possible\n"
        "- Return ONLY valid JSON in this shape: {\"selected_urls\":[\"https://...\", ...], \"reason\":\"...\"}\n\n"
        f"WHY: {why_text}\n"
        f"SEARCH_TERM: {st}\n"
        f"CANDIDATES_JSON: {json.dumps(candidates, ensure_ascii=False)}"
    )
    raw_filter = _llm_complete_text(
        api_key=openai_api_key,
        model=chosen_model,
        messages=[{"role": "system", "content": "Return only JSON."}, {"role": "user", "content": filter_prompt}],
    )
    sel_obj = _extract_json_object(raw_filter) or {}
    selected_urls = sel_obj.get("selected_urls")
    if not isinstance(selected_urls, list):
        selected_urls = []
    picked: list[str] = []
    for u in selected_urls:
        if isinstance(u, str) and (u.startswith("http://") or u.startswith("https://")):
            picked.append(u.strip())
    if not picked:
        picked = [candidates[0]["url"]]
    picked = picked[:4]

    if progress_fn:
        try:
            progress_fn(f"Fetching {len(picked)} page(s)…")
        except Exception:
            pass

    # 3) Embeddings.
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"OpenAI SDK not installed: {exc}") from exc

    oai = OpenAI(api_key=openai_api_key)

    per_page: list[dict[str, Any]] = []
    for i, page_url in enumerate(picked, start=1):
        if progress_fn:
            try:
                progress_fn(f"Reading page {i}/{len(picked)}…")
            except Exception:
                pass
        html, err = _scrape_via_scrapingbee(
            scrapingbee_api_key=scrapingbee_api_key,
            url=page_url,
            timeout_s=cfg.timeout_s,
            user_agent=cfg.user_agent,
        )
        if not html:
            per_page.append({"url": page_url, "ok": False, "error": err})
            continue
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
            per_page.append({"url": page_url, "ok": False, "error": {"message": "No text after cleaning"}})
            continue

        chunk_embeddings: list[list[float]] = []
        for batch in _batched(chunks, 96):
            emb = oai.embeddings.create(model=cfg.embedding_model, input=batch)
            for item in getattr(emb, "data", []) or []:
                vec = getattr(item, "embedding", None)
                if isinstance(vec, list) and vec:
                    chunk_embeddings.append(vec)

        if len(chunk_embeddings) != len(chunks):
            per_page.append(
                {
                    "url": page_url,
                    "ok": False,
                    "error": {
                        "message": "Embedding count mismatch",
                        "chunks": len(chunks),
                        "embeddings": len(chunk_embeddings),
                    },
                }
            )
            continue

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

        per_page.append(
            {
                "url": page_url,
                "ok": True,
                "page": {"chars_total": len(page_text), "truncated": truncated},
                "chunks": {"count": len(chunks)},
                "results": results,
            }
        )

    # 4) Summarize using the same model (guided by `why`).
    if progress_fn:
        try:
            progress_fn("Summarizing…")
        except Exception:
            pass
    summarize_prompt = (
        "You are summarizing web research results.\n"
        "Write a concise answer that directly addresses WHY.\n"
        "Only use the provided snippets; if something is uncertain, say so.\n"
        "Include source URLs inline when making claims.\n\n"
        f"WHY: {why_text}\n"
        f"SEARCH_TERM: {st}\n"
        f"VECTOR_SEARCH_QUERIES: {', '.join(queries)}\n"
        f"PAGE_RESULTS_JSON: {json.dumps(per_page, ensure_ascii=False)}"
    )
    summary = _llm_complete_text(
        api_key=openai_api_key,
        model=chosen_model,
        messages=[{"role": "system", "content": "Be concise and cite sources."}, {"role": "user", "content": summarize_prompt}],
    )

    return summary.strip()
