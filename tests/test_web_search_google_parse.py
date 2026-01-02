from __future__ import annotations

from voicebot.tools.web_search import _google_search_via_scrapingbee


def test_google_api_parser_smoke(monkeypatch):
    # Patch httpx.Client.get used inside _google_search_via_scrapingbee.
    import httpx

    class DummyResp:
        status_code = 200

        def __init__(self, obj):
            self._obj = obj
            self.text = "ok"

        def json(self):
            return self._obj

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None):
            assert "store/google" in url
            assert params["search"] == "candor health"
            return DummyResp(
                {
                    "organic_results": [
                        {"title": "A", "url": "https://example.com/a", "description": "da"},
                        {"title": "B", "url": "https://example.com/b", "description": "db"},
                    ]
                }
            )

    monkeypatch.setattr(httpx, "Client", DummyClient)
    results, err = _google_search_via_scrapingbee(
        scrapingbee_api_key="k",
        search_term="candor health",
        timeout_s=10.0,
        user_agent="ua",
        max_results=10,
    )
    assert err == {}
    assert len(results) == 2
    assert results[0]["url"] == "https://example.com/a"

