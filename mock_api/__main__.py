from __future__ import annotations

import os


def main() -> None:
    try:
        import uvicorn  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit("uvicorn is required: pip install uvicorn") from exc

    host = os.environ.get("MOCK_API_HOST", "127.0.0.1")
    port = int(os.environ.get("MOCK_API_PORT", "9001"))
    uvicorn.run("mock_api.app:app", host=host, port=port, reload=True, log_level="info")


if __name__ == "__main__":
    main()

