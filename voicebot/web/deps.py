from __future__ import annotations

from typing import Generator

from fastapi import Depends, HTTPException, Request
from sqlmodel import Session

from voicebot.crypto import CryptoError, get_crypto_box

from .state import AppState


def get_state(request: Request) -> AppState:
    state = getattr(request.app.state, "igx_state", None)
    if state is None:
        raise RuntimeError("App state not initialized")
    return state


def get_session(state: AppState = Depends(get_state)) -> Generator[Session, None, None]:
    with Session(state.engine) as s:
        yield s


def require_crypto(state: AppState = Depends(get_state)):
    try:
        return get_crypto_box(state.settings.secret_key)
    except CryptoError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
