from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken


class CryptoError(RuntimeError):
    pass


def mask_secret(value: str, *, show_start: int = 6, show_end: int = 4) -> str:
    v = value.strip()
    if len(v) <= show_start + show_end:
        return "*" * len(v)
    return f"{v[:show_start]}{'*' * 8}{v[-show_end:]}"


def build_hint(value: str) -> str:
    v = value.strip()
    if not v:
        return ""
    return mask_secret(v)


@dataclass(frozen=True)
class CryptoBox:
    fernet: Fernet

    def encrypt_str(self, value: str) -> bytes:
        return self.fernet.encrypt(value.encode("utf-8"))

    def decrypt_str(self, token: bytes) -> str:
        try:
            return self.fernet.decrypt(token).decode("utf-8")
        except InvalidToken as exc:
            raise CryptoError("Failed to decrypt secret (wrong VOICEBOT_SECRET_KEY?)") from exc


def get_crypto_box(secret_key: Optional[str]) -> CryptoBox:
    if not secret_key:
        raise CryptoError(
            "VOICEBOT_SECRET_KEY is required to store/use encrypted API keys. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return CryptoBox(fernet=Fernet(secret_key.encode("utf-8")))

