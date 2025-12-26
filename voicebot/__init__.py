__all__ = ["__version__"]

__version__ = "0.1.0"

# Apply small runtime shims early (import-time) to avoid crashes in downstream libs.
try:
    from voicebot.compat.torch_mps import ensure_torch_mps_compat

    ensure_torch_mps_compat()
except Exception:
    pass
