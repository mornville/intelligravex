from __future__ import annotations


def ensure_torch_mps_compat() -> None:
    """
    PyTorch's MPS backend is exposed via `torch.backends.mps`, but some packages assume
    CUDA-like helpers exist under `torch.mps` (e.g. `torch.mps.current_device()`).

    On some PyTorch builds (e.g. 2.8.0 on macOS), these helpers are missing. This shim
    defines them when possible to prevent runtime crashes in downstream libraries.
    """
    try:
        import torch
    except Exception:
        return

    mps = getattr(torch, "mps", None)
    if mps is None:
        return

    if not hasattr(mps, "current_device"):
        setattr(mps, "current_device", lambda: 0)
    if not hasattr(mps, "device_count"):
        setattr(mps, "device_count", lambda: 1)

