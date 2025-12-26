from __future__ import annotations

from typing import List, Optional, Union


def parse_device(value: Optional[str]) -> Optional[Union[int, str]]:
    """
    sounddevice `device` can be an index (int) or substring name (str).
    Allow users to pass either via env/CLI.
    """
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    if v.isdigit():
        return int(v)
    return v


def list_audio_devices() -> List[str]:
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover
        return [f"sounddevice unavailable: {exc}"]

    devices = sd.query_devices()
    lines: List[str] = []
    for idx, dev in enumerate(devices):
        name = dev.get("name", "unknown")
        hostapi = dev.get("hostapi", None)
        sr = dev.get("default_samplerate", None)
        ins = dev.get("max_input_channels", 0)
        outs = dev.get("max_output_channels", 0)
        lines.append(f"[{idx}] {name} (in={ins}, out={outs}, sr={sr}, hostapi={hostapi})")
    return lines
