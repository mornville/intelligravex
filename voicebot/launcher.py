from __future__ import annotations

import os
import signal
import socket
import platform
import shutil
import subprocess
import threading
import time
import webbrowser
from typing import Iterable, Optional


def _port_available(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def _find_pids_on_port(port: int) -> list[int]:
    cmds = [
        ["lsof", "-ti", f"tcp:{port}"],
        ["lsof", "-ti", f":{port}"],
    ]
    for cmd in cmds:
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        except FileNotFoundError:
            continue
        if p.returncode != 0:
            continue
        out = (p.stdout or "").strip()
        if not out:
            continue
        pids: list[int] = []
        for line in out.splitlines():
            try:
                pids.append(int(line.strip()))
            except Exception:
                continue
        if pids:
            return pids
    return []


def _terminate_pids(pids: Iterable[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            continue


def _kill_pids(pids: Iterable[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            continue


def _stop_processes_on_port(host: str, port: int) -> bool:
    pids = _find_pids_on_port(port)
    if not pids:
        return False
    _terminate_pids(pids)
    time.sleep(0.6)
    if _port_available(host, port):
        return True
    _kill_pids(pids)
    time.sleep(0.6)
    return _port_available(host, port)


def _ask_port_gui(port: int) -> Optional[int]:
    try:
        import tkinter as tk
        from tkinter import messagebox, simpledialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    msg = (
        f"Port {port} is already in use.\n\n"
        "Yes: stop the process on this port and reuse it.\n"
        "No: choose a different port.\n"
        "Cancel: exit."
    )
    resp = messagebox.askyesnocancel("Port in use", msg)
    if resp is None:
        return 0
    if resp is True:
        return port
    new_port = simpledialog.askinteger(
        "Choose port",
        "Enter a port (1024-65535):",
        initialvalue=port + 1,
        minvalue=1024,
        maxvalue=65535,
    )
    if not new_port:
        return 0
    return int(new_port)


def _ask_initial_port_gui(default_port: int) -> Optional[int]:
    try:
        import tkinter as tk
        from tkinter import simpledialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    new_port = simpledialog.askinteger(
        "Choose port",
        "Enter a port (1024-65535):",
        initialvalue=default_port,
        minvalue=1024,
        maxvalue=65535,
    )
    if not new_port:
        return 0
    return int(new_port)


def _ask_port_osascript(port: int) -> Optional[int]:
    if platform.system() != "Darwin":
        return None
    if not shutil.which("osascript"):
        return None
    msg = (
        f"Port {port} is already in use.\n\n"
        "Stop and reuse: stop the process on this port.\n"
        "Choose Port: pick a different port.\n"
        "Cancel: exit."
    )
    try:
        p = subprocess.run(
            [
                "osascript",
                "-e",
                'button returned of (display dialog "{}" buttons {{"Cancel","Choose Port","Stop and reuse"}} default button "Stop and reuse")'.format(
                    msg.replace('"', '\\"')
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return None
    choice = (p.stdout or "").strip()
    if choice == "Stop and reuse":
        return port
    if choice == "Choose Port":
        try:
            p2 = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'text returned of (display dialog "Enter a port (1024-65535):" default answer "{}")'.format(
                        port + 1
                    ),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except Exception:
            return None
        txt = (p2.stdout or "").strip()
        try:
            val = int(txt)
        except Exception:
            return 0
        return val
    return 0


def _ask_initial_port_osascript(default_port: int) -> Optional[int]:
    if platform.system() != "Darwin":
        return None
    if not shutil.which("osascript"):
        return None
    try:
        p = subprocess.run(
            [
                "osascript",
                "-e",
                'text returned of (display dialog "Enter a port (1024-65535):" default answer "{}")'.format(
                    default_port
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if p.returncode != 0:
        return 0
    txt = (p.stdout or "").strip()
    try:
        return int(txt)
    except Exception:
        return 0


def _ask_port_zenity(port: int) -> Optional[int]:
    if platform.system() != "Linux":
        return None
    if not shutil.which("zenity"):
        return None
    msg = (
        f"Port {port} is already in use.\n\n"
        "Stop and reuse: stop the process on this port.\n"
        "Choose Port: pick a different port.\n"
        "Cancel: exit."
    )
    p = subprocess.run(
        ["zenity", "--question", "--text", msg, "--ok-label", "Stop and reuse", "--cancel-label", "Choose Port"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if p.returncode == 0:
        return port
    if p.returncode == 1:
        p2 = subprocess.run(
            ["zenity", "--entry", "--text", "Enter a port (1024-65535):", "--entry-text", str(port + 1)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if p2.returncode != 0:
            return 0
        txt = (p2.stdout or "").strip()
        try:
            return int(txt)
        except Exception:
            return 0
    return 0


def _ask_initial_port_zenity(default_port: int) -> Optional[int]:
    if platform.system() != "Linux":
        return None
    if not shutil.which("zenity"):
        return None
    p = subprocess.run(
        ["zenity", "--entry", "--text", "Enter a port (1024-65535):", "--entry-text", str(default_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if p.returncode != 0:
        return 0
    txt = (p.stdout or "").strip()
    try:
        return int(txt)
    except Exception:
        return 0


def _select_port(host: str, port: int) -> int:
    initial = _ask_initial_port_gui(port)
    if initial is None:
        initial = _ask_initial_port_osascript(port)
    if initial is None:
        initial = _ask_initial_port_zenity(port)
    if initial == 0:
        raise SystemExit(0)
    if initial is not None:
        port = int(initial)
    if port < 1024 or port > 65535:
        port = 8000
    if _port_available(host, port):
        return port
    while True:
        choice = _ask_port_gui(port)
        if choice is None:
            choice = _ask_port_osascript(port)
        if choice is None:
            choice = _ask_port_zenity(port)
        if choice is None:
            # No GUI available; fall back to auto-pick.
            for candidate in range(port + 1, port + 50):
                if _port_available(host, candidate):
                    return candidate
            return port
        if choice == 0:
            raise SystemExit(0)
        if choice == port:
            if _stop_processes_on_port(host, port):
                return port
            continue
        if _port_available(host, choice):
            return choice


def launch(host: str = "127.0.0.1", port: int = 8000, *, open_browser: bool = True) -> None:
    """
    Launch the Studio server and optionally open the browser.

    Intended for packaged desktop wrappers (macOS/Linux).
    """
    port = _select_port(host, port)

    if open_browser:
        url = f"http://{host}:{port}"

        def _open() -> None:
            time.sleep(1.2)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing web dependencies. Install: pip install -e '.[web]'") from exc

    uvicorn.run("voicebot.web.app:create_app", host=host, port=port, reload=False, factory=True)


def main() -> None:
    launch()


if __name__ == "__main__":
    main()
