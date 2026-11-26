#!/usr/bin/env python3
"""Unified entry point for the single-file thinkfarm provider executable.

Routes CLI flags (--headless, --download, --download-runtime) to app.main(),
and defaults to launching the PyQt6 dashboard GUI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
if str(ROOT / "lcpp") not in sys.path:
    sys.path.insert(0, str(ROOT / "lcpp"))

if getattr(sys, "frozen", False):
    try:
        import certifi
        ca_bundle = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
    except Exception:
        pass


def _attach_windows_console() -> None:
    """Attach to the parent console when running CLI flags from a windowed Windows exe."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return
    try:
        import ctypes
        ATTACH_PARENT_PROCESS = -1
        if ctypes.windll.kernel32.AttachConsole(ATTACH_PARENT_PROCESS):
            sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)
            sys.stderr = open("CONOUT$", "w", encoding="utf-8", buffering=1)
    except Exception:
        pass


def main() -> None:
    args = sys.argv[1:]
    cli_flags = {"--headless", "--download", "--download-runtime"}
    if any(a in cli_flags for a in args):
        _attach_windows_console()
        if "--headless" in sys.argv:
            sys.argv = [a for a in sys.argv if a != "--headless"]
        import app
        app.main()
    else:
        import gui
        gui.main()


if __name__ == "__main__":
    main()

