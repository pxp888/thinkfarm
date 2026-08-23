import os
import sys
import multiprocessing

# ---------------------------------------------------------------------------
# PyInstaller & Windows Console Redirection
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    # When packaged with console=False (windowed mode), stdout/stderr/stdin are None.
    # Redirect them to prevent crashes in uvicorn, logging, and other libraries.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    if sys.stdin is None:
        sys.stdin = open(os.devnull, "r", encoding="utf-8")

# Required for PyInstaller + multiprocessing support
multiprocessing.freeze_support()

from PyQt6.QtWidgets import QApplication
from app_gui import ThinkfarmApp

def main():
    app = QApplication(sys.argv)
    window = ThinkfarmApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
