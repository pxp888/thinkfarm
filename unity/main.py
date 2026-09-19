import os
import sys
import multiprocessing

# ---------------------------------------------------------------------------
# SSL Certificate Configuration (PyInstaller compatibility)
# ---------------------------------------------------------------------------
# In a PyInstaller bundle on Windows, macOS, or different Linux distros, OpenSSL's
# default certificate paths do not exist. Point SSL_CERT_FILE to certifi's bundle.
try:
    import certifi
    if 'SSL_CERT_FILE' not in os.environ or not os.path.exists(os.environ['SSL_CERT_FILE']):
        os.environ['SSL_CERT_FILE'] = certifi.where()
except Exception:
    pass

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
