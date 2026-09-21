#!/usr/bin/env bash
# One-command start for the self-contained thinkfarm provider app.
# Requires: Linux x86_64 with an nvidia driver (no CUDA toolkit, no Ollama needed), python >= 3.10.
# Usage: ./thinkfarm.sh [--model NAME]             PyQt6 dashboard GUI (close minimizes to tray; Exit quits)
#        ./thinkfarm.sh --headless [--model NAME]  headless daemon (Ctrl+C to stop)
#        ./thinkfarm.sh --download [--model NAME]  download required model weights from the CLI
#        --model picks the bundled model to load (app.py MODELS; default qwen3.8-27b)
set -euo pipefail
cd "$(dirname "$0")"

MODE=gui
if [ "${1:-}" = "--headless" ]; then MODE=headless; shift; fi

# CLI download runs directly in terminal mode
for arg in "$@"; do
    if [ "$arg" = "--download" ]; then
        MODE=headless
        break
    fi
done

PKG_LIST="httpx websockets"
IMPORT_TEST='import httpx, websockets'
if [ "$MODE" = gui ]; then
    PKG_LIST="$PKG_LIST PyQt6"
    IMPORT_TEST="$IMPORT_TEST, PyQt6"
fi

# Find a python with the runtime deps; create a venv once if none.
have_deps() { "$1" -c "$IMPORT_TEST" >/dev/null 2>&1; }

if [ -x venv/bin/python ] && have_deps venv/bin/python; then
    PY=./venv/bin/python
elif have_deps python3; then
    PY=python3
else
    echo "[run] missing deps — creating ./venv and installing $PKG_LIST (one-time)..."
    python3 -m venv venv
    if [ -d wheelhouse ] && ls wheelhouse/*.whl >/dev/null 2>&1; then
        # Offline: install from the bundled wheels (covers CPython 3.10-3.15);
        # fall back to network if this interpreter isn't covered.
        ./venv/bin/pip install --quiet --no-index --find-links wheelhouse $PKG_LIST \
            || { echo "[run] bundled wheels don't cover this Python (need 3.10-3.15) — installing from network..."
                 ./venv/bin/pip install --quiet $PKG_LIST; }
    else
        ./venv/bin/pip install --quiet $PKG_LIST
    fi
    PY=./venv/bin/python
fi

if [ "$MODE" = gui ]; then
    exec "$PY" gui.py "$@"
fi
exec "$PY" app.py "$@"
