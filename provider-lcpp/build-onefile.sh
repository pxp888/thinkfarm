#!/usr/bin/env bash
# Build single-file thinkfarm-provider executable for Linux using PyInstaller.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ./venv/bin/pyinstaller ]; then
    echo "[build-onefile] Installing pyinstaller into ./venv..."
    ./venv/bin/pip install --quiet pyinstaller
fi

./venv/bin/pyinstaller --clean --noconfirm thinkfarm-provider.spec

echo
echo "[build-onefile] Built single-file executable:"
ls -lh dist/thinkfarm-provider

