#!/usr/bin/env bash
# Build distributable standalone packages for thinkfarm provider.
# Targets: linux-cuda, linux-vulkan, win-cuda, win-vulkan, all
# Usage: ./build.sh [target] [dist-dir]
set -euo pipefail
cd "$(dirname "$0")"

TARGET="${1:-all}"
DIST_DIR="${2:-dist}"

# Handle case where 1st argument is a dist directory
if [[ "$TARGET" != "all" && "$TARGET" != "linux-cuda" && "$TARGET" != "linux-vulkan" && "$TARGET" != "win-cuda" && "$TARGET" != "win-vulkan" ]]; then
    if [ -d "$TARGET" ] || [ "$TARGET" = "dist" ]; then
        DIST_DIR="$TARGET"
        TARGET="all"
    else
        echo "Usage: $0 [linux-cuda | linux-vulkan | win-cuda | win-vulkan | all] [dist-dir]" >&2
        exit 1
    fi
fi

# Locate pip
if [ -x ./venv/bin/pip ]; then
    PIP=./venv/bin/pip
elif command -v pip3 &>/dev/null; then
    PIP=pip3
elif command -v pip &>/dev/null; then
    PIP=pip
else
    echo "[build] ERROR: pip not found" >&2
    exit 1
fi

UPSTREAM="upstream-bins"
WHEEL_CACHE_ROOT=".wheel-cache"
VERSION="b11125"
PROVIDER_VERSION=$(sed -n -E "s/^PROVIDER_VERSION[[:space:]]*=[[:space:]]*['\"]?([^ '\"#]+).*/\1/p" gui.py | head -n 1)
if [ -z "$PROVIDER_VERSION" ]; then
    echo "[build] ERROR: Could not extract PROVIDER_VERSION from gui.py" >&2
    exit 1
fi
echo "[build] Provider version: $PROVIDER_VERSION (upstream: $VERSION)"
DIST_DIR="$(realpath -m "$DIST_DIR")"
mkdir -p "$DIST_DIR"

prepare_wheelhouse() {
    local os="$1"       # "linux" or "win"
    local dest="$2"     # target wheelhouse directory
    local cache="$WHEEL_CACHE_ROOT/$os"
    mkdir -p "$cache" "$dest"

    echo "[build] Checking offline wheels for $os..."
    local pvs="310 311 312 313 314 315"

    if [ "$os" = "linux" ]; then
        # Check if already cached
        local count
        count=$(find "$cache" -maxdepth 1 -name "*.whl" 2>/dev/null | wc -l)
        if [ "$count" -lt 10 ]; then
            echo "[build] Downloading Linux wheels into $cache..."
            for PV in $pvs; do
                "$PIP" download httpx websockets \
                    --only-binary=:all: \
                    --python-version "$PV" \
                    --implementation cp \
                    --abi "cp${PV}" \
                    -d "$cache" > /dev/null || true
                "$PIP" download PyQt6-sip --no-deps \
                    --only-binary=:all: \
                    --python-version "$PV" \
                    --implementation cp \
                    --abi "cp${PV}" \
                    -d "$cache" > /dev/null || true
            done
            for PKG in PyQt6-Qt6 PyQt6; do
                "$PIP" download "$PKG" --no-deps \
                    --only-binary=:all: \
                    -d "$cache" > /dev/null || true
            done
        fi
    else
        # Windows wheels
        local count
        count=$(find "$cache" -maxdepth 1 -name "*.whl" 2>/dev/null | wc -l)
        if [ "$count" -lt 10 ]; then
            echo "[build] Downloading Windows wheels into $cache..."
            for PV in $pvs; do
                "$PIP" download httpx websockets \
                    --platform win_amd64 \
                    --only-binary=:all: \
                    --python-version "$PV" \
                    --implementation cp \
                    --abi "cp${PV}" \
                    -d "$cache" > /dev/null || true
                "$PIP" download PyQt6-sip --no-deps \
                    --platform win_amd64 \
                    --only-binary=:all: \
                    --python-version "$PV" \
                    --implementation cp \
                    --abi "cp${PV}" \
                    -d "$cache" > /dev/null || true
            done
            for PKG in PyQt6-Qt6 PyQt6; do
                "$PIP" download "$PKG" --no-deps \
                    --platform win_amd64 \
                    --only-binary=:all: \
                    -d "$cache" > /dev/null || true
            done
        fi
    fi

    cp "$cache"/*.whl "$dest/"
    echo "[build] Wheelhouse populated ($(ls "$dest"/*.whl 2>/dev/null | wc -l) wheels)."
}

build_variant() {
    local variant="$1"  # linux-cuda, linux-vulkan, win-cuda, win-vulkan
    echo "============================================================"
    echo "[build] Building variant: $variant"
    echo "============================================================"

    local os="linux"
    local backend="cuda"
    [[ "$variant" =~ ^win- ]] && os="win"
    [[ "$variant" =~ -vulkan$ ]] && backend="vulkan"

    local bundle="thinkfarm-provider-${variant}-${VERSION}-${PROVIDER_VERSION}"
    local stage="$DIST_DIR/stage/$bundle"
    rm -rf "$stage"
    mkdir -p "$stage/bin"

    # 1. Unpack upstream binaries
    local src_dir="$UPSTREAM/$variant"
    if [ ! -d "$src_dir" ]; then
        echo "[build] ERROR: Source directory $src_dir not found!" >&2
        return 1
    fi

    echo "[build] Extracting binaries from $src_dir..."
    # Find llama-server archive
    local llama_archive
    llama_archive=$(find "$src_dir" -maxdepth 1 \( -name "llama-*.tar.gz" -o -name "llama-*.zip" \) | head -n1 || true)
    if [ -z "$llama_archive" ]; then
        echo "[build] ERROR: No llama-* archive found in $src_dir!" >&2
        return 1
    fi

    if [[ "$llama_archive" =~ \.tar\.gz$ ]]; then
        tar -xzf "$llama_archive" --strip-components=1 -C "$stage/bin"
    else
        unzip -q -o "$llama_archive" -d "$stage/bin"
    fi

    # If CUDA, unpack cudart
    if [ "$backend" = "cuda" ]; then
        local cudart_archive
        cudart_archive=$(find "$src_dir" -maxdepth 1 \( -name "cudart-*.tar.gz" -o -name "cudart-*.zip" \) | head -n1 || true)
        if [ -z "$cudart_archive" ]; then
            echo "[build] ERROR: No cudart-* archive found in $src_dir for CUDA build!" >&2
            return 1
        fi

        if [ "$os" = "linux" ]; then
            mkdir -p "$stage/cudart"
            tar -xzf "$cudart_archive" --strip-components=1 -C "$stage/cudart"
        else
            # For Windows, co-locate CUDA DLLs in bin/ next to llama-server.exe for automatic DLL discovery
            unzip -q -o "$cudart_archive" -d "$stage/bin"
        fi
    fi

    # 2. Copy app files
    echo "[build] Copying application files..."
    cp app.py gui.py downloader.py "$stage/"
    cp README-bundle.md "$stage/README.md"
    mkdir -p "$stage/lcpp"
    cp lcpp/config.py lcpp/provider_client.py lcpp/thinkfarm.webp "$stage/lcpp/"

    if [ "$os" = "linux" ]; then
        cp thinkfarm.sh "$stage/"
        chmod +x "$stage/thinkfarm.sh"
    else
        cp thinkfarm.bat "$stage/"
        # Also copy thinkfarm.sh for Windows users who use Git Bash / MSYS2
        cp thinkfarm.sh "$stage/"
    fi

    # 3. Wheelhouse
    prepare_wheelhouse "$os" "$stage/wheelhouse"

    # 4. Model placeholders
    mkdir -p "$stage/qwen3.8-27b" "$stage/qwen3.6-35b"
    touch "$stage/qwen3.8-27b/.gitkeep" "$stage/qwen3.6-35b/.gitkeep"

    # 5. Create final archive
    echo "[build] Packaging $bundle..."
    local stage_parent="$DIST_DIR/stage"
    if [ "$os" = "linux" ]; then
        local archive="$DIST_DIR/$bundle.tar.gz"
        tar -czf "$archive" -C "$stage_parent" "$bundle"
        sha256sum "$archive" >> "$DIST_DIR/SHA256SUMS"
        echo "[build] Created $archive"
    else
        local archive="$DIST_DIR/$bundle.zip"
        (cd "$stage_parent" && zip -r -q "$archive" "$bundle")
        sha256sum "$archive" >> "$DIST_DIR/SHA256SUMS"
        echo "[build] Created $archive"
    fi

    rm -rf "$stage"
    echo "[build] Completed $variant successfully."
}

# Clear previous checksum file
rm -f "$DIST_DIR/SHA256SUMS"

if [ "$TARGET" = "all" ]; then
    for V in linux-cuda linux-vulkan win-cuda win-vulkan; do
        build_variant "$V"
    done
else
    build_variant "$TARGET"
fi

echo
echo "============================================================"
echo "[build] All requested builds complete in $DIST_DIR:"
echo "============================================================"
ls -lh "$DIST_DIR"/*.tar.gz "$DIST_DIR"/*.zip 2>/dev/null || true
echo
echo "Checksums ($DIST_DIR/SHA256SUMS):"
cat "$DIST_DIR/SHA256SUMS" 2>/dev/null || true
