# thinkfarm self-contained provider app

A thinkfarm provider (same provider → central-server WebSocket protocol as
`lcpp/headless.py`), **minus any separate model-runner install**: the app
downloads and caches `llama.cpp` (`llama-server`, build `b12431`, CUDA or
Vulkan) into `~/.thinkfarm/runtime/`, loads your chosen model, waits until it is
ready, points the provider at it, and shuts it down cleanly on exit.

## Quick start (single-file executable)

No Python, `venv`, Ollama, or CUDA toolkit install is required — just download
and run the single executable (`thinkfarm-provider` on Linux,
`thinkfarm-provider.exe` on Windows):

```bash
# Linux
chmod +x thinkfarm-provider
./thinkfarm-provider                                      # PyQt6 dashboard GUI (default)
./thinkfarm-provider --headless                           # headless daemon (Ctrl+C to stop)
./thinkfarm-provider --download-runtime [--backend cuda|vulkan]
./thinkfarm-provider --download [--model NAME] [--backend cuda|vulkan]
```

```powershell
# Windows
.\thinkfarm-provider.exe                                  # PyQt6 dashboard GUI (default)
.\thinkfarm-provider.exe --headless                       # headless daemon (Ctrl+C to stop)
.\thinkfarm-provider.exe --download-runtime --backend cuda
.\thinkfarm-provider.exe --download --model qwen3.8-27b
```

### First-run flow
1. On first launch, `~/.thinkfarm/custom.ini` is created automatically with a
   generated `provider_id` and ready-to-edit defaults.
2. Pick your **Backend** (`CUDA` for NVIDIA GPUs, auto-detected when an NVIDIA
   driver is present; `Vulkan` for AMD/Intel/NVIDIA) and **Model** in the GUI.
3. Click **Download Runtime** (once per backend/build) and **Download Model**
   (once per model). Downloads are resumable, pre-checked for free disk space,
   and verified with SHA-256.
4. Click **Start Provider**. Closing the window minimizes to the system tray;
   use **Exit** from the tray menu (or `Ctrl+C` in `--headless` mode) to stop
   `llama-server` and quit cleanly.

To update later, simply replace the `thinkfarm-provider` executable — your
config, cached runtime, and model weights in `~/.thinkfarm/` are preserved.

## Running from source

You can also run directly from this folder without building an executable
(requires Python 3.10+):

```bash
./thinkfarm.sh                           # Linux GUI (creates ./venv if needed)
./thinkfarm.sh --headless                # Linux headless daemon
./thinkfarm.sh --download [--model NAME] # Download runtime + model weights
```

On Windows, use `thinkfarm.bat` with the same flags.

## Building the single-file executable

PyInstaller builds native executables (no cross-compilation):

```bash
# Linux (outputs dist/thinkfarm-provider)
./build-onefile.sh
```

```powershell
# Windows (outputs dist\thinkfarm-provider.exe)
.\build-onefile.ps1
```

> **Linux portability note:** PyInstaller links against the host's `glibc`. To
> distribute a Linux binary that runs across older and newer distros, run
> `./build-onefile.sh` inside an older-glibc container (e.g. Ubuntu 22.04).

Legacy tarball/zip bundles (with pre-packed `bin/` and `wheelhouse/`) can still
be built with `./build.sh [linux-cuda | linux-vulkan | win-cuda | win-vulkan | all]`.

## Requirements

- **OS**: Linux x86_64 or Windows x64
- **GPU**:
  - **CUDA backend**: NVIDIA driver supporting CUDA 13 (no CUDA toolkit needed;
    `cudart` / `cublas` are downloaded with the runtime). On Linux, the app
    locates `libcuda.so.1` in standard `/usr/lib*` paths or the Nix store.
  - **Vulkan backend**: Any Vulkan-capable GPU driver (AMD, Intel, or NVIDIA).
- **Network**: Access to `https://app.thinkfarm.eu`, GitHub Releases (for
  runtime downloads), and Hugging Face (for model weights).

## File layout

### User data (`~/.thinkfarm/`)
| Path | Purpose |
|---|---|
| `~/.thinkfarm/custom.ini` | Provider config (auto-created on first run) |
| `~/.thinkfarm/.env` | Optional environment overrides (`CENTRAL_SERVER_URL`, etc.) |
| `~/.thinkfarm/runtime/<os>-<backend>-b12431/` | Downloaded & extracted `llama-server` (+ `cudart/` on Linux CUDA) |
| `~/.thinkfarm/models/<model>/` | Downloaded GGUF weights (override with `MODELS_PATH`) |
| `~/.thinkfarm/logs/llama-server.log` | Child `llama-server` output (appended per start) |

### Project files
| Path | Purpose |
|---|---|
| `entrypoint.py` | Unified PyInstaller entry point (routes CLI flags vs GUI) |
| `app.py` | Headless provider + `LlamaServer` child-process lifecycle |
| `gui.py` | PyQt6 dashboard (config, runtime/model downloads, status, logs, tray) |
| `runtime.py` | Runtime backend detection, pinned GitHub Release download, SHA-256 verification, and extraction |
| `downloader.py` | Resumable chunked downloader with progress reporting and SHA-256 verification |
| `lcpp/` | Shared provider client, `ConfigManager`, and protocol docs |
| `thinkfarm-provider.spec` | PyInstaller `--onefile` spec |
| `build-onefile.sh` / `build-onefile.ps1` | Single-file executable build scripts |

## Configuration

`~/.thinkfarm/custom.ini` is created automatically on first run:

```ini
[provider]
provider_id = <generated-uuid>
slots = 1
models_path =
backend =
selected_model =
llama_args =
```

Leaving optional keys blank is safe and uses defaults:
- `provider_id`: Your provider identity (auto-generated UUID if blank).
- `slots`: Concurrent jobs / `llama-server --parallel` (default `1`). Each slot
  allocates its own KV cache, so only raise this if you have VRAM headroom.
- `models_path` (or `MODELS_PATH` env var): Relocate model downloads/loading
  (default `~/.thinkfarm/models`), keeping per-model subfolders underneath.
- `backend` (or `THINKFARM_BACKEND` env var / `--backend`): `cuda` or `vulkan`
  (blank = auto-detect).
- `selected_model` (or `THINKFARM_MODEL` env var / `--model`): Which model to
  load (default `qwen3.8-27b`).
- `llama_args` (or `LLAMA_ARGS` env var): Extra flags passed verbatim to
  `llama-server` (e.g. `--tensor-split 57,43`).

Additional environment variables:
- `LOCAL_PORT`: Pin the local `llama-server` port (default: free ephemeral port).
- `READY_TIMEOUT`: Seconds to wait for model load readiness (default `1800`).
