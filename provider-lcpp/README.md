# thinkfarm self-contained provider app

A thinkfarm provider (same provider → central-server WebSocket protocol as
`lcpp/headless.py`), **minus the separate model-runner install**: the app
spawns the bundled `llama.cpp` (`llama-server`, CUDA 13 build b11064) itself,
loads **Qwen3.8-27B (Q4_K_M) + vision projector + MTP draft model**, waits
until it is ready, points the provider at it, and shuts it down cleanly when
you stop the app.

## Run

```bash
./thinkfarm.sh                           # PyQt6 dashboard: config form, status dots, activity log,
                                         #   tray icon (close = minimize to tray, Exit = stop & quit)
./thinkfarm.sh --headless                # headless daemon
./thinkfarm.sh --download [--model NAME] # download required model weights directly from the CLI
```

If model weights are not yet downloaded on the machine, you can either click **Download Model** in the GUI dashboard or run `./thinkfarm.sh --download` in the terminal. Downloads are resumable, verified for file integrity (SHA-256), and pre-checked for sufficient free disk space.

Ctrl+C (or SIGTERM) stops the provider and the child `llama-server` cleanly.
That's the whole install story — copy the folder to the machine and run it.

## Requirements

- **Linux x86_64** (the bundled binaries are x64/x86_64 — no aarch64 build here)
- **NVIDIA driver only** — no CUDA toolkit, no Ollama, no separate
  `llama-server` needed. The CUDA runtime libraries (`libcudart.so.13`,
  `libcublas*`, ...) are bundled in `cudart-llama-b11064-bin-ubuntu-cuda-13.3-x64/`. The driver must
  support CUDA 13 (driver 580+/590+). The app finds `libcuda.so.1` from
  standard `/usr/lib*` locations or a nix store.
- **GPU memory ≈ 19 GB+ total** for the model + draft + projector
  (16.5 GB main GGUF, 1.4 GB MTP draft, 0.9 GB mmproj). It splits across all
  visible NVIDIA GPUs (tested: RTX 4080 16 GB + RTX 3080 12 GB). A single GPU
  needs headroom for all three weights plus KV cache (~20 GB+).
- **Python 3.10+**. Only two runtime deps for headless mode (`httpx`,
  `websockets`). The GUI adds `PyQt6`. If the current interpreter lacks them,
  `thinkfarm.sh` creates `./venv` and installs them once (offline bundles ship a
  `wheelhouse/`; PyQt6 wheels are included when available at build time).

## What ships where

| Path | Content |
|---|---|
| `app.py` | the app: spawns the child llama-server, waits for readiness, runs lcpp's provider daemon, clean shutdown |
| `gui.py` | PyQt6 dashboard (same worker flow as `app.py`, in a thread; default mode of `./thinkfarm.sh`) |
| `downloader.py` | resumable chunked model downloader with Range resume, progress reporting, and SHA256 checks |
| `thinkfarm.sh` | one-command launcher (`--headless` for the daemon) |
| `lcpp/` | provider client + protocol code (reused as-is; `provider-protocol.md` is the protocol spec) |
| `llama-b11064/` | `llama-server` binary + its `.so` files (llama.cpp b11064, CUDA 13) |
| `cudart-llama-b11064-bin-ubuntu-cuda-13.3-x64/` | bundled CUDA runtime libraries |
| `qwen3.8-27b/` | model, vision projector, MTP draft GGUFs (canonical invocation: `qwen3.8-27b/oqwen.sh`) |
| `logs/` | `llama-server.log` (child server output), `app-run.log` when run via `nohup ./thinkfarm.sh` |

## Configuration

Reuses the standard thinkfarm provider config: `~/.thinkfarm/custom.ini`
(`provider_id`, `slots`, ...) and the `~/.thinkfarm/*.json` context/baseline
caches. Defaults to the public central server `https://app.thinkfarm.net`.

- `LOCAL_PORT` (env, default: a free ephemeral port) — pin the llama-server
  port you own, e.g. `LOCAL_PORT=8086 ./thinkfarm.sh`.
- `READY_TIMEOUT` (env seconds, default 1800) — how long to wait for the
  child to report ready.
- `MODELS_PATH` (`custom.ini`, `[provider]`) — relocate where model weights
  are downloaded and loaded from, e.g. `MODELS_PATH=/mnt/persist/models`
  (per-model subfolders like `qwen3.8-27b/` are kept underneath it).

## Gotchas

- x64 binaries only; needs a recent glibc (built/tested against glibc 2.42).
- First load reads ~19 GB of weights from disk — allow up to a minute or two.
- Server logs land in `logs/llama-server.log` (appended per start).
- If the child fails to start, the app exits and prints the last server log
  lines; check the driver (`nvidia-smi`) and GPU memory.
