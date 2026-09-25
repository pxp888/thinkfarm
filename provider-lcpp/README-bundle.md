# thinkfarm provider — self-contained bundle

Bundled `llama-server` (llama.cpp build b11064, CUDA 13.3, Ubuntu x86_64) + the
thinkfarm provider app. No CUDA toolkit, no Ollama, no llama.cpp install needed —
only an NVIDIA driver.

## Requirements

- Linux x86_64 with glibc compatible with the Ubuntu-built binaries
- NVIDIA driver installed (`libcuda.so.1` present in a standard path or nix store)
- Python >= 3.10 (any distro python3 is fine; core offline wheels cover 3.10–3.15, otherwise installed from network)
- Network access to `https://app.thinkfarm.eu` at runtime

## Run

```bash
tar -xzf thinkfarm-provider-linux-x64-cuda13-b11064.tar.gz -C <dir>
cd <dir>
./thinkfarm.sh                           # PyQt6 dashboard GUI (close minimizes to tray; Exit quits)
./thinkfarm.sh --headless                # headless daemon (Ctrl+C to stop)
./thinkfarm.sh --download [--model NAME] # download required model weights from the CLI
```

`thinkfarm.sh` finds a Python with the runtime deps (`httpx`, `websockets`; plus
`PyQt6` for the GUI), or creates a `venv/` and installs them from network or
the bundled `wheelhouse/` (offline). If PyQt6 wheels were not bundled at build
time, only the GUI needs network on first run — headless is unaffected.

## Models

Each model lives in its own folder; exactly one is loaded at a time (they never
run simultaneously). Choose with `--model NAME`, the `THINKFARM_MODEL` env var,
or the Model dropdown in the GUI (persisted to `~/.thinkfarm/custom.ini`):

- `qwen3.8-27b/` (default) — `Qwen3.8-27B-UD-Q4_K_M.gguf`, `mmproj-BF16.gguf`,
  `mtp-Qwen3.8-27B-Q4_0.gguf` (MTP draft for speculative decoding)
- `qwen3.6-35b/` — `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, `mmproj-BF16.gguf`
- `qwen3.5-9b/` — `Qwen3.5-9B-Q4_K_M.gguf`, `mmproj-BF16.gguf` (MTP module baked
  into the GGUF for speculative decoding)

By default each model downloads into and loads from its folder next to the app.
To store weights elsewhere (e.g. a bigger disk), set `MODELS_PATH` in
`~/.thinkfarm/custom.ini` — both download and load use it, keeping the same
per-model subfolders:

    [provider]
    MODELS_PATH = /mnt/persist/models

The selected model's required files are checked at startup. If files are missing,
you can download them automatically by clicking **Download Model** in the dashboard GUI,
or running `./thinkfarm.sh --download --model <name>` in the terminal. Downloads are resumable
and verify file integrity via SHA-256.

Context window: left to llama.cpp's default — the provider probes it from the
server and uses it to gate job acceptance.

## Config

Read from `~/.thinkfarm/custom.ini` (provider id, slots, etc.) and optional `.env`
next to the app. Logs go to `logs/llama-server.log`. Stop: **Exit** from the tray
menu (GUI) or Ctrl+C (`--headless`) — the llama-server child is terminated cleanly.
