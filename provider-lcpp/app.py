#!/usr/bin/env python3
"""Self-contained thinkfarm provider app.

Same as lcpp/headless.py (provider daemon speaking the provider -> central-server
WebSocket protocol), except inference is self-managed: this app SPAWNS a bundled
llama-server (prebuilt CUDA 13 binaries) as a child process, waits until it is
ready, points the provider config at its local port, and shuts the child down
cleanly on exit. No Ollama / standalone llama-server / CUDA toolkit install
required on the user's machine — only an nvidia driver.

One bundled model is loaded at a time (see MODELS below); choose with
--model NAME or the THINKFARM_MODEL env var, e.g. ./thinkfarm.sh --headless --model qwen3.6-35b
Stop with Ctrl+C; the llama-server child is terminated cleanly.
"""

import asyncio
import logging
import os
import signal
import socket
from dataclasses import dataclass, replace
import subprocess
import sys
import time
from pathlib import Path

# Ensure PyQt6-related speedups are never imported (saves time/memory)
sys.modules["websockets.speedups"] = None

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lcpp"))  # reuse lcpp's provider implementation

from downloader import (
    ModelFileArtifact,
    download_artifacts,
    format_bytes,
    format_eta,
    ProgressInfo,
)
from config import ConfigManager  # noqa: E402
from provider_client import ProviderClient  # noqa: E402

# --- bundled layout ---------------------------------------------------------
# Standard bundle layout uses bin/ (and optional cudart/); fall back to legacy folders.
BIN_DIR = ROOT / "bin" if (ROOT / "bin").exists() else ROOT / "llama-b11064"
CUDART_DIR = ROOT / "cudart" if (ROOT / "cudart").exists() else ROOT / "cudart-llama-b11064-bin-ubuntu-cuda-13.3-x64"
SERVER_BIN = BIN_DIR / ("llama-server.exe" if sys.platform == "win32" else "llama-server")


def is_cuda_backend() -> bool:
    """Detect whether the bundled binaries use the CUDA backend."""
    if CUDART_DIR.exists():
        return True
    return (BIN_DIR / "libggml-cuda.so").exists() or (BIN_DIR / "ggml-cuda.dll").exists()


def required_binaries() -> list[Path]:
    """Binaries and runtime libraries required to launch llama-server."""
    bins = [SERVER_BIN]
    if is_cuda_backend():
        if sys.platform == "win32":
            cuda_dlls = list(CUDART_DIR.glob("cudart64_*.dll")) + list(BIN_DIR.glob("cudart64_*.dll"))
            if cuda_dlls:
                bins.append(cuda_dlls[0])
            else:
                bins.append(CUDART_DIR / "cudart64_13.dll")
        else:
            cudarts = list(CUDART_DIR.glob("libcudart.so.*")) + list(BIN_DIR.glob("libcudart.so.*"))
            if cudarts:
                bins.append(cudarts[0])
            else:
                bins.append(CUDART_DIR / "libcudart.so.13")
    return bins
# --- bundled models ---------------------------------------------------------
# Exactly one of these is loaded into the child llama-server at a time. They
# are not co-resident: weights don't fit in VRAM together and they never need
# to run simultaneously. Selection precedence: --model > THINKFARM_MODEL env >
# saved GUI choice (SELECTED_MODEL) > DEFAULT_MODEL.
#
# The context window is left to llama.cpp's default; the provider probes the
# live n_ctx from /v1/models and gates job acceptance on it.

@dataclass(frozen=True)
class ModelSpec:
    label: str            # display name (GUI dropdown)
    directory: Path       # folder holding the gguf files
    main_model: Path      # dir-relative; passed to -m
    mmproj: Path | None = None  # dir-relative vision projector, if any
    extra_args: tuple[tuple[str, object], ...] = ()
    downloads: tuple[ModelFileArtifact, ...] = ()
    publish_name: str = ""
    publish_digest: str = ""
    # Model-specific flags appended verbatim; Path values resolve against directory.

MODELS = {
    "qwen3.8-27b": ModelSpec(
        label="Qwen3.8 27B",
        directory=ROOT / "qwen3.8-27b",
        main_model=Path("Qwen3.8-27B-UD-Q4_K_M.gguf"),
        mmproj=Path("mmproj-BF16.gguf"),
        extra_args=(
            ("--spec-type", "draft-mtp"),
            ("--spec-draft-model", Path("mtp-Qwen3.8-27B-Q4_0.gguf")),
            ("--spec-draft-n-max", "3"),
            ("--spec-draft-ngl", "99"),
        ),
        downloads=(
            ModelFileArtifact(
                rel_path=Path("Qwen3.8-27B-UD-Q4_K_M.gguf"),
                url="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf",
                size_bytes=16464440224,
                sha256="322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
            ),
            ModelFileArtifact(
                rel_path=Path("mmproj-BF16.gguf"),
                url="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/mmproj-BF16.gguf",
                size_bytes=931146432,
                sha256="83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53",
            ),
            ModelFileArtifact(
                rel_path=Path("mtp-Qwen3.8-27B-Q4_0.gguf"),
                url="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
                size_bytes=1369590656,
                sha256="50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e",
            ),
        ),
        publish_name="qwen3.8:27b-ud-q4_k_m",
        publish_digest="322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
    ),
    "qwen3.6-35b": ModelSpec(
        label="Qwen3.6 35B-A3B (MoE)",
        directory=ROOT / "qwen3.6-35b",
        main_model=Path("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
        mmproj=Path("mmproj-BF16.gguf"),
        downloads=(
            ModelFileArtifact(
                rel_path=Path("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
                url="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                size_bytes=22663387424,
                sha256="0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b",
            ),
            ModelFileArtifact(
                rel_path=Path("mmproj-BF16.gguf"),
                url="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/mmproj-BF16.gguf",
                size_bytes=902822528,
                sha256="da63cb47a76763c712393f8a017070188a304fa39f8aeea6edc629ed7b975cfa",
            ),
        ),
        publish_name="qwen3.6:35b-a3b-ud-q4_k_m",
        publish_digest="0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b",
    ),
    "qwen3.5-9b": ModelSpec(
        label="Qwen3.5 9B",
        directory=ROOT / "qwen3.5-9b",
        main_model=Path("Qwen3.5-9B-Q4_K_M.gguf"),
        mmproj=Path("mmproj-BF16.gguf"),
        # MTP module is baked into the main GGUF (unsloth -MTP-GGUF repo); no
        # separate --spec-draft-model file needed.
        extra_args=(
            ("--spec-type", "draft-mtp"),
            ("--spec-draft-n-max", "6"),
        ),
        downloads=(
            ModelFileArtifact(
                rel_path=Path("Qwen3.5-9B-Q4_K_M.gguf"),
                url="https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
                size_bytes=5868826976,
                sha256="e8dd94817e95d6c0939102049d068418269978377b13616c4726235e232841fe",
            ),
            ModelFileArtifact(
                rel_path=Path("mmproj-BF16.gguf"),
                url="https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF/resolve/main/mmproj-BF16.gguf",
                size_bytes=921704928,
                sha256="73c90471c349fec797f3969b59249259318ee3fa855a684da7ef35a46f063a9e",
            ),
        ),
        publish_name="qwen3.5:9b-q4_k_m",
        publish_digest="e8dd94817e95d6c0939102049d068418269978377b13616c4726235e232841fe",
    ),
}
DEFAULT_MODEL = "qwen3.8-27b"


def apply_models_dir(config) -> None:
    """Rebind model folders to MODELS_PATH (custom.ini / .env). Affects both
    the download target and where llama-server loads from; empty = next to
    the app."""
    if not config.models_path:
        return
    base = Path(os.path.expanduser(config.models_path))
    for name in MODELS:
        MODELS[name] = replace(MODELS[name], directory=base / name)


def parse_model_arg(argv: list[str]) -> str | None:
    """Value of --model NAME (also accepts --model=NAME) from argv, else None."""
    for i, a in enumerate(argv):
        if a == "--model":
            if i + 1 >= len(argv):
                print(f"[app] ERROR: --model needs a value. Valid models: {', '.join(MODELS)}", file=sys.stderr)
                sys.exit(1)
            return argv[i + 1]
        if a.startswith("--model="):
            return a.split("=", 1)[1]
    return None


def resolve_model(cli_value: str | None = None, env=None, saved: str = "") -> str:
    """Pick which bundled model to load; unknown names are fatal (a node should
    never silently run a model it wasn't told to)."""
    for cand in (cli_value, os.environ.get("THINKFARM_MODEL"), saved):
        if not cand:
            continue
        if cand in MODELS:
            return cand
        print(f"[app] ERROR: unknown model '{cand}'. Valid models: {', '.join(MODELS)}", file=sys.stderr)
        sys.exit(1)
    return DEFAULT_MODEL


def model_files(name: str) -> list[Path]:
    """gguf files required to run a model (startup preflight)."""
    m = MODELS[name]
    files = [m.directory / m.main_model]
    if m.mmproj is not None:
        files.append(m.directory / m.mmproj)
    files += [m.directory / v for _, v in m.extra_args if isinstance(v, Path)]
    return files

def model_download_status(name: str) -> dict:
    """Check status of model files (installed, missing, partial, bytes remaining)."""
    m = MODELS[name]
    total_bytes = sum(a.size_bytes for a in m.downloads)
    downloaded_bytes = 0
    missing = []
    partial = []

    for a in m.downloads:
        target = m.directory / a.rel_path
        part = target.with_name(target.name + ".part")
        if target.exists() and target.stat().st_size == a.size_bytes:
            downloaded_bytes += a.size_bytes
        elif part.exists():
            part_size = min(part.stat().st_size, a.size_bytes)
            downloaded_bytes += part_size
            partial.append(a)
            missing.append(a)
        else:
            missing.append(a)

    remaining_bytes = max(0, total_bytes - downloaded_bytes)
    installed = (len(missing) == 0) and (total_bytes > 0)

    return {
        "installed": installed,
        "total_bytes": total_bytes,
        "downloaded_bytes": downloaded_bytes,
        "remaining_bytes": remaining_bytes,
        "missing_artifacts": missing,
        "partial_artifacts": partial,
    }


def run_cli_download(model_name: str):
    """Download missing model files from the command line with progress output."""
    m = MODELS[model_name]
    status = model_download_status(model_name)
    if status["installed"]:
        print(f"[download] Model '{m.label}' ({model_name}) is already fully downloaded.")
        return

    missing = status["missing_artifacts"]
    rem_fmt = format_bytes(status["remaining_bytes"])
    print(f"[download] Downloading {len(missing)} file(s) for '{m.label}' ({rem_fmt} remaining)...")
    print(f"[download] Target directory: {m.directory}")

    last_line_len = 0

    def on_progress(p: ProgressInfo):
        nonlocal last_line_len
        msg = f"[{p.file_index}/{p.total_files}] {p.filename}: {p.file_percent:5.1f}% ({format_bytes(p.file_bytes_done)}/{format_bytes(p.file_bytes_total)}) • {format_bytes(p.speed_bps)}/s • ETA {format_eta(p.eta_seconds)}"
        out = chr(13) + "[download] " + msg
        pad = " " * max(0, last_line_len - len(out))
        print(out + pad, end="", flush=True)
        last_line_len = len(out)
    try:
        download_artifacts(
            list(m.downloads),
            m.directory,
            progress_callback=on_progress,
        )
        print("\n[download] Download complete for '" + m.label + "'! Ready to start.")
    except KeyboardInterrupt:
        print("\n[download] Download paused (Ctrl+C). Resume any time by running download again.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[download] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

def apply_publish_info(cfg, name: str):
    """Point config at the model's published name/digest."""
    m = MODELS[name]
    cfg.model_name = m.publish_name
    cfg.model_digest = m.publish_digest


HOST = "127.0.0.1"
PORT = int(os.environ.get("LOCAL_PORT", "8086"))  # 8086 = oqwen.sh's host mapping

LOG_DIR = ROOT / "logs"
SERVER_LOG = LOG_DIR / "llama-server.log"

READY_TIMEOUT = float(os.environ.get("READY_TIMEOUT", "1800"))  # generous; a dead child trips this fast


class ShutdownRequested(Exception):
    """User asked to leave while the model was still loading."""


def pick_port() -> int:
    if os.environ.get("LOCAL_PORT"):
        return PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def build_args(port: int, name: str) -> list[str]:
    """Canonical invocation of a bundled model (Qwen3.x sampling per oqwen.sh)."""
    m = MODELS[name]
    alias = m.publish_name or name
    args = [
        str(SERVER_BIN),
        "-m", str(m.directory / m.main_model),
        "--alias", alias,
        "--host", HOST,
        "--port", str(port),
        "-ngl", "99",
        "--parallel", "1",
        "--jinja",
        "--reasoning-format", "deepseek",
        "--temp", "0.8",
        "--top-p", "0.9",
        "--top-k", "40",
        "--min-p", "0.0",
        "--repeat-penalty", "1.1",
        "--repeat-last-n", "64",
    ]
    if m.mmproj is not None:
        args += ["--mmproj", str(m.directory / m.mmproj)]
    for flag, value in m.extra_args:
        args.append(flag)
        args.append(str(m.directory / value) if isinstance(value, Path) else str(value))
    return args


def find_driver_dirs() -> list[str]:
    """Locate the nvidia driver's libcuda.so.1 for the child's loader path.

    Standard distros ship it in /usr/lib* (loader finds it anyway; appending is
    harmless). nix-based systems keep it in the nix store, which a plain process
    does not search — so find it explicitly there. Target box only needs the
    nvidia driver (no CUDA toolkit) to run this app.
    """
    import glob

    found: list[str] = []
    for pat in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib",
                "/nix/store/*-nvidia-x11-[0-9.]*-lib/lib", "/nix/store/*-nvidia-x11-[0-9.]*/lib"):
        for d in glob.glob(pat):
            if Path(d, "libcuda.so.1").exists() and d not in found:
                found.append(d)
    return found


class LlamaServer:
    """Owns the child llama-server process."""

    def __init__(self, port: int, model_name: str = DEFAULT_MODEL):
        self.port = port
        self.model_name = model_name
        self.proc: subprocess.Popen | None = None
        self.log_file = None

    def start(self):
        args = build_args(self.port, self.model_name)
        LOG_DIR.mkdir(exist_ok=True)
        env = dict(os.environ)
        # Configure shared library / DLL search paths
        if sys.platform == "win32":
            paths = [str(BIN_DIR)]
            if CUDART_DIR.exists():
                paths.append(str(CUDART_DIR))
            if env.get("PATH"):
                paths.append(env["PATH"])
            env["PATH"] = os.pathsep.join(paths)
        else:
            paths = [str(BIN_DIR)]
            if is_cuda_backend() and CUDART_DIR.exists():
                paths.insert(0, str(CUDART_DIR))
                paths.extend(find_driver_dirs())
            if env.get("LD_LIBRARY_PATH"):
                paths.append(env["LD_LIBRARY_PATH"])
            env["LD_LIBRARY_PATH"] = os.pathsep.join(paths)

        self.log_file = open(SERVER_LOG, "ab", buffering=0)
        self.log_file.write(
            b"\n=== " + str(time.strftime("%Y-%m-%d %H:%M:%S")).encode() + b" start ===\n"
        )
        print(f"[app] Starting llama-server on {HOST}:{self.port} (model: {MODELS[self.model_name].label})")
        print(f"[app]   args: {' '.join(args)}")
        print(f"[app]   log:  {SERVER_LOG}")

        popen_kwargs = {"env": env, "stdout": self.log_file, "stderr": subprocess.STDOUT}
        if sys.platform == "win32":
            # Suppress console window popup for the child process on Windows
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        self.proc = subprocess.Popen(args, **popen_kwargs)
        print(f"[app] llama-server pid {self.proc.pid} — loading model (first load can take a while)...")

    def _alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    async def wait_ready(self, shutdown: asyncio.Event | None = None):
        import httpx

        deadline = time.time() + READY_TIMEOUT
        async with httpx.AsyncClient(timeout=3) as client:
            while time.time() < deadline:
                if shutdown is not None and shutdown.is_set():
                    raise ShutdownRequested()
                if not self._alive():
                    raise RuntimeError(
                        f"llama-server exited with code {self.proc.returncode} during startup. "
                        f"Last log lines:\n{self._tail_log()}"
                    )
                try:
                    r = await client.get(f"http://{HOST}:{self.port}/v1/models")
                    if r.status_code == 200:
                        entry = r.json().get("data", [{}])[0]
                        name = entry.get("id", "?")
                        n_ctx = (entry.get("meta") or {}).get("n_ctx")
                        return name, n_ctx
                except Exception:
                    pass
                await asyncio.sleep(1.0)
        raise RuntimeError(
            f"llama-server did not become ready within {READY_TIMEOUT}s. "
            f"Last log lines:\n{self._tail_log()}"
        )

    def _tail_log(self, lines: int = 25) -> str:
        try:
            data = SERVER_LOG.read_bytes()
            tail = data.decode(errors="replace").splitlines()[-lines:]
            return "\n".join(tail)
        except Exception:
            return "(no log available)"

    def stop(self):
        if not self._alive():
            return
        print("[app] Stopping llama-server...")
        self.proc.terminate()  # SIGTERM: graceful in llama-server
        try:
            self.proc.wait(timeout=15)
            print("[app] llama-server stopped.")
        except subprocess.TimeoutExpired:
            print("[app] llama-server did not exit in 15s — SIGKILL.")
            self.proc.kill()
            self.proc.wait(timeout=5)
        if self.log_file:
            self.log_file.close()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence per-request noise from the readiness probes / provider traffic.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    config = ConfigManager(str(ROOT))
    apply_models_dir(config)
    model_name = resolve_model(parse_model_arg(sys.argv), saved=config.selected_model)

    if "--download" in sys.argv:
        run_cli_download(model_name)
        sys.exit(0)

    apply_publish_info(config, model_name)
    missing_bins = [p for p in required_binaries() if not p.exists()]
    missing_models = [p for p in model_files(model_name) if not p.exists()]
    if missing_bins or missing_models:
        for p in missing_bins:
            print(f"[app] ERROR: required binary missing: {p}", file=sys.stderr)
        if missing_models:
            print(f"[app] ERROR: model files missing for '{model_name}':", file=sys.stderr)
            for p in missing_models:
                print(f"[app]   - {p}", file=sys.stderr)
            print(f"[app] Run './thinkfarm.sh --download --model {model_name}' to download them automatically.", file=sys.stderr)
        sys.exit(1)

    port = pick_port()
    config.local_lcpp_url = f"http://{HOST}:{port}"  # point provider at the child we own

    shutdown_event = asyncio.Event()
    loop = asyncio.new_event_loop()

    def _sigint_handler():
        loop.call_soon_threadsafe(shutdown_event.set)

    signal.signal(signal.SIGINT, lambda *_: _sigint_handler())
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: _sigint_handler())
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, lambda *_: _sigint_handler())

    runner = LlamaServer(port, model_name)

    async def run():
        runner.start()
        try:
            ready_name, n_ctx = await runner.wait_ready(shutdown_event)
        except ShutdownRequested:
            print("\n[app] Shutdown requested during model load.")
            runner.stop()
            if not shutdown_event.is_set():
                shutdown_event.set()
            return

        print(f"[app] llama-server ready — model: {ready_name}"
              + (f", context window: {n_ctx:,} tokens" if n_ctx else ""))

        try:
            p_client = ProviderClient(
                config,
                status_callback=lambda s: print(f"[status] {s}"),
                log_callback=lambda msg, level=logging.INFO: logging.getLogger("thinkfarm.provider").log(level, msg),
            )
            task = loop.create_task(p_client.run())

            await shutdown_event.wait()
            print("\n[app] Shutting down provider...")
            await p_client.stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            runner.stop()

    try:
        loop.run_until_complete(run())
    finally:
        loop.close()
        print("[app] Done.")


if __name__ == "__main__":
    main()
