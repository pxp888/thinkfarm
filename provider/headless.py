"""
thinkfarm Provider Launcher (headless)

CLI front-end that starts/stops the provider via solo.py (no GUI).
Configuration loaded from ~/.thinkfarm/config.ini.
"""

import asyncio
import configparser
import uuid
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
import context_prober
from ollama_client import OllamaClient
from model_manager import ModelManager
from dotenv import load_dotenv

load_dotenv()

_CONFIG_PATH = os.path.expanduser("~/.thinkfarm/config.ini")
_PID_FILE = os.path.expanduser("~/.thinkfarm/provider.pid")

_server_process = None
_stop_event = threading.Event()
_mgmt_stop_event = threading.Event()
_ollama = None
_model_manager = None
_shutdown_requested = False


def _load_config():
    """Load configuration from ~/.thinkfarm/config.ini."""
    config = configparser.ConfigParser()
    config.read(_CONFIG_PATH)

    # Priority: Config File > Environment Variable > System ID > Default
    provider_id = os.environ.get("PROVIDER_ID", "") or \
                  (config.get("provider", "provider_id", fallback=None) if config.has_section("provider") else None) or \
                  str(uuid.uuid4())
    ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    managed_storage_gb = os.environ.get("MANAGED_STORAGE_GB", "30")
    auto_manage_str = os.environ.get("AUTO_MANAGE", "False").lower()
    auto_manage = auto_manage_str in ("true", "1", "yes")

    if config.has_section("provider"):
        provider_id = config.get("provider", "provider_id", fallback=provider_id)
        ollama_url = config.get("provider", "ollama_url", fallback=ollama_url)
        managed_storage_gb = config.get("provider", "managed_storage_gb", fallback=managed_storage_gb)
        auto_manage = config.getboolean("provider", "auto_manage", fallback=auto_manage)

    return {
        "provider_id": provider_id,
        "ollama_url": ollama_url,
        "managed_storage_gb": managed_storage_gb,
        "auto_manage": auto_manage,
    }


def _write_pid():
    """Write current PID to .pid file."""
    config_dir = os.path.dirname(_PID_FILE)
    os.makedirs(config_dir, exist_ok=True)
    with open(_PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def _read_pid():
    """Read PID from .pid file."""
    if not os.path.exists(_PID_FILE):
        return None
    try:
        with open(_PID_FILE, "r") as f:
            return int(f.read().strip())
    except (ValueError, IOError):
        return None


def _remove_pid():
    """Remove PID file."""
    try:
        os.remove(_PID_FILE)
    except OSError:
        pass


def _managed_model_loop():
    """Background loop to handle automated model management."""
    global _ollama, _model_manager, _shutdown_requested

    while not _shutdown_requested and not _stop_event.is_set():
        try:
            config = configparser.ConfigParser()
            config.read(_CONFIG_PATH)

            # Re-apply priority: Config File > Environment Variable > Default
            auto_manage_str = os.environ.get("AUTO_MANAGE", "False").lower()
            auto_manage = auto_manage_str in ("true", "1", "yes")
            limit_gb_str = os.environ.get("MANAGED_STORAGE_GB", "30")

            if config.has_section("provider"):
                auto_manage = config.getboolean("provider", "auto_manage", fallback=auto_manage)
                limit_gb_str = config.get("provider", "managed_storage_gb", fallback=limit_gb_str)

            limit_gb = float(limit_gb_str)

            if auto_manage and limit_gb > 0:
                # Run optimization
                newly_pulled = asyncio.run(_model_manager.optimize_portfolio(limit_gb))

                if newly_pulled:
                    print(f"[MGMT-LOOP] New models pulled: {newly_pulled}. Restarting service for probing...")

                    # Programmatically stop service
                    was_running = _server_process is not None
                    if was_running:
                        _stop_service_sync()
                        while _server_process is not None:
                            time.sleep(1)

                    # Run probing for new models
                    print(f"[MGMT-LOOP] Probing context for: {newly_pulled}")
                    existing_limits = context_prober.load_context_limits()
                    asyncio.run(context_prober.run_context_probing(
                        _ollama.base_url,
                        newly_pulled,
                        existing_limits
                    ))

                    # Start service again
                    if was_running:
                        _start_service_sync()

        except Exception as e:
            print(f"[MGMT-LOOP] Error in managed model loop: {e}")

        # Wait up to 1 hour, but wake if shutdown is requested
        while not _shutdown_requested and not _stop_event.is_set():
            time.sleep(1)
            if time.time() % 3600 < 1.1:
                break


def _start_service_sync():
    """Start solo.py subprocess. Called from management loop."""
    global _server_process
    if _server_process is not None:
        print("[STARTUP] Service is already running.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "solo.py")
    project_root = os.path.dirname(script_dir)

    _server_process = subprocess.Popen(
        [sys.executable, "-u", script_path],
        cwd=project_root,
    )
    print(f"[STARTUP] solo.py started (PID { _server_process.pid })")


def _stop_service_sync():
    """Stop solo.py subprocess. Called from management loop."""
    global _server_process
    if _server_process is None:
        return
    _stop_event.set()
    print("[STOP] Initiating server shutdown...")
    _server_process.terminate()
    try:
        _server_process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        print("[STOP] Soft stop timed out, forcing termination...")
        _server_process.kill()
        _server_process.wait()
    _server_process = None
    print("[STOP] Server stopped.")


def _get_status():
    """Get current provider status string."""
    if _server_process is not None and _server_process.poll() is None:
        return "running"
    return "stopped"


def cmd_start():
    """Execute the same logic as pressing the Start button."""
    global _server_process, _stop_event, _shutdown_requested, _ollama, _model_manager, _mgmt_stop_event

    # Check if already running
    if _get_status() == "running":
        print("ERROR: Provider is already running.")
        return 1

    pid = _read_pid()
    if pid and pid != os.getpid():
        # Check if the old process is still alive
        try:
            os.kill(pid, 0)
            print(f"ERROR: A provider process is still running with PID {pid}. Stop it first (e.g., 'python headless.py stop').")
            return 1
        except OSError:
            os.remove(_PID_FILE) if os.path.exists(_PID_FILE) else None

    # Load config
    config = _load_config()
    print(f"[CONFIG] Provider ID: {config['provider_id']}")
    print(f"[CONFIG] Ollama URL: {config['ollama_url']}")
    print(f"[CONFIG] Managed Storage: {config['managed_storage_gb']} GB")
    print(f"[CONFIG] Auto Manage Models: {config['auto_manage']}")

    # 1. Check for unscanned models
    url = config["ollama_url"].strip() or "http://127.0.0.1:11434"
    print(f"[STARTUP] Checking for unscanned models at {url}...")

    all_models = asyncio.run(context_prober.get_ollama_models(url))
    existing_limits = context_prober.load_context_limits()
    unscanned = [m for m in all_models if m not in existing_limits]

    if unscanned:
        print(f"[STARTUP] Found {len(unscanned)} unscanned model(s): {unscanned}")
        print("[STARTUP] Running context probing...")
        asyncio.run(context_prober.run_context_probing(
            url,
            unscanned,
            existing_limits
        ))
        print("[STARTUP] Probing complete.")
    else:
        print("[STARTUP] All models already scanned.")

    # 2. Initialize managed model state
    _ollama = OllamaClient()
    _model_manager = ModelManager(_ollama, os.environ.get("SERVER_URL", "https://app.thinkfarm.net"))

    # 3. Start solo.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "solo.py")
    project_root = os.path.dirname(script_dir)

    _stop_event.clear()
    _server_process = subprocess.Popen(
        [sys.executable, "-u", script_path],
        cwd=project_root,
    )
    _write_pid()
    print(f"[STARTUP] solo.py started (PID {_server_process.pid})")

    # 4. Start managed model loop in background
    config_dir = os.path.expanduser("~/.thinkfarm")
    os.makedirs(config_dir, exist_ok=True)

    def _mgmt():
        _managed_model_loop()

    mgmt_thread = threading.Thread(target=_mgmt, daemon=True)
    mgmt_thread.start()

    print("\nProvider is RUNNING. Status: python headless.py status")
    print("To stop:  python headless.py stop\n")

    # 5. Wait for the process or Ctrl-C
    try:
        while _server_process is not None and _server_process.poll() is None:
            time.sleep(1)
        if _server_process is not None:
            rc = _server_process.returncode
            print(f"\nsolo.py exited with code {rc}")
    except KeyboardInterrupt:
        print("\n\nReceived interrupt. Shutting down...")
        if _server_process is not None:
            _stop_service_sync()
        _remove_pid()
        _shutdown_requested = True
    else:
        print("\n\nServer process ended unexpectedly.")


def cmd_stop():
    """Stop the running provider (same as pressing the Stop button)."""
    global _server_process, _stop_event, _shutdown_requested

    pid = _read_pid()
    if pid and pid != os.getpid():
        try:
            os.kill(pid, 0)
        except OSError:
            print("No running instance (stale PID file cleared).")
            _remove_pid()
            return 1

    if _server_process is not None and _server_process.poll() is None:
        _stop_event.set()
        _shutdown_requested = True
        _server_process.terminate()
        print("Stopping provider...")
        _server_process.wait()
        _server_process = None
        _remove_pid()
        print("Provider stopped.")
        return 0
    elif pid:
        # Process managed externally (not by this script), signal it
        print(f"Attempting to stop provider (PID {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
            os.waitpid(pid, 0)
            print("Provider stopped.")
        except (ProcessLookupError, ChildProcessError):
            print("No running instance (stale PID file removed).")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            return 1
        finally:
            _remove_pid()
    else:
        print("No running instance. Nothing to stop.")
        return 1
    return 0


def cmd_status():
    """Print current status (like the status bar)."""
    pid = _read_pid()
    status = _get_status()

    if status == "running":
        dot = "●"
        text = f"SYSTEM RUNNING ({pid})"
        color = "[32m"  # green escape
        reset = "\x1b[0m"
        print(f"\x1b[32m{dot} {text}\x1b[0m")
    else:
        dot = "●"
        text = "SYSTEM STOPPED"
        color = "\x1b[31m"
        reset = "\x1b[0m"
        print(f"\x1b[31m{dot} {text}\x1b[0m")
    return 0 if status == "running" else 1


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python headless.py start     - Start the provider")
        print("  python headless.py stop      - Stop the provider")
        print("  python headless.py status    - Show provider status")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "start":
        cmd_start()
    elif cmd == "stop":
        exit_code = cmd_stop()
        sys.exit(exit_code)
    elif cmd == "status":
        exit_code = cmd_status()
        sys.exit(exit_code)
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python headless.py [start|stop|status]")
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
