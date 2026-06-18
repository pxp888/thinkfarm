"""
thinkfarm Provider Launcher (headless)

CLI front-end that starts/stops the provider via zombie.py (no GUI).
Configuration loaded from ~/.thinkfarm/config.ini.
"""

import asyncio
import configparser
import json
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
_PRIORITY_MODEL_PATH = os.path.expanduser("~/.thinkfarm/priority_model.txt")
_PID_FILE = os.path.expanduser("~/.thinkfarm/provider.pid")
_SLOPEMON_DATA_PATH = os.path.expanduser("~/.thinkfarm/_slopemon.json")
_TRIGGER_PATH = os.path.expanduser("~/.thinkfarm/solo_slope_trigger.json")

# ---------------------------------------------------------------------------
# Blacklist
# ---------------------------------------------------------------------------
_BLACKLIST_FILE = os.path.expanduser("~/.thinkfarm/blacklisted_models.json")
_blacklisted_models: set = set()


def load_blacklist() -> None:
    """Load the blacklist from disk into ``_blacklisted_models``."""
    global _blacklisted_models
    try:
        if os.path.exists(_BLACKLIST_FILE):
            with open(_BLACKLIST_FILE, "r") as f:
                _blacklisted_models = set(json.load(f))
        else:
            _blacklisted_models = set()
        print(f"[BLACKLIST] Loaded {len(_blacklisted_models)} model(s) from {_BLACKLIST_FILE}.")
    except Exception as e:
        print(f"[BLACKLIST] Error loading blacklist: {e}")
        _blacklisted_models = set()


def save_blacklist() -> None:
    """Persist the current blacklist to disk."""
    os.makedirs(os.path.dirname(_BLACKLIST_FILE), exist_ok=True)
    try:
        with open(_BLACKLIST_FILE, "w") as f:
            json.dump(sorted(_blacklisted_models), f, indent=2)
        print(f"[BLACKLIST] Saved {len(_blacklisted_models)} model(s) to {_BLACKLIST_FILE}.")
    except Exception as e:
        print(f"[BLACKLIST] Error saving blacklist: {e}")


def add_to_blacklist(model_name: str) -> None:
    """Add *model_name* to the blacklist and persist it."""
    global _blacklisted_models
    _blacklisted_models.add(model_name)
    save_blacklist()
    print(f"[BLACKLIST] Added \"{model_name}\" to blacklist.")

# ---------------------------------------------------------------------------
# Performance-based auto-blacklist (check against global peak)
# ---------------------------------------------------------------------------
_PERF_THRESHOLD = 1 / 3   # blacklisted if baseline < 1/3 * peak
_PERF_MIN_SAMPLES = 30      # need > 30 samples before trusting the peak


def _write_priority_model(model: str) -> None:
    """Write the priority model to a file for zombie.py to consume."""
    if not model:
        return
    config_dir = os.path.dirname(_CONFIG_PATH)
    os.makedirs(config_dir, exist_ok=True)
    with open(_PRIORITY_MODEL_PATH, "w") as f:
        f.write(model)
    print(f"[MGMT] Wrote priority model to {_PRIORITY_MODEL_PATH}: {model}")


def _check_slopotrigger() -> dict | None:
    try:
        if os.path.exists(_TRIGGER_PATH):
            with open(_TRIGGER_PATH, "r") as f:
                data = json.load(f)
            os.remove(_TRIGGER_PATH)
            return data
    except Exception as e:
        print(f"[SLOPE-MON] Error checking trigger file: {e}")
    return None

def _decide_action(triggered_model: str, trigger_throughput: float) -> str:
    try:
        baselines_data = context_prober.load_performance_baselines()
        all_baselines = baselines_data.get("baselines", {})
        local_entry = all_baselines.get(triggered_model, {})
        local_slope = local_entry.get("slope")
        
        if local_slope and trigger_throughput < 0.5 * local_slope:
            return "restart_ollama"
    except Exception as e:
        print(f"[SLOPE-MON] Error in _decide_action: {e}")
    return "blacklist_and_restart"


_server_process = None
_stop_event = threading.Event()
_mgmt_stop_event = threading.Event()
_ollama = None
_model_manager = None
_shutdown_requested = False


def _load_config():
    """Load configuration from ~/.thinkfarm/config.ini.

    If config.ini is missing, defaults to 30GB with model management turned on.
    Priority: Config File > Environment Variable > Missing-config defaults > System ID
    """
    config = configparser.ConfigParser()
    config.read(_CONFIG_PATH)

    # Defaults: used when config.ini is missing, and as fallback when it exists
    auto_manage = True
    managed_storage_gb = "50"
    ollama_url = "http://127.0.0.1:11434"
    ollama_restart_cmd = ""
    provider_id = None  # will fall through to env/system ID

    # Priority: Config File > Environment Variable > (Missing) defaults > System ID > UUID
    provider_id = os.environ.get("PROVIDER_ID", "") or \
                  (config.get("provider", "provider_id", fallback=None) if config.has_section("provider") else None) or \
                  provider_id or \
                  str(uuid.uuid4())
    ollama_url = os.environ.get("OLLAMA_URL", "") or ollama_url
    managed_storage_gb = os.environ.get("MANAGED_STORAGE_GB", "") or managed_storage_gb
    auto_manage_str = os.environ.get("AUTO_MANAGE", "").lower()
    auto_manage = auto_manage_str in ("true", "1", "yes") if auto_manage_str else auto_manage

    if config.has_section("provider"):
        provider_id = config.get("provider", "provider_id", fallback=provider_id)
        ollama_url = config.get("provider", "ollama_url", fallback=ollama_url)
        managed_storage_gb = config.get("provider", "managed_storage_gb", fallback=managed_storage_gb)
        auto_manage = config.getboolean("provider", "auto_manage", fallback=auto_manage)
        ollama_restart_cmd = config.get("provider", "ollama_restart_cmd", fallback=ollama_restart_cmd)

    return {
        "provider_id": provider_id,
        "ollama_url": ollama_url,
        "managed_storage_gb": managed_storage_gb,
        "auto_manage": auto_manage,
        "ollama_restart_cmd": ollama_restart_cmd,
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

            # Fetch and write slopemon state
            try:
                import httpx
                from datetime import datetime
                with httpx.Client(timeout=15.0) as client:
                    resp = client.get("https://www.thinkfarm.net/api/performance")
                    resp.raise_for_status()
                    perf_data = resp.json()

                state = {"updated_at": datetime.now().isoformat(), "models": {}}
                for entry in perf_data.get("data", []):
                    name = entry.get("model", "")
                    peak = entry.get("peak", 0)
                    n_samples = entry.get("n", 0)
                    if n_samples > _PERF_MIN_SAMPLES and peak > 0:
                        state["models"][name] = {"peak": round(peak, 1)}

                with open(_SLOPEMON_DATA_PATH, "w") as f:
                    json.dump(state, f, indent=2)
            except Exception as e:
                print(f"[MGMT-LOOP] Error fetching /performance for slopemon: {e}")

            if auto_manage and limit_gb > 0:
                # Run optimization
                newly_pulled, priority_model = asyncio.run(_model_manager.optimize_portfolio(limit_gb))

                _write_priority_model(priority_model)

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


def _restart_ollama(cmd_raw, url):
    if not cmd_raw:
        print("[RECOVERY] Error: Ollama restart command is not configured in config.ini")
        return False
    import shlex
    try:
        cmd = shlex.split(cmd_raw)
    except ValueError as e:
        print(f"[RECOVERY] Invalid Ollama restart command syntax: {e}")
        return False
    
    print(f"[RECOVERY] Restarting Ollama via: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[RECOVERY] Error restarting Ollama: {e}")
        return False
    
    import httpx
    print("[RECOVERY] Polling /api/tags for Ollama recovery...")
    tags_url = f"{url.rstrip('/')}/api/tags"
    
    for _ in range(30):
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(tags_url)
                if resp.status_code == 200:
                    print("[RECOVERY] Ollama is back up!")
                    return True
        except Exception:
            pass
        time.sleep(2)
            
    print("[RECOVERY] Timed out waiting for Ollama to come back.")
    return False


def _start_service_sync():
    """Start zombie.py subprocess. Called from management loop."""
    global _server_process
    if _server_process is not None:
        print("[STARTUP] Service is already running.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "zombie.py")
    project_root = os.path.dirname(script_dir)

    _server_process = subprocess.Popen(
        [sys.executable, "-u", script_path],
        cwd=project_root,
    )
    print(f"[STARTUP] zombie.py started (PID { _server_process.pid })")


def _stop_service_sync():
    """Stop zombie.py subprocess. Called from management loop."""
    global _server_process
    if _server_process is None:
        return
    _stop_event.set()
    print("[STOP] Initiating server shutdown...")
    _server_process.terminate()
    try:
        _server_process.wait(timeout=180)
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
    if not os.path.exists(_CONFIG_PATH):
        print("[CONFIG] Using defaults: 30GB storage, model management ON (no config.ini found)")

    # Load blacklist
    load_blacklist()

    # 1. Performance-based auto-blacklist check
    print("[STARTUP] Fetching global performance data...")
    perf_data = {"data": []}
    try:
        import httpx as _httpx_sync
        with _httpx_sync.Client(timeout=15.0) as client:
            resp = client.get("https://www.thinkfarm.net/api/performance")
            resp.raise_for_status()
            perf_data = resp.json()
        print(f"[STARTUP] Global performance data loaded for "
              f"{len(perf_data.get('data', []))} model(s).")
    except Exception as e:
        print(f"[STARTUP] Could not fetch performance data ({e}). "
              "Skipping auto-blacklist check.")

    # Compare each known model against the global peak
    baselines_data = context_prober.load_performance_baselines()
    all_baselines = baselines_data.get("baselines", {})

    blacklisted_any = False
    for entry in perf_data.get("data", []):
        model_name = entry.get("model", "")
        if not model_name:
            continue
        global_peak = entry.get("peak", 0)
        n_samples = entry.get("n", 0)

        local_entry = all_baselines.get(model_name, {})
        local_slope = local_entry.get("slope")
        if local_slope is None:
            continue

        if n_samples > _PERF_MIN_SAMPLES and \
           local_slope < _PERF_THRESHOLD * global_peak:
            print(f"[BLACKLIST-PERF] {model_name}: "
                  f"baseline={local_slope:.1f} tokens/s "
                  f"< {_PERF_THRESHOLD*100:.0f}% of global peak={global_peak:.1f} "
                  f"(n={n_samples}) => blacklisting.")
            add_to_blacklist(model_name)
            blacklisted_any = True

    if blacklisted_any:
        print(f"[BLACKLIST-PERF] Models auto-blacklisted by performance. "
              f"{len(_blacklisted_models)} total now:")
        for m in sorted(_blacklisted_models):
            print(f"  - {m}")
    else:
        print("[STARTUP] No models auto-blacklisted by performance.")

    # 2. Check for unscanned models
    url = config["ollama_url"].strip() or "http://127.0.0.1:11434"
    print(f"[STARTUP] Checking for unscanned models at {url}...")

    all_models = asyncio.run(context_prober.get_ollama_models(url))
    existing_limits = context_prober.load_context_limits()
    unscanned = [m for m in all_models if m not in existing_limits]

    if unscanned:
        print(f"[STARTUP] Found {len(unscanned)} unscanned model(s): {unscanned}")
        print("[STARTUP] Running context probing...")
    else:
        print("[STARTUP] All models already scanned. Verifying custom models...")

    # Run probing & custom model verification on all models
    asyncio.run(context_prober.run_context_probing(
        url,
        all_models,
        existing_limits
    ))
    print("[STARTUP] Probing and custom model verification complete.")

    # 3. Initialize managed model state
    _ollama = OllamaClient()
    _model_manager = ModelManager(_ollama, os.environ.get("SERVER_URL", "https://app.thinkfarm.net"))

    # 4. Start zombie.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "zombie.py")
    project_root = os.path.dirname(script_dir)

    _stop_event.clear()
    _server_process = subprocess.Popen(
        [sys.executable, "-u", script_path],
        cwd=project_root,
    )
    _write_pid()
    print(f"[STARTUP] zombie.py started (PID {_server_process.pid})")

    # 5. Start managed model loop in background
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
        while not _shutdown_requested:
            if _server_process is not None:
                rc = _server_process.poll()
                if rc is not None:
                    if not _stop_event.is_set():
                        print(f"\nzombie.py exited with code {rc}")
                        if rc == 42:
                            trigger_data = _check_slopotrigger()
                            if trigger_data:
                                model = trigger_data.get("model", "unknown")
                                throughput = trigger_data.get("throughput", 0.0)
                                action = _decide_action(model, throughput)
                                
                                print(f"[SLOPE-MON] Model '{model}' underperformed vs global peak. Actioned: {action}")
                                
                                if action == "restart_ollama":
                                    if _restart_ollama(config.get("ollama_restart_cmd", ""), config["ollama_url"]):
                                        print("[SLOPE-MON] Recovery successful. Restarting zombie...")
                                        _server_process = None
                                        _start_service_sync()
                                        continue
                                    else:
                                        print("\n\nServer process ended unexpectedly due to failed Ollama restart.")
                                        break
                                else:
                                    add_to_blacklist(model)
                                    print(f"[SLOPE-MON] Model {model} blacklisted. Restarting zombie...")
                                    _server_process = None
                                    _start_service_sync()
                                    continue
                        
                        print("\n\nServer process ended unexpectedly.")
                        break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nReceived interrupt. Shutting down...")
        if _server_process is not None:
            _stop_service_sync()
        _remove_pid()
        _shutdown_requested = True


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
