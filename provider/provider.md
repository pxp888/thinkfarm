# thinkfarm Provider Launcher (standalone)

A PyQt6 GUI front-end that starts/stops an AI model serving provider via `solo.py` (no FastAPI). Configuration and status management only — no log output in the UI.

## Control Flow

### 1. APPLICATION LAUNCH

```
App instantiation → ProviderGUI.__init__()

    ├── restart_ollama_server()
    │     • Read config for ollama_models_path
    │     • Pick a free port (non-11434)
    │     • Spawn: Popen(["ollama", "serve"])
    │     • Set self._ollama_url = http://127.0.0.1:{port}
    │     • Update OllamaClient HTTP base URL
    │
    ├── setup_ui()
    │     • Build sidebar (logo, start/stop/save buttons)
    │     • Build config card (ID, model path, storage, auto)
    │     • Build status bar (dot + text)
    │
    ├── _load_config()
    │     • Read ~/.thinkfarm/config.ini
    │     • Populate UI fields from config
    │     • Remove stale ollama_url config if present
    │
    ├── setup_tray()
    │     • Create system tray icon + Restore/Exit menu
    │     • Double-click → restore window
    │     • SystemTrayAvailable → close hides to tray
    │
    └── _managed_model_loop Thread (daemon)
          • Poll: service running? NO → sleep 2s, continue
```

### 2. USER CLICKS "START PROVIDER"

```
start_service() [main thread]
    ├── Disable start_btn
    ├── Clear _stop_event
    └── Spawn background thread: _startup_logic()

        _startup_logic() [background thread]
            ├── get_ollama_models(url)
            ├── load_context_limits()
            ├── unscanned = all_models - existing_limits
            
            if unscanned:
                emit "probing"
                run_context_probing(unscanned)
            
            emit trigger_actual_start [slot → UI thread]

        _actual_start_service() [UI thread]
            if frozen (PyInstaller):
                Popen([sys.executable, "--solo"])
            else:
                Popen([python, "solo.py"])

            Spawn check_startup() [polling thread]
                Loop 50× (≤5s):
                    poll _server_process alive?
                    YES → emit "started", return
                    stop_event set? → return
                    NO → sleep(0.1), retry

    UI state "started":
        start_btn disabled (grey)
        stop_btn enabled (grey)
        status green ● SYSTEM RUNNING
```

### 3. USER CLICKS "STOP PROVIDER" (or closes window with running)

```
stop_service() [main thread]
    ├── _stop_event.set()
    ├── server_process.terminate()
    └── emit "stopping" → orange ● FINISHING JOBS

    wait_join() [thread]
        server_process.wait(timeout=180s)
            timeout? → kill() + wait()
        _server_process = None
        emit "stopped"
            start_btn enabled (green)
            stop_btn disabled (grey)
            status grey ● SYSTEM STOPPED
```

### 4. AUTO-MANAGED MODEL LOOP [daemon background thread, every 1 hour]

```
_managed_model_loop()
    loop forever:
        service running? NO → sleep(2s), continue
        
        YES → read config
            auto_manage = true AND limit_gb > 0?
                YES → optimize_portfolio(limit_gb)
                    _write_priority_model(priority_model)
                    
                    newly_pulled models?
                        NO → sleep(1h), continue
                        
                        YES:
                            stop_service()
                            while server_process is not None: sleep(1s)
                            
                            run_context_probing(new_models)
                            
                            start_service()
        
        sleep(3600s) [poll every 2s for responsiveness]
```

### 5. WINDOW CLOSED (with running provider)

```
closeEvent():
    ├── server_process? → stop_service()
    ├── _ollama_process? → terminate + wait(5s)
    └── _ollama_log_file? → close
```

### 6. MINIMIZE

```
changeEvent() minimized:
    hide() window → appears in system tray only
```

### 7. SAVE SETTINGS

```
save_config():
    ├── Validate: provider_id non-empty
    ├── Validate: storage is numeric
    └── Errors? → show warning
    
    write to ~/.thinkfarm/config.ini
    restart_ollama_server()
    → show "Settings saved" success box
```

### 8. CLI ARGUMENT DISPATCH (if __name__ == "__main__")

```
sys.argv flags:
    --solo        → asyncio.run(solo.main()); exit(0)
    --probe <...> → context_prober.main(); exit(0)
    (none)        → run the full GUI as described above
```

### 9. PRIORITY MODEL FILE (producer → consumer IPC)

**File:** `~/.thinkfarm/priority_model.txt` — single disk file shared between provider.py and solo.py.

| producer                      | consumer (solo.py)                       |
|-------------------------------|------------------------------------------|
| `optimize_portfolio()`        | `_ensure_priority_model_loaded():`       |
| determines priority model     | 1. Read & cache model name from disk    |
| async.run(...)                | 2. Check: is VRAM empty?                 |
| `write_priority_model(x)` ←──│   NO → skip (already loaded)             |
|                               |   YES → `load_model(priority_model)`    |

**VRAM load rule:** Priority model only enters VRAM when no other model is currently resident. This is enforced in `_ensure_priority_model_loaded()` which queries Ollama's loaded models list first — if anything is loaded, it returns without touching the priority model.

**Fallback role during heartbeat:** When checking whether to send a keepalive ping, solo.py uses `model = _load_priority_model() or _last_model`. So it also serves as the **second-tier fallback** after `_last_model` (the user's most recent interactive model) — not before it.

**Priority of model resolution during job processing:** `_last_model` → `_load_priority_model()` → error. The priority model is the last resort, only used when no other model is already loaded or available.
