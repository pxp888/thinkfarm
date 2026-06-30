# thinkfarm Consumer (qclient.py) — Architecture Overview

`qclient.py` is a **PyQt6 GUI** that wraps the `main.py` FastAPI app into a desktop application. It gives users control over their local consumer server — a proxy layer that sits between *providers* and a local Ollama instance, enforcing a model whitelist so only approved models are forwarded.

---

## Pseudocode

```
1. ON STARTUP:
   a. Load SERVER_URL from .env (Ollama server address)
   b. Load consumer_id and port from ~/.thinkfarm/config.ini [consumer] section
       - If no config exists, defaults: empty ID, port 11434
   c. Setup UI widgets, tray icon, signal connection
   d. On delayed tick (100ms): fetch available models from Ollama via /api/tags

2. MODEL WHITELIST MANAGEMENT:
   
   FETCH MODELS:
      Thread runs: requests.get(f"{SERVER_URL}/api/tags") -> parse model list
      Emit models_loaded signal with parsed list
   
   POPULATE UI:
      For each model name from Ollama:
          if matches current filter text (case-insensitive):
              create QCheckBox(name)
              store in self.model_vars[name]
   
   LOAD WHITELIST FROM DISK (~/.thinkfarm/config.ini [consumer]):
      whitelist_enabled -> cb.checked
      whitelist_models  (comma-separated string) -> mark matching checkboxes
   
   SELECT ALL / NONE:
      Iterate model_vars, set each checkbox to True/False
   
   FILTER CHANGED:
      Trigger refresh_models() -> re-fetches from Ollama
   
   SAVE WHITELIST TO DISK:
      enabled = self.whitelist_enabled_cb.isChecked()
      selected = [name for name, cb in model_vars.items() if cb.checked]
      Write to config.ini under [consumer]:
          whitelist_enabled = true/false
          whitelist_models = "model1,model2,..."

3. START SERVICE:
   a. Load port from config (default 11434)
   b. Create uvicorn.config with:
        app=fastapi_app (from main.py)
        host="0.0.0.0", port=<configured>, log_level="info"
   c. Start uvicorn.Server in a daemon thread
   d. Poll local port every 100ms via socket connect check:
      - Try 127.0.0.1, ::1, 0.0.0.0 on the port (timeout 10ms each)
      - If any succeeds -> UI switches to "SYSTEM RUNNING" (green dot)
      - Timeout at 300 polls (30s) -> silent failure

4. STOP SERVICE:
   a. Set server.should_exit = True
   b. Join server_thread with 2s timeout
   c. Set server = None
   d. UI switches to "SYSTEM STOPPED" (gray dot)

5. CONSUMER ID MANAGEMENT:
   SAVE:
      Validate non-empty string, write to config.ini [consumer] consumer_id
   
   LOAD:
      Read from config.ini [consumer] consumer_id, display in status label

6. SERVER PORT MANAGEMENT:
   SAVE:
      Validate integer 1-65535, write to config.ini [consumer] port
   
   LOAD:
      Read from config.ini [consumer] port, populate entry field

7. SYSTEM TRAY:
   a. Icon loaded from "thinkfarm.webp" in same dir (fallback to standard icon)
   b. Menu: "Restore" -> show window; "Exit" -> close + stop_service
   c. Double-click tray icon -> restore_window()
   d. When minimized -> auto-hide to tray

8. UI STATE MACHINE:
   "stopped"  (gray dot, red text)
      -> Start Client clicked -> polling port...
      -> port responds -> "running" (green dot)
   
   "running"  (green dot, green text)
      -> Stop Client clicked -> set should_exit -> joined -> back to "stopped"

9. CLOSE EVENT:
   If server running -> stop_service() before window closes
```

---

## What qclient.py Does NOT Do Directly

The actual HTTP proxy logic lives in **`main.py`** (FastAPI app started by uvicorn). `qclient.py` does not handle any of the request/response routing itself — it only:

- Launches/tears down the uvicorn server
- Provides the config UI (consumer ID, port, model whitelist)
- Displays running/stopped status
- Manages system tray presence

The proxy in `main.py` uses the whitelist to decide whether to forward or block incoming provider requests for a given model name, acting as a gate between providers and the local Ollama instance.

---

## Key Architectural Points

```
┌──────────────────────────────────────────────┐   uvicorn (port 11434)    ┌─────────────────┐
│             qclient.py (GUI)                  │<------------------------>│  main.py (FastAPI)│
│                                              │                           │  (proxy app)     │
│  • Start/Stop uvicorn server                 │<------------------------>│                  │
│  • Model whitelist UI (checkboxes)            │   filtered requests       │  whitelist gate  │
│  • Consumer ID config                          │                            │  logic           │
│  • Port config                                 │                           └────────┬─────────┘
│  • System tray support                         │                                    │
│  • Config persistence to config.ini            │                       forward / block   │
│                                              │                                    ▼
└──────────────────────────────────────────────┘                          ┌─────────────────┐
                                                                         │  Ollama server |
                                                                         │  (local/remote) │
                                                                         └─────────────────┘
```

Shared config: `~/.thinkfarm/config.ini` section `[consumer]`:
| Field             | Value                                  |
|-------------------|----------------------------------------|
| consumer_id       | User-set client identifier string      |
| port              | Local HTTP listening port (default 11434) |
| whitelist_enabled | `true` or `false`                      |
| whitelist_models  | Comma-separated model names            |
