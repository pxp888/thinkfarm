# thinkfarm Provider — Architecture Overview

Two scripts work together:

| File | Role |
|------|------|
| `baseprovider.py` | **GUI Manager** — PyQt6 app that starts/stops the provider, handles config, model management, and blacklist UI. |
| `solo.py` | **Worker** — headless async process that maintains a WebSocket to the central server, accepts inference jobs, runs them on Ollama, and streams results back. |

---

## baseprovider.py (GUI Manager) — Pseudocode

```
1. ON STARTUP:
   a. Load config from ~/.thinkfarm/config.ini  (provider_id, ollama_url, restart_cmd, storage_limit, auto_manage flag)
   b. Load blacklist from disk -> _blacklisted_models set
   c. Start background thread: _managed_model_loop()

2. BACKGROUND LOOP (_managed_model_loop), every 1 hour:
   a. If auto_manage is enabled:
      i. Fetch global performance data from https://www.thinkfarm.net/api/performance
      ii. Write model peaks to ._slopemon.json (used by solo.py for baseline thresholds)
     iii. Call model_manager.optimize_portfolio(storage_gb_limit) -> returns {newly_pulled_models, priority_model}
      iv. Write priority_model to ~/.thinkfarm/priority_model.txt  (solo.py will load it into VRAM)
       v. If new models were pulled:
          - Stop provider service
          - Run context probing on the new models
          - Restart provider service

3. USER CLICKS "START":
   a. Launch background thread _startup_logic():
      i.  Fetch global performance data from server
     ii.  For each model with enough samples:
            if local_baseline < 0.33 * global_peak -> add_to_blacklist(model)
    iii. Get all Ollama models via httpx
     iv.  Load existing context limits from disk
      v.  If any unscanned/newly probed models exist: emit UI state "probing"
     vi.  Run context probing on ALL models (updates baseline slopes)
    vii. Emit signal trigger_actual_start -> calls _actual_start_service()

4. ACTUAL START (_actual_start_service):
   a. Spawn solo.py as a subprocess:
      - If frozen (PyInstaller): run self with "--solo" flag
      - Else: run python solo.py
   b. Poll subprocess: if it stays alive for 5s -> emit UI state "started"
   c. While subprocess runs, loop:
      - Check for ZERO_EVAL file -> if present, terminate process, emit "stopped_error"
      - If subprocess exits with code 42 (performance fault):
          * Read solo_slope_trigger.json for {model, throughput}
          * Decide action: restart_ollama or blacklist_and_restart
          * Apply fix and re-launch solo.py

5. USER CLICKS "STOP":
   a. Set stop event, terminate subprocess
   b. Wait up to 180s for graceful finish; if timeout -> force kill
   c. Emit UI state "stopped"

6. SAVE SETTINGS:
   a. Validate provider_id (non-empty), ollama_url (http/https), storage_gb (numeric)
   b. Write to ~/.thinkfarm/config.ini under [provider] section
   c. Update live Ollama client URLs in-memory

7. UI STATE MACHINE:
   "stopped" -> Start clicked -> "probing" (if models scanned) -> "started"
   "started" -> Stop clicked  -> "stopping" -> "stopped"
   Error paths: "stopped_error" (zero-eval fault)
```

---

## solo.py (Worker) — Pseudocode

```
1. ON STARTUP (async main()):
   a. Load config from .env + ~/.thinkfarm/config.ini  (PROVIDER_ID, SERVER_URL, SERVER_WS_URL)
      - If no PROVIDER_ID configured, generate UUID and persist it
   b. Load cached gpu_context_limits from disk -> {model: num_ctx}
   c. Load blacklist from disk -> _blacklisted_models set
   d. Load slopemon performance data:
      for each model in _slopemon_data_path:
          _slope_peer_thresholds[model] = peak / 3
   e. Create OllamaClient instance
   f. Launch WebSocket connection loop: asyncio.create_task(connect_to_server())
   g. Launch heartbeat task: asyncio.create_task(_heartbeat_loop())
   h. Wait for SIGINT/SIGTERM shutdown signal

2. SOLO WORKER LOOP (per job):
   
   A. CONNECT TO SERVER (connect_to_server):
      - Persistent WebSocket loop to wss://app.thinkfarm.net/ws/provider/{PROVIDER_ID}
      - On connect: send initial status, launch periodic_status_updates(), listen_for_messages()
      - On disconnect: sleep 5s and retry

   B. PERIODIC STATUS UPDATES (every 30s or on event trigger):
      a. Get current models + loaded models from Ollama
      b. Filter out those with gpu_context_limit=0 or in _blacklisted_models
      c. Compare against previous status
      d. Send {"type":"status", ...} if set changed, otherwise send ping

   C. RECEIVING SERVER MESSAGES:
      case "job_published":
         - Model is known locally? -> Yes
         - Context fits (gpu_context_limits)? -> Yes
         - _try_accept_job() -> sends {"type":"accept"}

      case "job_assigned":
          start background task: execute_job_with_heartbeat_reset()

      case "cancel_job":
          - Cancel active_tasks[job_id]

   D. EXECUTE JOB (execute_job):
      a. is_busy += 1
      b. With execution_lock (only one job at a time):
         i.   Map endpoint (chat/generate/embed/v1/...) to Ollama API path
       ii.   Resolve model: if gpu_context_limit > 0, re-write model name as "thinkfarm-{model}" 
            iii. If streaming:
              - Collect tokens from Ollama stream into 75ms batches
              - Flush batch as {"type":"chunk","job_id":...,"data":...} via WebSocket
       iv.   If non-streaming: single POST, send result as chunk
        v.   Extract usage stats (prompt_eval_count, eval_count) from response
      c. Send {"type":"job_done", "eval_count": N, ...} to server
      d. Compute actual_throughput = eval_count / seconds
      e. Zero-eval check: if eval_count == 0 -> launch _run_zero_eval_inspection(model)

   E. SLOPE MONITOR (during job_done):
      - If throughput < threshold for consecutive jobs:
          _consecutive_bad[model] += 1
      - If >= _TRIGGER_COUNT (3) consecutive bad:
          Write solo_slope_trigger.json {model, throughput, threshold}
          Set global stopping = True -> triggers shutdown and exit code 42

   F. ZERO-EVAL INSPECTION (_run_zero_eval_inspection):
      a. Loop with lock (only one active at a time)
      b. Generate "hello" to the suspicious model
      c. If succeeds: break out, return to service
      d. If fails: write ZERO_EVAL file -> wait 60s -> retry

3. HEARTBEAT LOOP (_heartbeat_loop):
   a. Every 120s: check if priority model needs loading into VRAM
   b. Every 20 min (idle, no jobs): send minimal "hello" generate to {loaded_model} to keep it warm

4. RECONNECT LOGIC:
   - Count completed jobs since last reconnect: job_counter
   - At job_counter >= 30 and is_busy == 0: close WebSocket -> triggers connect_to_server() retry

5. EXIT (main()):
   a. On shutdown signal: set stopping = True, wait for all active jobs to finish
   b. Close Ollama client
   c. If _consecutive_bad values >= 3 for any model: return exit code 42
   d. Else: return exit code 0
```

---

## How They Work Together

```
┌──────────────────────┐         subprocess          ┌──────────────────────┐
│                      │    python solo.py           │                      │
│  baseprovider.py     │--------------------------->│   solo.py            │
│  (GUI Manager)       │                             │   (Worker)           │
│                      │<--- exit code 0/42 ---------│                      │
└──────────────────────┘                             └──────────────────────┘
        |                                                  |
        |  manages                                       |  persistent WebSocket
        |  configuration                                 |  wss://app.thinkfarm.net
        |  blacklist UI                                  |         ^
        |  model portfolio optimization                /|----------|
        |  context probing                             / |  HTTP/WSS
        \                                          (thinkfarm) server
         \_________________________________________/

Shared state files between the two:
- ~/.thinkfarm/priority_model.txt        (baseprovider -> solo: which model to hot-load)
- ~/.thinkfarm/blacklisted_models.json   (both read/write: blacklisted models)
- ~/.thinkfarm/gpu_context_limits.json   (both read: context length limits per model)
- ~/.thinkfarm/_slopemon.json            (baseprovider writes, solo reads: performance baselines)
- ~/.thinkfarm/solo_slope_trigger.json   (solo -> baseprovider: performance fault trigger)
- ~/.thinkfarm/ZERO_EVAL                 (solo -> baseprovider: zero-eval failure marker)
```
