# thinkfarm Provider Protocol

A reference for how the `unity` application works and how it communicates with the
central thinkfarm server and the local Ollama instance.

## 1. Overview

thinkfarm is a client for a distributed LLM inference-sharing network. One process
plays (or can simultaneously play) two roles:

- **Consumer (Client Proxy)** — a local, Ollama-compatible HTTP API that forwards
  requests to the central server (`https://app.thinkfarm.net`). Any application can
  point its Ollama/OpenAI base URL at `http://127.0.0.1:11435` (default) and the
  request is routed to whichever provider node in the pool serves that model.
- **Provider (Inference Node)** — a WebSocket client that registers the local
  machine's Ollama models with the central server, accepts routed jobs, runs them on
  the local Ollama, and streams tokens back.

Additionally, the node self-manages: it probes models to discover the largest
GPU-safe context window, creates shadow "custom" models with that window baked in,
monitors its own throughput, and (optionally) auto-pulls/deletes managed models to
maximize demand coverage within a storage budget.

### File map

| File | Role |
|---|---|
| `main.py` | Entry point. Creates the Qt application and the `ThinkfarmApp` window. `multiprocessing.freeze_support()` for PyInstaller. |
| `app_gui.py` | PyQt6 GUI (`ThinkfarmApp`): dashboard, config panels, model whitelist, log pane, tray icon. Hosts the two worker threads (`ClientThread`, `ProviderThread`). On Windows it also *manages* Ollama as a child subprocess on a random free port. |
| `client_server.py` | The Consumer side: a FastAPI app (`create_client_app`) exposing an Ollama-compatible API and proxying everything to the central server with an `X-Consumer-ID` header. |
| `provider_client.py` | The Provider side: `ProviderClient` (WebSocket job engine) and `OllamaClient` (HTTP helper wrappers for the local Ollama server). |
| `context_prober.py` | Discovery of the max GPU-safe context per model (binary search over `num_ctx`), performance baselining, and creation of `thinkfarm-` shadow models. |
| `model_manager.py` | `ModelManager`: automated portfolio of *managed* models — picks models to pull/delete based on demand, opportunity score, VRAM fit, and estimated tokens/s. |
| `config.py` | `ConfigManager`: loads/saves `~/.thinkfarm/config.ini` (with migrations from the old `client` section and dotenv-style files), overlays a local `.env`, and holds defaults. |
| `headless.py` | Non-GUI mode: runs only the `ProviderClient` with signal-based shutdown. |

### Persistent state (all under `~/.thinkfarm/`)

| File | Written by | Contents |
|---|---|---|
| `config.ini` | `config.py` | `[provider]` and `[consumer]` sections (IDs, URLs, whitelist, budget, context pressure, …). |
| `gpu_context_limits.json` | `context_prober.py` | `model_name -> max GPU-safe num_ctx` (`-1` = embedding model, `0` = no GPU / failed). |
| `performance_baselines.json` | `context_prober.py` | `model_name -> {slope, samples, last_probed}` baseline throughput in tokens/s. |
| `blacklisted_models.json` | `provider_client.py` | Models permanently excluded from the advertised set (e.g. failed sanity checks). |
| `managed_models_names.json` | `model_manager.py` | Names of models this node itself pulled (managed set). |
| `user_models.json` | `model_manager.py` | Accumulated set of user-owned models (never deleted by the manager). |
| `_slopemon.json` | *(server-side/shared data, read if present)* | Network peak performance per model, used for slow-job thresholds. |
| `ollama_internal.log` | `app_gui.py` | Logs of the Windows-managed child Ollama process. |

---

## 2. Process and threading model

```
main thread (Qt event loop)
├── ThinkfarmApp (GUI, config, tray)
├── ClientThread  ── own asyncio loop + uvicorn server on 0.0.0.0:<port>   (consumer proxy)
├── ProviderThread ── own asyncio loop running ProviderClient.run()        (provider)
├── background helper threads (model refresh, Ollama readiness polling, offline restarts)
└── child process (Windows only): ollama serve on 127.0.0.1:<random free port>
```

Key rules:

- Qt signals cross thread boundaries (`LogEmitter` converts Python `logging` records
  into a `log_received(msg, level)` signal so the log pane updates safely; `QTimer.singleShot(0, …)` re-enters the Qt thread).
- GUI buttons toggle `ClientThread` (uvicorn `should_exit = True` to stop) and
  `ProviderThread` (`soft_stop()` vs `force_stop()` executed via
  `asyncio.run_coroutine_threadsafe`).
- On **Windows** the app is the *owner* of Ollama (`managed_ollama = True`): it
  spawns `ollama serve` with `OLLAMA_HOST=127.0.0.1:<free_port>`, waits for
  `/api/tags` to respond (up to 30 s), and points `local_ollama_url` at it. On
  Linux/macOS it just talks to whatever the user configured (default
  `http://localhost:11434`).

---

## 3. Consumer side: the Ollama-compatible proxy (`client_server.py`)

A FastAPI app exposed locally (default port **11435**) with CORS fully open. It
mimics the Ollama API surface so standard clients work unmodified:

| Local endpoint | Proxied to (central server) | Body transformation |
|---|---|---|
| `GET  /api/tags` | `GET  /api/tags` | model whitelist filter |
| `GET  /api/ps` | `GET  /api/ps` | model whitelist filter |
| `GET  /v1/models` | `GET  /v1/models` | whitelist filter (on `data[].id`) |
| `POST /api/chat` | `POST /api/chat` | **`num_ctx` injection** |
| `POST /api/generate` | `POST /api/generate` | **`num_ctx` injection** |
| `POST /api/embeddings`, `POST /api/embed` | same | **`num_ctx` injection** |
| `POST /api/show` | `POST /api/show` | none |
| `POST /v1/chat/completions`, `/v1/completions`, `/v1/responses` | same | none |
| `GET  /`, `/version`, `/api/version` | local responses | — |

Transformation rules:

1. **Identity** — every proxied request carries the header
   `X-Consumer-ID: <consumer_id>` (from config) and `Content-Type: application/json`.
   The central server uses this to account attribution.
2. **Whitelist** — if `whitelist_enabled` is set, responses from `/api/tags`,
   `/api/ps` (field `models[].name`) and `/v1/models` (`data[].id`) are filtered
   down to `whitelist_models`. This is also used by the GUI to decide which models
   this consumer wants to see.
3. **`num_ctx` injection** (`inject_num_ctx`) — if the caller did not set a
   positive `options.num_ctx`, the proxy estimates it:
   `tokens ≈ (prompt text length) / 3`, adds a buffer (`options.num_predict` or
   `max_tokens`, else 2048), and rounds **up to the next 1024** (minimum 4096),
   then writes `options.num_ctx` into the body. This keeps providers from
   allocating oversized KV caches.
4. **Streaming** — if the (already-parsed) body has `stream: true`, the proxy opens
   a streaming upstream request and relays raw bytes (`resp.aiter_bytes()`),
   choosing the media type by path: `text/event-stream` for `v1/` (OpenAI SSE)
   otherwise `application/x-ndjson` (Ollama NDJSON). Error responses (≥ 400) are
   buffered and re-sent as JSON with the original status code. Any upstream failure
   becomes `502 "Failed to connect to central thinkfarm server"`.

The shared `httpx.AsyncClient` (timeout 60 s) is created on the FastAPI `startup`
hook and closed on `shutdown`.

**End user → tokens flow:**

```
App (base URL :11435) → proxy → central server → (routing) → provider WebSocket
                                                            → provider's local Ollama
                                                            → tokens stream back
```

---

## 4. Provider side: the WebSocket protocol (`provider_client.py`)

`ProviderClient.run()` is a reconnection loop. Each iteration:

1. Optionally runs portfolio optimization (hourly, if `auto_manage_models` and
   `gb_allowed > 0`) via `ModelManager.optimize_portfolio()`.
2. Checks `check_for_new_models()` — if any local model lacks a context limit, a
   baseline, or its `thinkfarm-` shadow variant, it first soft-stops the
   connection (drains jobs) and runs the **context probing** routine
   (`context_prober.run_context_probing`), then continues.
3. On first startup with no model loaded, fires `load_most_desirable_model()`
   (loads the highest-revenue/least-competition model it hosts, using the central
   `/api/demandchart`).
4. Connects the WebSocket:
   `wss://<host>/ws/provider/<provider_id>` (`ws` for http servers), with
   `ping_interval=20`, `ping_timeout=20`.
5. Immediately sends a **full status** (see below), then runs a heartbeat task
   every 5 s: re-check for new models, send status or `{"type":"ping"}`, and
   (only when idle) a 20-minute idle-keepalive that pings loaded Ollama models
   with a tiny `"hello"` generate to keep them in VRAM (`keep_alive: -1`).

### 4.1 Provider → server messages

| Type | When | Payload |
|---|---|---|
| `status` | on connect, after model/job changes, every 30 completed jobs, and when the announced model set or ≥ 20 min has changed (heartbeat otherwise sends `ping`) | see below |
| `ping` | heartbeat when nothing changed | `{}` |
| `accept` | a new `job_published` is accepted (model is local, probed, not blacklisted, and requested `num_ctx` within advertised limit) | `job_id`, `provider_id` |
| `chunk` | streamed Ollama output, batched in ~75 ms windows (or the whole non-stream response as one chunk) | `job_id`, `data` (raw NDJSON bytes concatenated; SSE lines for `v1/` endpoints; model name de-aliased back to the base name) |
| `job_done` | job finished | `job_id`, `prompt_eval_count`, `eval_count`, `total_duration` (ns), `is_busy` |

The full **status** message:

```json
{
  "type": "status",
  "provider_id": "<uuid>",
  "connected_at": "2025-01-01T00:00:00Z",
  "models": [ /* Ollama /api/tags entries, names stripped of the thinkfarm- prefix, thinkfarm- derivatives deduped, only probed + non-blacklisted */ ],
  "loaded_models": ["llama3.2", "…"],
  "context_limits": { "llama3.2": 46080, "bge-m3": -1, "…": 8192 },
  "is_busy": false
}
```

`context_limits` semantics: `limit > 0` → advertised as `int(limit * 0.9)` (the
`context_pressure` safe fraction, i.e. 90 % of the probed max); `limit == -1` →
embedding / unlimited; otherwise fallback `8192`.

### 4.2 Server → provider messages

| Type | Meaning | Provider behavior |
|---|---|---|
| `job_published` | *Advertisement*: this job **may** be handled here | Validate model locally + context limit; optionally delay 1.5 s if busy / 0.5 s if model not loaded; then send `accept` and optimistic `status(is_busy=true)`. Jobs for models we don't have are silently ignored (the server will route to someone else). |
| `job_assigned` | The server committed the job to us after our `accept` | Increment job counters, pre-load the model with an empty keep-alive call, run `execute_job` as a background task, send `status`. |
| `cancel_job` | Abort the job | `task.cancel()` the Ollama request; counters reset in `finally`. |
| `error` | Server-side error | Logged only. |
| (text / binary WS frames from Ollama) | — | All job I/O goes out as JSON envelopes above. |

So the protocol is a **bidirectional handshake**: `job_published` → `accept` →
`job_assigned`. The server decides acceptance winners among multiple providers and
tells the winning provider to execute exactly once.

### 4.3 Job execution details (`execute_job`)

- Endpoint translation: `chat/generate/embed/embeddings/show` →
  `/api/…`; `v1/chat/completions`, `v1/completions`, `v1/responses` → `/v1/…`.
- **Model aliasing**: for non-embedding endpoints the base model is internally
  mapped to its shadow variant `thinkfarm-<model>` (which has the correct
  `num_ctx` baked in via `num_ctx_train`), the Ollama response stream is scanned
  for the alias and rewritten back to the base model name, and non-stream JSON has
  its `model` field restored. Embedding endpoints use the plain model (probed
  shadow models still exist but are bypassed since embeddings don't need context).
- Forcing: `stream_options.include_usage = true` on all streams (so token counts
  are available), `keep_alive: -1` on every request (models stay resident in VRAM
  forever).
- Streaming path: `aiter_bytes()` relayed into a buffer; complete frames are parsed
  *only* to extract `eval_count` / `eval_duration` / usage stats; every 75 ms (or
  at end-of-stream) the accumulated raw text is flushed as one `chunk` message.
- The Ollama HTTP timeout is 120 s.

### 4.4 State machine & lifecycle flags

```
stopped ──start──▶ connecting ──ws open──▶ connected
                                          │
                 probing ◀── new model detected (drain jobs, close ws)
                 restart ◀── sanity-check failure or severe slowdown (drain jobs)
  stopping (soft_stop) ── all jobs drained ──▶ stopped
  force_stop ──▶ immediate (cancel jobs, close ws)
```

Important invariants:

- `current_jobs` / `active_jobs` — in-flight job accounting; `soft_stop` blocks
  until it hits 0 and then closes the socket, and *never* accepts new
  `job_published` ads.
- Every job exit (success/cancel/failure) ends with `send_status(force_full=True)`
  so the server's view of `is_busy` is always accurate.
- After **30 successful jobs** the WebSocket is deliberately closed and the
  reconnect loop re-announces status ("refresh routing").
- Reconnect backoff on transport errors: 5 s.

### 4.5 Self-healing subsystems (all inside `monitor_performance`)

After each non-embedding job:

1. **Zero-eval detection** — `eval_count == 0` *and* no output seen → run a
   sanity check (`/api/generate`, prompt `"hello"`, up to 300 s).
   - Pass → fine.
   - Fail → trigger `restart_ollama()`: soft-stop jobs, drop the socket, run
     `ollama_restart_cmd` (or the managed Windows subprocess restart), poll
     `/api/tags` for up to 3 min, run a test prompt on the most desirable model
     and verify the measured "slope" (`(eval_count + 0.003·prompt_eval_count) /
     compute_seconds`, tokens/s) is ≥ 70 % of the stored baseline. Any failure
     leaves the node in an explicit error status (`Error: Ollama offline`,
     `Error: Test prompt failed`, `performance_degraded`).
2. **Slope monitor** — keep the last 5 `(tokens/s, duration)` samples per model.
   If 3 consecutive jobs ≥ 10 s are all below `peak/3` (the "slopemon" global,
   from `https://www.thinkfarm.net/api/performance` or `_slopemon.json`):
   - measured ≥ 70 % baseline → **blacklist** the model locally
     (`blacklisted_models.json`), unload it, re-announce, resume.
   - measured < 70 % baseline → treat as hardware issue → `restart_ollama()`.
   - no baseline → blacklist by default.

---

## 5. Context probing (`context_prober.py`)

Purpose: find, per model, the largest `num_ctx` that keeps the whole model
(weights + KV cache) in GPU VRAM, then pin it into a derived Ollama model.

Algorithm (per model, in `_find_max_gpu_ctx`):

1. **Classify** via `/api/show`:
   - `is_embed` is true when the model has no chat template, lists `embedding` in
     capabilities, or is from a BERT/nomic-bert family. Embedded models short-circuit
     to `limits[model] = -1` (no binary search needed — CPU spill is acceptable and
     there is no KV-cache OOM risk).
   - `is_eligible` is false for cloud models, `:cloud` tags, and `thinkfarm-*`
     derivatives (those become `limits[model] = 0`, announced as unavailable).
   - `upper_bound` = the largest `*.context_length` / `num_ctx_train` / Modelfile
     `num_ctx` found in `model_info`, else `262 144`. On NVIDIA an *analytical*
     KV cap is also computed: `(VRAM − 1.5 GiB − weights) / (2·l·h_kv·d·2 bytes)`.
2. **Binary search** `num_ctx` in `[512, upper_bound]`, `≤ 10` iterations,
   probing with a tiny prompt (`"Hi"`, `num_predict: 1`) through `/api/generate`
   (or `/api/embed` for embedding models).
3. **GPU-only test**:
   - Linux/Windows: poll `/api/ps` (≤ 20 × 1.5 s) and compare
     `size` vs `size_vram` — equal ⇒ fully on GPU (OK), different ⇒ CPU
     spillover (fail).
   - macOS/Apple Silicon unified memory: instead verify ≥ 2 GiB free system RAM
     (`psutil` → `vm_stat` → `/proc/meminfo`) after the load.
4. If any probe attempt errors and `/api/tags` is dead, `OllamaConnectionError`
   aborts the whole pass (crash-safe: partial results are already saved).
5. **Baseline**: `_probe_performance_baseline` runs 3 identical non-stream
   `/api/generate` calls (`"what is the history of Sweden?"`, `num_ctx = best_ctx`,
   temperature 0.2) and stores the average slope in `performance_baselines.json`.
6. After each model: `save_context_limits()` (crash-safe), `keep_alive: 0`
   unload, then next model.

At the end (and on every load of the cache via `load_context_limits`),
**`thinkfarm-` shadow models** are (re)created for every eligible local model with
`limit > 0` via `/api/create`:

```json
{ "model": "thinkfarm-llama3.2",
  "from": "llama3.2",
  "parameters": { "num_ctx": "<limit * context_pressure>" },
  "stream": false }
```

`context_pressure` (default 0.9, GUI slider, `[provider] context_pressure` in
`config.ini`) is the safe operating fraction of the discovered max. Orphaned
`thinkfarm-` models whose base model was deleted are cleaned up via
`/api/delete`.

---

## 6. Model portfolio management (`model_manager.py`)

Runs hourly when `auto_manage_models = true` and `gb_allowed > 0`, from the
provider loop (and is also the only writer of the managed-model manifest on disk).

Pipeline of `optimize_portfolio(limit_gb)`:

1. **Partition models** — *user models* = local tags − manifest (accumulate into
   `user_models.json`, never touch); *managed* = the manifest.
2. **Hardware discovery** — total VRAM via `nvidia-smi` / `rocm-smi` / AMD sysfs /
   `amdsmi` / macOS `sysctl hw.memsize`, and an estimated memory bandwidth from a
   GPU-name → GB/s lookup table (falls back to VRAM-size heuristics if
   unrecognised).
3. **Demand chart** — `GET /api/demandchart` (from central server, or
   `https://app.thinkfarm.net` when on thinkfarm.net) → per-model `revenue` and
   `providers` counts.
4. **Per-candidate suitability** (`_calculate_model_suitability`) — from the
   central server's `/api/show` payload: parse `parameter_count`,
   `block_count`, `head_count(_kv)`, `key_length`, detect MoE
   (`expert_count`/`expert_used_count`, ~15 % non-expert / 85 % expert split),
   and compute:
   - `fits_in_vram = (weights + kv_cache(8192 ctx, FP16)) · 1.10 ≤ VRAM`
   - `estimated_tps ≈ 0.75 · (bandwidth / active_weight_bytes·1.10 + kv)`
     (roofline decode estimate, MoE-aware).
   - Skip if it doesn't fit, or if macOS isn't running a Mac-native variant.
5. **Opportunity ranking** — `opportunity = revenue / max(1, providers)`,
   **× 1.2 stickiness** for models already in the manifest (hysteresis to
   prevent churn), sorted descending.
6. **Quota fill** — greedy knapsack of candidates into `gb_allowed` (sorted by
   opportunity); then decide which low-opportunity manifest models to remove so
   the set fits, filling leftover space with retained manifest models if they
   fit.
7. **Execution gate** — if *any* removal is needed, skip the whole cycle with an
   80 % probability (deliberate anti-churn "coin flip").
8. **Apply** — `DELETE /api/delete` for removals, stream `POST /api/pull`
   (verifying the model actually appears in `/api/tags` afterwards), update
   `managed_models_names.json`. Newly pulled models return up so the outer loop
   context-probes them.
9. A **priority model** (best opportunity among user + target sets that also have a
   positive context limit) is returned for post-restart verification.

---

## 7. Configuration (`config.py`)

- File: `~/.thinkfarm/config.ini`, sections `[provider]` and `[consumer]`
  (auto-migrated from a legacy single `[client]` section; also tolerant of
  dotenv files without section headers).
- Local `.env` overlay for `CENTRAL_SERVER_URL`, `CONSUMER_ID`, `CLIENT_PORT`,
  `WHITELIST_*`.
- Defaults: `server_url = https://app.thinkfarm.net`, `provider_id = <uuid4>`,
  `port = 11435`, `local_ollama_url = http://localhost:11434`,
  `context_pressure = 0.9`, `whitelist_enabled = false`.
- `provider_id` is the node's stable identity (used in the WebSocket URL and
  every `status`/`accept`/`job_done` envelope); `consumer_id` is only required
  for the consumer proxy path.

## 8. End-to-end request walkthrough (consumer → provider)

```
1.  Client POST http://localhost:11435/api/chat  {model:"llama3.x", prompt, stream:true}
2.  client_server.py: adds X-Consumer-ID, injects options.num_ctx (≈ prompt_tokens
    + predict_buffer, rounded up to 1024s), opens an upstream stream to
    https://app.thinkfarm.net/api/chat, relays bytes back as NDJSON.
3.  Central server picks a provider hosting "llama3.x" (based on last status push)
    and pushes over wss://…/ws/provider/<id>:
        {type:"job_published", job_id, model, num_ctx?, body:{model,prompt,stream,…"options":{num_ctx}}}
4.  provider_client.handle_message: confirms "llama3.x" ∈ local probed set, not
    blacklisted, and num_ctx ≤ advertised 90 % limit. Sends {type:"accept"}.
5.  Server: {type:"job_assigned", job_id, endpoint:"chat", body:{…}}
6.  execute_job: rewrites body.model → "thinkfarm-llama3.x", sets keep_alive=-1
    and include_usage, POST streams to http://127.0.0.1:<ollama_port>/api/chat.
7.  Ollama NDJSON chunks accumulate; every ~75 ms a chunk envelope is sent:
        {type:"chunk", job_id, data:"<raw ndjson lines>"}
    (thinkingfarm- name string-replaced back to llama3.x on the fly)
8.  Ollama finishes; final chunk flushed; then
        {type:"job_done", job_id, eval_count, prompt_eval_count, total_duration, is_busy}
    counters reset, forced full status push, performance monitors run.
9.  (server-side: routed bytes back down to the consumer proxy and on to the app)
```

If the node instead needs to *learn* a newly pulled model first, step 4-5 never
happen: `check_for_new_models` → context probe (binary search + baseline) →
`thinkfarm-` custom model created → status re-announced → model is now eligible.

## 9. Quick endpoint cheat-sheet

| Endpoint | From | Purpose |
|---|---|---|
| `GET /api/tags`, `GET /api/ps`, `GET /v1/models`, `GET /api/show` | proxy or provider | model inventory & metadata (local Ollama or central) |
| `POST /api/chat`, `/api/generate`, `/api/embeddings`, `/api/embed`, `/api/show` | proxy / provider | inference jobs |
| `POST /v1/chat/completions`, `/v1/completions`, `/v1/responses` | proxy | OpenAI-compatible inference |
| `POST /api/pull`, `POST /api/create`, `DELETE /api/delete` | provider (model mgmt / prober) | model lifecycle |
| `wss://<host>/ws/provider/<provider_id>` | provider ↔ server | the main protocol described in §4 |
| `GET /api/performance`, `GET /api/demandchart` | provider | global peaks (slopemon) and demand chart for opportunity ranking / startup hot-load |
| `127.0.0.1:<free>/api/tags` (Windows) | GUI / provider | the app-managed child Ollama instance |
