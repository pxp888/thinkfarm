# Thinkfarm Provider Protocol

This document describes the data exchange protocol between a **provider** (a machine running an Ollama server that offers inference) and the **central thinkfarm server**. The primary communication channel is a WebSocket; the server also exposes a set of REST APIs.

---

## 1. Connection

### Establishing the Link

The provider opens a persistent WebSocket connection to:

```
wss://app.thinkfarm.net/ws/provider/{provider_id}
```

- `{provider_id}` — a unique string identifying this provider instance (user-configured or auto-generated UUID).
- The server validates whether the provider is blocked; if so, it sends an error message and closes the connection. Duplicate `provider_id` values are rejected.

On successful connect the provider immediately begins its **status → job → response** lifecycle. Disconnection triggers automatic reconnection on the provider side after a 5-second delay. The server retains all routing state for 10–60 minutes after last contact, so transient drops do not lose jobs that were dispatched but not yet accepted.

---

## 2. Message Types

All messages travel as JSON over the WebSocket and contain at minimum:

```json
{"type": "…" }
```

Message types are separated into **Provider → Server** and **Server → Provider**.

### 2.1 Provider → Server Messages

#### `status` — Periodic provider status report

Sent once on connect (initial status) and then every ~30 seconds (or sooner when triggered by state changes).

```json
{
  "type": "status",
  "provider_id": "my-device-abc123",
  "connected_at": "2026-06-23T14:00:00.000Z",
  "models": [
    {"name": "llama3.2:latest"},
    {"name": "mistral:nemo"}
  ],
  "loaded_models": ["llama3.2:latest"],
  "context_limits": {
    "llama3.2:latest": 8192,
    "mistral:nemo": 4096
  },
  "is_busy": false
}
```

| Field | Type | Description |
|---|---|---|
| `provider_id` | string | Same ID used in the WebSocket URL. Echoed for consistency. |
| `connected_at` | string (ISO-8601) | Timestamp when this connection was established. |
| `models` | array of strings or objects | All Ollama models available on this machine (both loaded and unloaded). Objects carry extra metadata; strings are bare names. |
| `loaded_models` | array | Models currently resident in VRAM. |
| `context_limits` | object (optional) | `{ model_name: max_num_ctx }` discovered by context probing. A value of `0` means the provider can handle that model but has effectively no GPU context capacity — the server will not route context-heavy jobs. A value of `-1` means "unlimited". |
| `is_busy` | boolean | `true` if there is at least one queued or running inference job. |

The server uses this to:
- Maintain per-model routing **idle** and **all** sets (via Redis sorted sets).
- Track provider presence via a 10-minute TTL heartbeat (refreshed on every message).

#### `accept` — Job acceptance

Sent when the provider agrees to handle an incoming job (after seeing its initial dispatch/published message).

```json
{
  "type": "accept",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "provider_id": "my-device-abc123"
}
```

| Field | Type | Description |
|---|---|---|
| `job_id` | string (UUID) | The job the provider is accepting. Only one provider's acceptance wins; others that also send `accept` for the same job are ignored. |
| `provider_id` | string | Echoed back to the server so it knows who accepted. |

#### `chunk` — Inference data chunk

Sent by the provider to deliver inference outputs back to the server (and ultimately to the consumer). Chunks are always sent as standard JSON messages over the WebSocket. Tokens from Ollama are batched in ~75 ms intervals.

```json
{
  "type": "chunk",
  "job_id": "550e8400-...",
  "data": "data: \"content\":\"Hello\"\n\n"
}
```

| Field | Type | Description |
|---|---|---|
| `job_id` | string (UUID) | The job this chunk belongs to. |
| `data` | string | One or more SSE lines (for stream) or a full JSON response object (for non-stream). Tokens are batched in ~75 ms intervals to reduce message count. The `"C:<length>:" ` prefix is used **only** between server instances over Redis pub/sub, not on the provider WebSocket. |

#### `job_done` — Job completion notice

Sent when the provider finishes a job. Provides usage / timing stats and the provider's busy state.

```json
{
  "type": "job_done",
  "job_id": "550e8400-...",
  "prompt_eval_count": 128,
  "eval_count": 340,
  "total_duration": 875000000,
  "is_busy": false
}
```

| Field | Type | Description |
|---|---|---|
| `job_id` | string (UUID) | The completed job. |
| `prompt_eval_count` | integer | Number of prompt tokens evaluated. |
| `eval_count` | integer | Number of output tokens generated. |
| `total_duration` | integer (nanoseconds) | Wall-clock time in nanoseconds for the full inference. |
| `is_busy` | boolean | Whether the provider still has other active/queued jobs after this one completes. |

The server uses these fields to:
- Persist usage stats to the database (for billing & performance analytics).
- Clear the provider's busy flag and restore it to the idle model sets when `is_busy` is `false`.
- Publish a sentinel (`"C:__DONE__"`) on the job channel to signal stream termination to the consumer side.

---

### 2.2 Server → Provider Messages

#### `job_published` — Job announcement (dispatch)

Sent to idle or all-known providers when a new inference request arrives. It tells the provider a job is available for a particular model and invites acceptance.

```json
{
  "type": "job_published",
  "job_id": "550e8400-...",
  "model": "llama3.2:latest"
}
```

| Field | Type | Description |
|---|---|---|
| `job_id` | string (UUID) | Unique job identifier. |
| `model` | string | The Ollama model name the request requires. Must match one of the provider's registered models and pass the context-limit check to be accepted. |

The server broadcasts this to up to 14 randomly-selected idle providers for a given model, then to up to 7 fallback providers if no idle provider accepts within 2 seconds.

#### `job_assigned` — Concrete job assignment with payload

Sent after an acceptance has been won. This carries the full inference request.

```json
{
  "type": "job_assigned",
  "job_id": "550e8400-...",
  "endpoint": "v1/chat/completions",
  "body": {
    "model": "llama3.2:latest",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "stream": true,
    "options": {},
    "stream_options": {"include_usage": true}
  }
}
```

| Field | Type | Description |
|---|---|---|
| `job_id` | string (UUID) | Matches the published job. |
| `endpoint` | string | The internal Ollama API endpoint to use (`/api/chat`, `/api/generate`, `/api/embed`, `/api/show`, `/v1/chat/completions`, etc.). Maps to one of: `"chat"`, `"generate"`, `"embed"`, `"embeddings"`, `"show"`, `"v1/chat/completions"`, `"v1/completions"`, `"v1/responses"`. |
| `body` | object | The complete inference request payload, as the consumer sent it. Includes `model`, `messages`, `stream`, `options`, etc. For streaming requests the provider **must add** `stream_options: { "include_usage": true }` if absent. Context-limited models are auto-remapped to a `thinkfarm-{model}` prefix internally. |

#### `cancel_job` — Inference cancellation

Sent to stop an ongoing job (e.g., because a timeout expired and the server is trying another provider).

```json
{
  "type": "cancel_job",
  "job_id": "550e8400-..."
}
```

| Field | Type | Description |
|---|---|---|
| `job_id` | string (UUID) | The job to cancel. |

The provider aborts the corresponding asyncio task and cleans up local state. No further chunks should be sent for that job.

#### Error message

Sent when a provider is blocked or has another fatal error.

```json
{
  "type": "error",
  "detail": "Provider is blocked."
}
```

| Field | Type | Description |
|---|---|---|
| `detail` | string | Human-readable error description (e.g., permanent block reason or temporary ban duration). |

---

## 3. Job Lifecycle (WebSocket)

The full flow from consumer request to response:

```
[Consumer/API] → Central Server ←→ [Provider WebSocket] → Local Ollama
```

1. **Dispatch** — The server publishes `job_published` to random idle/fallback providers with just the `model` name. Each provider checks if it has that model and whether the context limit allows the job's `num_ctx`.

2. **Accept** — A provider that can handle the job sends back an `accept` message with its `provider_id` and the `job_id`. Only the first acceptance wins; others are silently discarded. If no one accepts within 2s (idle pool) or 10s total, the server tries fallback providers.

3. **Assign** — The server publishes a `job_assigned` message carrying the full request body to the winning provider via the dedicated per-job Redis pub/sub channel.

4. **Execute & Stream chunks back** — The provider forwards the `body` to its local Ollama instance (through the mapped endpoint) and streams response chunks back as `chunk` messages, batched in ~75 ms windows. For direct-streamed jobs, chunks are piped internally via a HTTP POST with length-prefixed delivery (`<length>:<data>`).

5. **DONE** — The provider sends `job_done` with timing / token counts. The server publishes the `C:__DONE__` sentinel to end the consumer's stream and saves usage data to PostgreSQL.

6. **Busy state** — During a job, the provider is marked busy in Redis. A status message with `is_busy: true` prevents re-routing new jobs until completion. Once `is_busy` returns `false`, the provider is restored to idle sets.

7. **Cancellation / Timeout** — If an accepted provider fails to produce the first chunk within 90 seconds, the server sends `cancel_job`, removes it from consideration, and retries with another provider (up to 3 total attempts). After a final successful chunk arrives, streaming continues until `job_done`.

8. **Periodic reconnection** — Every 30 completed jobs while idle, the provider proactively closes and reconnects its WebSocket for a fresh routing state.

---

## 4. REST API Endpoints (Consumer / Dashboard → Server)

These endpoints are not used by the provider but are part of the full system protocol. They route inference requests to providers using the WebSocket job lifecycle above.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Chat completion. Routes to an idle/fallback provider's `/api/chat` endpoint. |
| `POST` | `/api/generate` | Text generation. Routes to a provider's `/api/generate`. |
| `POST` | `/api/embeddings` | Embedding vector generation. Routes to `/api/embeddings`. |
| `POST` | `/api/embed` | Legacy embedding endpoint. Routes to `/api/embed`. |
| `POST` | `/api/show` | Model metadata inspection. Cached in Redis for 30 days. |
| `POST` | `/api/target` | Targeted inference: specifies a provider_id directly (bypasses idle selection). |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat endpoint. |
| `POST` | `/v1/completions` | OpenAI-compatible text completion endpoint. |
| `POST` | `/v1/responses` | OpenAI-compatible responses endpoint. |
| `GET` | `/v1/models` | Returns list of all models from all providers in OpenAI list format. |
| `GET` | `/api/providers` | Lists all currently registered providers (status snapshot at time of query). |
| `GET` | `/api/tags` | Unified model library — one entry per distinct model name across all providers. |
| `GET` | `/api/ps` | Models that are currently loaded in VRAM across all providers. |
| `GET` | `/api/performance` | Dominant-mode throughput (`peak` tokens/s) per model computed from the last 48 hours of job data, using GMM clustering on non-fallback jobs. Cached for 30 minutes. Used by providers' auto-blacklist logic. |
| `GET` | `/api/demandchart` | Revenue and unique provider counts per model over the last 24 hours. Cached for 5 minutes. |
| `GET` | `/api/randomprovider?model=...` | Returns a random active provider that supports a given model. |
| `GET` | `/api/version` | Server version string (currently `"0.1.32"`). |
| `GET` | `/stats` | Aggregate stats: number of providers, number busy, per-model counts. |
| `GET` | `/` | Health check: returns `{"status": "ok"}`. |
| `POST` | `/api/internal/jobs/{job_id}/stream` | Internal endpoint for receiving direct-streamed chunks from provider-side HTTP POST (length-prefixed format). Returns `200 OK` when the stream completes. |
| `POST` | `/api/admin/block-provider` | Admin: block (permanent or temporary) or unblock a provider. JSON body `{ "provider_id", "length", "reason" }`. |

### Authentication

- Consumer-facing endpoints (`/api/chat`, `/v1/...`) require either an `Authorization: Bearer <consumer_id>` header or an `X-Consumer-ID` header.
- The admin endpoint is not described as having auth in this file (admin-only access assumed from deployment context).
- The WebSocket server-side rejects connections without checking consumer auth; the provider itself carries no credentials beyond its `provider_id`.

---

## 5. Internal Data Formats & Conventions

### Endpoint Mapping

The provider internally translates these endpoint keys to Ollama API paths:

| Provider endpoint key | Ollama path |
|-----------------------|-------------|
| `chat` | `/api/chat` |
| `generate` | `/api/generate` |
| `embed` | `/api/embed` |
| `embeddings` | `/api/embeddings` |
| `show` | `/api/show` |
| `v1/chat/completions` | `/v1/chat/completions` |
| `v1/completions` | `/v1/completions` |
| `v1/responses` | `/v1/responses` |

### Streaming Format

- **Ollama-native** (`chat`, `generate`): NDJSON lines, one per SSE line. Separator in the WebSocket message: `\n`.
- **OpenAI-compatible** (`v1/...`): SSE format with `data:` prefix. Separator: `\n\n`. `[DONE]` sentinel is stripped internally.
- When `stream_options.include_usage` is enabled, usage stats arrive embedded in the JSON payload with a `usage` key containing `prompt_tokens` and `completion_tokens`.

### Context Remapping

If a provider has a cached GPU context limit for a model (value > 0), the server remaps the model name to `thinkfarm-{model_name}` and clears `num_ctx` from options before sending it to the provider. The provider then uses a custom Ollama model with that fixed context size on inference.

### Busy-State TTL

The server marks a provider as busy in Redis for 600 seconds (10 minutes) upon assignment. This is refreshed by each status message and cleared when `job_done` reports `is_busy: false`.

---

## 6. Provider Health & Performance Monitoring

Beyond the standard job protocol, providers self-monitor performance using a built-in **Slope Monitor**:

- On startup (or every hour while running), the provider fetches `/api/performance` from the central server — which returns dominant-mode throughput per model across all participants.
- For each model, a threshold is computed as `global_peak / 3`. If the provider's own throughput for that model drops below this threshold for 3 consecutive job completions (with total duration ≥ 10s), the model is **blacklisted** locally.
- The blacklist is persisted to `~/.thinkfarm/blacklisted_models.json` and the model is removed from future inference routing on this provider.
- A sub-threshold check also considers whether throughput fell below 50% of the known baseline, triggering an attempt to restart Ollama before resorting to blacklisting.

### Zero-Evaluation Detection

If a provider completes a job with `eval_count == 0` (no output tokens generated) it automatically tests the model by running a small `"hello"` generation:
- If successful, it resumes normal operation.
- If it fails, it writes a trigger file (`~/.thinkfarm/ZERO_EVAL`) and waits 60 seconds for Ollama to recover; repeated failures (more than 3 triggers) cause the service to stop with exit code `42`.

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Normal shutdown (graceful or via SIGTERM). |
| `1` | Configuration error (missing PROVIDER_ID, etc.). |
| `42` | Persistent model failure — too many consecutive zero-evaluation failures. |

---

## 7. Reconnection & Failover

- **Provider-side**: On WebSocket disconnect, the provider retries after 5 seconds. If it detects a planned reconnection (every 30 jobs), it closes proactively before retrying.
- **Server-side**: Provider status keys expire in 10 minutes. Routing sets (idle / all) expire in 1 hour. Thus providers that disappear are automatically removed from the pool within those windows.
- **Job failover**: If an accepted provider fails to produce its first chunk within 90 seconds, it is blocked for the remainder of that consumer request, a cancel message is sent, and the server retries up to 3 total attempts across different providers.