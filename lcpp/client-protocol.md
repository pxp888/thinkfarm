# thinkfarm Consumer Protocol

This document describes the **network protocol** between the consumer (client) application and the central thinkfarm server. It covers what the client sends to the server, what the server sends back, and the overall request/response flow.

---

## 1. Configuration

### 1.1 Client Settings (stored locally in `~/.thinkfarm/config.ini`)

| Key | Description |
|-----|-------------|
| `consumer_id` | Unique identifier for this consumer client |
| `port` | Local port on which the consumer exposes its Ollama-compatible server (default: 11434) |
| `whitelist_enabled` | Whether model whitelisting is active (`true`/`false`) |
| `whitelist_models` | Comma-separated list of allowed model names |

### 1.2 Authentication Header

Every inference request from the consumer to the server includes:

```
X-Consumer-ID: <consumer_id>
```

This header identifies which registered consumer is making the request. It is used by the server for logging, rate-limiting, and routing decisions (e.g., idle vs. busy provider tracking).

---

## 2. Consumer as a Local Server (Ollama-Compatible)

The consumer app **runs its own local HTTP server** (via FastAPI/Uvicorn) on the configured port. It exposes an Ollama-compatible API that any local application can call. The consumer then **proxies** every request to the central server over HTTPS.

### 2.1 Exposed Endpoints (local, served by the consumer)

All endpoints accept the same format as a standard Ollama server (`ollama <host>:<port> ...`).

#### GET Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Health check. Returns `{ "status": "ok", "consumer_id": "<id>" }` |
| `GET /version` or `GET /api/version` | Model catalog version |
| `GET /api/tags` | Full list of available models (with optional whitelist filtering) |
| `GET /api/ps` | List of currently loaded models across the provider network |
| `GET /v1/models` | OpenAI-compatible model list |

#### POST Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /api/chat` | Chat completion (streaming or non-streaming) |
| `POST /api/generate` | Raw text generation (streaming or non-streaming) |
| `POST /api/embeddings` | Embedding generation (non-streaming) |
| `POST /api/embed` | Batch embed request (non-streaming) |
| `POST /api/show` | Get model details info |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |
| `POST /v1/completions` | OpenAI-compatible text completions |
| `POST /v1/responses` | OpenAI-compatible responses endpoint |

All of these endpoints forward the **entire request body** to the corresponding central server endpoint, with one transformation: if the endpoint is not `show`, the consumer may inject a `num_ctx` option into the request body if it is missing.

---

## 3. Client-to-Server Protocol (HTTP → Central Server)

All communication from the consumer to the central server occurs over HTTPS at a configured URL (default: `https://app.thinkfarm.net`). The central server exposes both Ollama-compatible paths (`/api/*`) and OpenAI-compatible paths (`/v1/*`).

### 3.1 Model Catalog Requests

#### `GET /api/tags`

**Consumer sends:**
```
GET {SERVER_URL}/api/tags
```

**Server returns:**
```json
{
  "models": [
    { "name": "llama3.2" },
    { "name": "gemma:7b" },
    ...
  ]
}
```

#### `GET /api/ps`

**Consumer sends:**
```
GET {SERVER_URL}/api/ps
```

**Server returns:** A list of currently loaded models with full metadata (sizes, parents, formats, etc.) from the provider network.

#### `GET /v1/models`

**Consumer sends:**
```
GET {SERVER_URL}/v1/models
```

**Server returns (OpenAI format):**
```json
{
  "object": "list",
  "data": [
    { "id": "llama3.2", "object": "model", "created": <timestamp>, "owned_by": "library" }
  ]
}
```

#### `GET /api/version`

**Consumer sends:**
```
GET {SERVER_URL}/api/version
```

**Server returns:**
```json
{ "version": "0.1.32" }
```

### 3.2 Inference Requests

For all inference endpoints, the consumer:

1. Receives a request from a local caller (e.g., `POST /api/chat`).
2. Reads and optionally transforms the body (injecting `num_ctx` if needed).
3. Sends a **single HTTPS POST** to the central server with:

| Header | Value |
|--------|-------|
| `Content-Type` | `application/json` |
| `X-Consumer-ID` | `<consumer_id>` |
| **Body** | The original (or transformed) request JSON, unchanged in structure |

#### Chat Completion (`POST /api/chat`)

**Request body** (Ollama format):
```json
{
  "model": "<model_name>",
  "messages": [
    { "role": "user", "content": "Hello" }
  ],
  "stream": true,
  "options": { ... }
}
```

**Server response (streaming):**
- If `stream: true`: a stream of SSE-like chunks of type `text/event-stream` or `application/x-ndjson`.
  - Each chunk contains a partial JSON object (e.g., an openai-style delta).
- If `stream: false`: a single JSON body with the full completion response.

#### Raw Generation (`POST /api/generate`)

**Request body:**
```json
{
  "model": "<model_name>",
  "prompt": "Generate text",
  "stream": true
}
```

Responds identically to chat — streaming or non-streaming JSON depending on the `stream` flag.

#### Embeddings (`POST /api/embeddings` or `POST /api/embed`)

**Request body:**
```json
{
  "model": "<embedding_model>",
  "input": ["Hello world"]
}
```

**Response (non-streaming):**
```json
{
  "model": "<model>",
  "embeddings": [[...]]
}
```

#### Model Info (`POST /api/show`)

**Request body:**
```json
{ "name": "<model_name>" }
```

**Response:** Full model metadata (size, parent models, format, family, licenses, etc.). Cached by the server for 30 days.

#### OpenAI-Compatible Endpoints

The consumer forwards these **unchanged** to:
- `POST /v1/chat/completions` → `{SERVER_URL}/v1/chat/completions`
- `POST /v1/completions` → `{SERVER_URL}/v1/completions`
- `POST /v1/responses` → `{SERVER_URL}/v1/responses`

Same request/response contract as above; the consumer does not transform these bodies. The server detects OpenAI-compat by endpoint name and handles SSE media-type (`text/event-stream`) accordingly.

### 3.3 Error Responses

When the server returns an HTTP error status (4xx/5xx), the consumer reads the response body, attempts to parse it as JSON for a `detail` field, and raises an equivalent HTTP exception back to the local caller. The consumer does **not** stream error details — the full error is returned at once.

---

## 4. Server-to-Client Summary (Central Server Infrastructure)

While the consumer primarily interacts via HTTP, the central server's internal protocol involves:

### 4.1 Provider ↔ Server (WebSocket)

The server brokers jobs to GPU providers (not consumers). When a consumer inference request arrives:

1. The server looks up idle/fallback providers for the requested model.
2. It **pub/subs via Redis** on a per-job channel (`llamashare:job:<job_id>`).
3. A provider accepts, receives an `assign` message with the full request body.
4. Chunks flow back from the provider to the server over the same job-specific Redis pub/sub channel or a direct HTTP POST (length-prefixed streaming).
5. Once complete, a `__DONE__` sentinel is published and the job is logged.

The consumer **does not participate** in this WebSocket/pub/sub protocol; it simply issues an HTTP request and receives back whatever the server streams.

### 4.2 Server Query Endpoints (No Authorization Required from Consumer's Perspective)

| Endpoint | Purpose |
|----------|---------|
| `GET /api/providers` | Full list of all registered providers and their status |
| `GET /api/randomprovider?model=...` | A random active provider for a given model |
| `GET /stats` | Aggregate stats: number of providers, how many are busy, model counts |
| `GET /api/demandchart` | 24-hour revenue and provider utilization per model |
| `POST /api/internal/jobs/<job_id>/stream` | Internal endpoint for piped streaming chunks between server tasks |

The consumer does not call these endpoints directly — they are used by admin dashboards and other internal tools.

---

## 5. Request Flow Diagram (Simplified)

```
┌──────────┐         HTTPS POST         ┌─────────────┐
│ Local    │   X-Consumer-ID header     │ Central     │
│ Caller   │ ─────────────────────────► │ Server      │
│(app, CLI)│                            │ (HTTP/SSE)  │
└──────────┘                            └──────┬──────┘
                                               │
                                               │ Redis Pub/Sub
                                               ▼
                                      ┌─────────────────┐
                                      │ Provider Pool   │
                                      │ (WebSocket)     │
                                      └─────────────────┘
```

1. **Local caller** → sends request to the consumer's local server (`localhost:<port>`).
2. **Consumer** → forwards it as an HTTPS POST to `{SERVER_URL}/{endpoint}`, with `X-Consumer-ID` header.
3. **Server** → routes to an idle GPU provider via Redis pub/sub broker. Provider streams chunks back through the server.
4. **Server** → sends chunks back over HTTP (streaming) or a single JSON response (non-streaming).
5. **Consumer** → pipes the stream unchanged to the local caller. The consumer is a transparent proxy.

---

## 6. Streaming Details

The consumer preserves streaming fidelity:

- It opens a connection to the server with `stream=True` in the httpx client.
- It reads each chunk from the server's HTTP response (`response.aiter_bytes()`).
- It yields each chunk immediately to the local caller via a FastAPI `StreamingResponse`.
- No buffering or reformatting occurs on the consumer side — chunks pass through verbatim.

The actual format of streamed chunks depends on the endpoint:

| Endpoint prefix | Chunk format |
|-----------------|-------------|
| `/api/*` (Ollama) | `application/x-ndjson` (newline-delimited JSON) |
| `/v1/*` (OpenAI) | `text/event-stream` (SSE-style) |

---

## 7. Model Whitelisting

When the consumer has whitelisting enabled, it filters the list of models returned by `/api/tags`, `/api/ps`, and `/v1/models` **locally** before passing the result back to the caller. The filtering logic:

- Reads `whitelist_enabled` and `whitelist_models` from `~/.thinkfarm/config.ini`.
- Strips any model from the server's response whose name does not appear in the whitelist.
- No whitelisting information is sent to or from the central server.

---

## 8. Request Transformation (num_ctx Injection)

For all inference endpoints except `/api/show`, if the `options.num_ctx` field is missing or empty in the request body, the consumer estimates it:

1. Sums the text length of all `prompt` or `messages[].content` fields.
2. Estimates tokens as `estimated_prompt_tokens = text_length // 3`.
3. Adds a buffer equal to either `options.num_predict`, `body.max_tokens`, or `2048`.
4. Ensures the final value is at least `4096`, rounded up to the nearest `1024`.
5. Injects the calculated `num_ctx` into `options` before forwarding to the server.

If `num_ctx` is already present and positive, it is forwarded unchanged.

---

## 9. Configuration Reload

The consumer loads its configuration fresh every time the FastAPI app starts (on lifespan entry):

1. Reads `.env` file for `SERVER_URL`.
2. Reads `~/.thinkfarm/config.ini` for `consumer_id`.
3. Stores the consumer ID in the FastAPI app state (`app.state.consumer_id`) for use on every request.

If the server URL changes, the user must restart the client application for it to take effect.