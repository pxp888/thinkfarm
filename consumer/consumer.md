# thinkfarm Consumer — Architectural Workflow

This document describes the operational logic of the thinkfarm Consumer (`consumer/main.py`), which acts as a local proxy that presents a distributed cluster as a standard local Ollama instance.

## 1. Startup & Initialization

The consumer identifies itself using a unique Consumer ID (CID) and connects to the central server.

```mermaid
sequenceDiagram
    participant C as Consumer App
    participant F as Filesystem (.env / config.ini)
    participant S as Central Server

    C->>F: load_config() (SERVER_URL)
    C->>F: Read [consumer] -> consumer_id
    
    alt CID Found
        C->>C: Store in app.state.consumer_id
    else CID Missing
        C->>C: app.state.consumer_id = None
        Note over C: Inference will fail (500)
    end
```

---

## 2. Metadata Passthrough (Tags & Versions)

The consumer mirrors the local Ollama API by fetching metadata from the central server.

```mermaid
sequenceDiagram
    participant U as User (UI / CLI)
    participant C as Consumer
    participant S as Central Server

    U->>C: GET /api/tags (or /api/version, /api/ps)
    C->>S: GET /api/tags
    
    alt Server Up
        S-->>C: JSON Model List
        C-->>U: JSON Model List
    else Server Down / Error
        C-->>U: Safe Default (e.g., empty list, v0.18.0)
    end
```

---

## 3. Distributed Inference Proxy

The core responsibility of the consumer is to forward inference requests while injecting the `X-Consumer-ID` header for server-side routing and logging.

```mermaid
sequenceDiagram
    participant U as User (Open WebUI / ollama run)
    participant C as Consumer
    participant S as Central Server

    U->>C: POST /api/chat {model, prompt, stream: true}
    
    C->>C: Retrieve CID from state
    C->>C: Capture request body bytes
    
    C->>S: POST /api/chat (Stream)
    Note over C,S: Header: X-Consumer-ID: {CID}
    
    alt Server Accepts Job
        S-->>C: Stream Start (200 OK)
        loop Data Relay
            S->>C: NDJSON Chunk
            C->>U: NDJSON Chunk
        end
        S-->>C: Stream End
    else Server Rejects (Busy/Invalid CID)
        S-->>C: 503 / 403 Error
        C-->>U: JSON Error Detail
    end
```

---

## 4. Key Logic Components

### The `_stream_to_server` Mechanism
Unlike simple metadata endpoints, inference uses a high-performance streaming relay:
1.  **Identity Injection**: Every request is tagged with the `X-Consumer-ID` header.
2.  **Zero-Buffer Streaming**: Chunks are yielded back to the user as soon as they arrive from the central server, ensuring minimal latency.
3.  **Error Propagation**: If the central server returns an error (e.g., "No providers available"), the consumer parses the error detail and raises a corresponding `HTTPException` back to the user.

### Protocol Compatibility
The consumer supports multiple API dialects simultaneously:
- **Ollama Native**: `/api/chat`, `/api/generate`, `/api/show`, etc.
- **OpenAI Compatible**: `/v1/chat/completions`, `/v1/completions`, and `/v1/models`.
- **Legacy Embeddings**: Handles both `/api/embeddings` and the newer `/api/embed` formats.

---

## 5. Configuration Summary

| Key | Source | Description |
|---|---|---|
| `SERVER_URL` | `.env` | The base URL of the thinkfarm central server. |
| `consumer_id` | `config.ini` | The unique UUID identifying this consumer for usage tracking. |
| **Local Port** | Default | Listens on port `11434` by default. Override via `config.ini` or the UI. |
