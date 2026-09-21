# lcpp (thinkfarm Provider Client) — llama.cpp fork

> ⚠️ **Experimental fork.** This is a work-in-progress fork of the thinkfarm provider component, adapted to talk to **llama.cpp (`llama-server`)** and other compatible model runners instead of Ollama. Expect rough edges; the interface and behavior may change or break at any time.

## What changed

- **Runner-agnostic backend** — the provider now communicates with a local model runner via llama.cpp-style APIs (`/v1/models`, `/props`, etc.) rather than the Ollama API. Any server exposing compatible endpoints should work.
- **Headless mode** — runs as a provider-only daemon with no GUI (`headless.py`), using the same config files as the main app.
- **Streamlined provider client** — GUI/PyQt dependencies removed in favor of a lightweight asyncio provider (`provider_client.py`).

## Layout

| File | Purpose |
|---|---|
| `provider_client.py` | Runs as a provider: connects to the central server over WebSocket, executes jobs against the local model runner, streams responses back |
| `headless.py` | Headless entry point — provider-only mode, no GUI (Ctrl+C for clean shutdown) |
| `client_server.py` | Consumer-side proxy: exposes a local API and forwards requests through the central server |
| `config.py` / `.env` | Configuration (server URL, provider ID, local runner URL, whitelists) |
| `context_prober.py` | Probes the local runner for available models and context limits |
| `provider-protocol.md` | WebSocket protocol between provider and central server |

## Running

```bash
# Provider (forwarding to a local llama.cpp server on :8080)
python headless.py
```

Point `LOCAL_LCPP_URL` at your model runner. The consumer proxy (`client_server.py`) works as before for consumers.

## Notes

- Still uses the thinkfarm central server and its protocol — see `provider-protocol.md`.
- This fork targets llama.cpp specifically but is intended to work with any model runner speaking a compatible API.
