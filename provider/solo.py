"""
thinkfarm Provider (standalone)

Runs on machines with an Ollama server. Maintains a persistent WebSocket
connection to the central thinkfarm server, announces available models,
and executes inference jobs routed by the server – streaming results back
via the same WebSocket. The underlying Ollama URL is never shared externally.

This is a drop-in replacement for main.py that avoids the FastAPI overhead,
since the provider only makes *outgoing* connections and needs no inbound
HTTP surface.
"""

import asyncio
import configparser
import json
import os
import signal
import sys
from collections import deque
from datetime import datetime

import httpx
from dotenv import load_dotenv
import websockets  # noqa: F401 – pyright / linters see the import

# ---------------------------------------------------------------------------
# Global lock – only one inference job at a time
# ---------------------------------------------------------------------------
execution_lock = asyncio.Lock()

from ollama_client import OllamaClient  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVER_URL = None
SERVER_WS_URL = None
PROVIDER_ID = None


def load_config():
    """Load configuration from .env and config.ini."""
    global PROVIDER_ID, SERVER_URL, SERVER_WS_URL

    if getattr(sys, "frozen", False):
        bundle_dir = getattr(sys, "_MEIPASS")
        env_path = os.path.join(bundle_dir, ".env")
        load_dotenv(env_path)
    else:
        load_dotenv()

    SERVER_URL = os.environ.get("SERVER_URL", "https://app.thinkfarm.net")
    SERVER_WS_URL = os.environ.get("SERVER_WS_URL", "wss://app.thinkfarm.net/ws/provider/")
    print("SERVER_URL", SERVER_URL)
    print("SERVER_WS_URL", SERVER_WS_URL)

    config = configparser.ConfigParser()
    config_dir = os.path.expanduser("~/.thinkfarm")
    os.makedirs(config_dir, exist_ok=True)
    config.read(os.path.join(config_dir, "config.ini"))

    try:
        PROVIDER_ID = config.get("provider", "provider_id")
    except (configparser.NoSectionError, configparser.NoOptionError):
        PROVIDER_ID = None

    if not PROVIDER_ID:
        print(f"Error: PROVIDER_ID not found in {os.path.join(config_dir, 'config.ini')}")

    print(f"Provider ID: {PROVIDER_ID}")
    print("Configuration loaded.")


# ---------------------------------------------------------------------------
# Initialize Ollama client
# Re-initialized in main() to support restarts within the same process.
ollama: OllamaClient = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Mutable provider state
# ---------------------------------------------------------------------------
is_busy: int = 0                       # Number of active or queued jobs
stopping: bool = False                 # Graceful shutdown flag
my_models: set = set()                 # names of all Ollama models on this machine
loaded_models: set = set()             # names of currently loaded/warm models

# GPU context limits discovered by the context prober { model_name: num_ctx }
gpu_context_limits: dict = {}

# For change-detection on status updates
_previous_status = None
_cached_full_models = []                # Original list of dicts from Ollama /api/tags

# Reference to the live WebSocket so execute_job can send chunks
_ws_ref = None


# ---------------------------------------------------------------------------
# Context-limit eligibility check
# ---------------------------------------------------------------------------
def _context_fits(model: str, job_body: dict) -> bool:
    """
    Return True if this provider can handle the job's requested context length.
    """
    limit = gpu_context_limits.get(model)
    if limit is None or limit == -1:
        return True
    if limit == 0:
        print(f"Context check: {model} has limit=0 — declining.")
        return False

    requested = (
        job_body.get("options", {}).get("num_ctx")
        if isinstance(job_body.get("options"), dict)
        else None
    )
    if requested is None:
        return True
    if requested > limit:
        print(f"Context check: {model} requested {requested:,} > limit {limit:,} — declining.")
        return False
    return True


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _update_local_model_sets(status: dict):
    """Synchronise the module-level my_models / loaded_models sets."""
    global my_models, loaded_models, _cached_full_models
    if "models" in status:
        _cached_full_models = status.get("models", [])
    my_models = {
        (m.get("name") if isinstance(m, dict) else m) for m in _cached_full_models
    }
    loaded_models = {
        (m.get("name") if isinstance(m, dict) else m)
        for m in status.get("loaded_models", [])
    }


_ENDPOINT_MAP = {
    "chat": "/api/chat",
    "generate": "/api/generate",
    "embed": "/api/embed",
    "embeddings": "/api/embeddings",
    "show": "/api/show",
    "v1/chat/completions": "/v1/chat/completions",
    "v1/completions": "/v1/completions",
    "v1/responses": "/v1/responses",
}


# ---------------------------------------------------------------------------
# Server connection & status reporting
# ---------------------------------------------------------------------------
async def connect_to_server():
    """Maintain a persistent WebSocket connection to the central server."""
    global _ws_ref
    while True:
        try:
            ws_url = f"{SERVER_WS_URL}{PROVIDER_ID}"
            async with websockets.connect(ws_url) as websocket:
                _ws_ref = websocket
                print(f"Connected to server: {ws_url}")
                await send_initial_status(websocket)
                asyncio.create_task(periodic_status_updates(websocket))
                await listen_for_server_messages(websocket)
        except Exception as e:
            print(f"WebSocket error: {e}, reconnecting in 5 s…")
            await asyncio.sleep(5)
        finally:
            _ws_ref = None


def _filter_by_gpu_limit(models: list) -> list:
    """Filter out models that have a GPU context limit of 0."""
    filtered = []
    for m in models:
        name = m.get("name") if isinstance(m, dict) else m
        if gpu_context_limits.get(name) != 0:
            filtered.append(m)
    return filtered


async def get_provider_status():
    status = await ollama.get_models()
    loaded_list = await ollama.get_loaded_models()
    if status is None or loaded_list is None:
        return None

    status = _filter_by_gpu_limit(status)
    loaded_list = _filter_by_gpu_limit(loaded_list)

    return {
        "provider_id": PROVIDER_ID,
        "connected_at": datetime.now().isoformat(),
        "models": status,
        "loaded_models": loaded_list,
        "context_limits": gpu_context_limits,
        "is_busy": is_busy > 0,
    }


async def get_loaded_models_only():
    loaded_list = await ollama.get_loaded_models()
    if loaded_list is None:
        return None

    loaded_list = _filter_by_gpu_limit(loaded_list)

    return {
        "provider_id": PROVIDER_ID,
        "connected_at": datetime.now().isoformat(),
        "loaded_models": loaded_list,
        "context_limits": gpu_context_limits,
        "is_busy": is_busy > 0,
    }


async def send_initial_status(websocket):
    global _previous_status
    status = await get_provider_status()
    if status is None:
        print("[STATUS] Could not fetch initial status from Ollama. Skipping send.")
        return
    _update_local_model_sets(status)
    _previous_status = {
        "models": frozenset(my_models),
        "loaded_models": frozenset(loaded_models),
    }
    await websocket.send(json.dumps({"type": "status", **status}))


async def _get_changed_status():
    status = await get_provider_status()
    if status is None:
        return None
    current = {
        "models": frozenset(
            (m.get("name") if isinstance(m, dict) else m)
            for m in status.get("models", [])
        ),
        "loaded_models": frozenset(
            (m.get("name") if isinstance(m, dict) else m)
            for m in status.get("loaded_models", [])
        ),
    }
    if _previous_status is None or current != _previous_status:
        return status
    return None


async def _get_changed_loaded_status() -> dict | None:
    status = await get_loaded_models_only()
    if status is None:
        return None
    current = {
        "loaded_models": frozenset(
            (m.get("name") if isinstance(m, dict) else m)
            for m in status.get("loaded_models", [])
        ),
    }
    if (
        _previous_status is None
        or current.get("loaded_models") != _previous_status.get("loaded_models")
        or (is_busy > 0) != _previous_status.get("is_busy")
    ):
        status["models"] = _cached_full_models
        status["is_busy"] = is_busy > 0
        return status
    return None


async def periodic_status_updates(websocket):
    global _previous_status
    _heartbeat_counter = 0
    while True:
        try:
            status = await _get_changed_loaded_status()
            _heartbeat_counter += 1
            if status is None and _heartbeat_counter >= 10:
                status = await get_provider_status()
                _heartbeat_counter = 0
            if status is not None:
                _update_local_model_sets(status)
                await websocket.send(json.dumps({"type": "status", **status}))
                _previous_status = {
                    "models": frozenset(
                        (m.get("name") if isinstance(m, dict) else m)
                        for m in status.get("models", [])
                    ),
                    "loaded_models": frozenset(
                        (m.get("name") if isinstance(m, dict) else m)
                        for m in status.get("loaded_models", [])
                    ),
                    "is_busy": status.get("is_busy", False),
                }
                print(
                    f"[STATUS] Sent update: "
                    f"{len(status.get('models', []))} models, "
                    f"{len(status.get('loaded_models', []))} loaded"
                )
            await asyncio.sleep(30)
        except websockets.exceptions.ConnectionClosed:
            print("[STATUS] WebSocket closed during status update — exiting loop.")
            break
        except Exception as e:
            print(f"[STATUS] Error during status update: {e}")


# ---------------------------------------------------------------------------
# Listening for server messages
# ---------------------------------------------------------------------------
async def listen_for_server_messages(websocket):
    while True:
        try:
            raw_text = await websocket.recv()
            data = json.loads(raw_text)
            msg_type = data.get("type")
            # print(f"Received from server: {msg_type}")
            await handle_server_message(websocket, data)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
            raise
        except Exception as e:
            print(f"Error receiving from server: {e}")
            raise


async def handle_server_message(websocket, data: dict):
    msg_type = data.get("type")
    if msg_type == "job_published":
        await handle_job_published(websocket, data)
    elif msg_type == "job_assigned":
        asyncio.create_task(execute_job(websocket, data))


async def handle_job_published(websocket, data: dict):
    if stopping:
        return
    model = data.get("model", "")
    body = data.get("body", {})
    if model not in my_models:
        return
    if not _context_fits(model, body):
        return

    if is_busy > 0:
        await asyncio.sleep(1.5)
    else:
        loaded = await ollama.get_loaded_models()
        loaded_names = {m.get("name") if isinstance(m, dict) else m for m in loaded}
        if model not in loaded_names:
            await asyncio.sleep(0.5)

    if model not in my_models:
        return
    await _try_accept_job(websocket, data)


# ---------------------------------------------------------------------------
# Job execution – stream from Ollama back to server via WebSocket
# ---------------------------------------------------------------------------
async def execute_job(websocket, data: dict):
    global is_busy
    is_busy += 1
    try:
        async with execution_lock:
            job_id = data.get("job_id")
            endpoint_key = data.get("endpoint", "chat")
            body = data.get("body", {})
            ollama_path = _ENDPOINT_MAP.get(endpoint_key, "/api/chat")
            stream = body.get("stream", True)

            # Normalise embed bodies
            if endpoint_key in ("embed", "embeddings"):
                stream = False
                body = {k: v for k, v in body.items() if k != "stream"}
                if ollama_path == "/api/embed" and "input" not in body and "prompt" in body:
                    body = {**body, "input": body["prompt"]}

            if endpoint_key == "show":
                stream = False
                if "name" not in body and "model" in body:
                    body = {**body, "name": body["model"]}

            # Force usage statistics for OpenAI-compatible streams
            if endpoint_key.startswith("v1/") and stream:
                if "stream_options" not in body:
                    body["stream_options"] = {"include_usage": True}
                elif isinstance(body["stream_options"], dict):
                    body["stream_options"]["include_usage"] = True

            end_start = datetime.now()
            prompt_eval_count = 0
            eval_count = 0

            try:
                async with httpx.AsyncClient(base_url=ollama.base_url, timeout=None) as client:
                    if stream:
                        async with client.stream("POST", ollama_path, json=body) as response:
                            async for line in response.aiter_lines():
                                if not line:
                                    continue
                                await websocket.send(
                                    json.dumps({"type": "chunk", "job_id": job_id, "data": line})
                                )
                                try:
                                    # Handle SSE format for v1 endpoints
                                    parse_line = line
                                    if line.startswith("data: "):
                                        parse_line = line[6:]
                                    
                                    if parse_line.strip() == "[DONE]":
                                        continue

                                    obj = json.loads(parse_line)
                                    
                                    # Capture Ollama native usage
                                    if obj.get("prompt_eval_count"):
                                        prompt_eval_count = obj["prompt_eval_count"]
                                    if obj.get("eval_count"):
                                        eval_count = obj["eval_count"]
                                        
                                    # Capture OpenAI compatible usage
                                    if "usage" in obj and obj["usage"]:
                                        u = obj["usage"]
                                        if u.get("prompt_tokens"):
                                            prompt_eval_count = u["prompt_tokens"]
                                        if u.get("completion_tokens"):
                                            eval_count = u["completion_tokens"]
                                except Exception:
                                    pass
                    else:
                        response = await client.post(ollama_path, json=body)
                        response_data = response.json()
                        if "prompt_eval_count" in response_data:
                            prompt_eval_count = response_data.get("prompt_eval_count", 0)
                            eval_count = response_data.get("eval_count", 0)
                        elif "usage" in response_data:
                            u = response_data["usage"]
                            prompt_eval_count = u.get("prompt_tokens", 0)
                            eval_count = u.get("completion_tokens", 0)
                        await websocket.send(
                            json.dumps(
                                {"type": "chunk", "job_id": job_id, "data": json.dumps(response_data)}
                            )
                        )

                total_duration = int((datetime.now() - end_start).total_seconds() * 1e9)

                await websocket.send(
                    json.dumps(
                        {
                            "type": "job_done",
                            "job_id": job_id,
                            "prompt_eval_count": prompt_eval_count,
                            "eval_count": eval_count,
                            "total_duration": total_duration,
                        }
                    )
                )
                print(f"Job {job_id} completed — sending job_done")

            except Exception as e:
                print(f"Error executing job {job_id}: {e}")
                await websocket.send(
                    json.dumps(
                        {
                            "type": "job_done",
                            "job_id": job_id,
                            "prompt_eval_count": 0,
                            "eval_count": 0,
                            "total_duration": 0,
                        }
                    )
                )
    finally:
        is_busy -= 1


async def _try_accept_job(websocket, data: dict):
    model = data.get("model", "")
    job_id = data.get("job_id")
    print(f"Accepting job {job_id} for model {model}")
    try:
        await websocket.send(
            json.dumps({"type": "accept", "job_id": job_id, "provider_id": PROVIDER_ID})
        )
    except Exception as e:
        print(f"Error sending accept for job {job_id}: {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def main():
    """Bootstrap the provider: load config, init Ollama, start lifecycle tasks."""
    load_config()
    if not PROVIDER_ID:
        print("[ERROR] Cannot start provider: PROVIDER_ID is missing in config.ini (searched in ~/.thinkfarm/)")
        return False

    # Load cached GPU context limits
    global gpu_context_limits
    config_dir = os.path.expanduser("~/.thinkfarm")
    cache_path = os.path.join(config_dir, "gpu_context_limits.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                gpu_context_limits = json.load(f)
            print(f"Loaded {len(gpu_context_limits)} cached context limit(s) from {cache_path}.")
        except Exception as e:
            print(f"WARNING: could not read cache file {cache_path} ({e}). Starting fresh.")
            gpu_context_limits = {}
    else:
        print(f"No cache file found at {cache_path} - starting fresh.")
        gpu_context_limits = {}

    # Fresh Ollama client
    global ollama
    ollama = OllamaClient()

    # Start the WebSocket connection loop
    asyncio.create_task(connect_to_server())

    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_shutdown():
        global stopping
        if not stopping:
            print("\nShutdown signal received, initiating soft stop…")
            stopping = True
            shutdown_event.set()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, request_shutdown)
    except NotImplementedError:
        # signal.add_signal_handler is not available on Windows
        pass

    # Keep alive until shutdown signal
    try:
        await shutdown_event.wait()
    except KeyboardInterrupt:
        # Fallback for systems where add_signal_handler might not catch it
        request_shutdown()

    # Graceful wait: stay alive as long as jobs are running
    while is_busy > 0:
        print(f"Waiting for {is_busy} active job(s) to finish…")
        await asyncio.sleep(1)

    print("All jobs finished. Shutting down Ollama client.")
    await ollama.close()
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
