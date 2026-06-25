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
import uuid
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
    config_path = os.path.join(config_dir, "config.ini")
    config.read(config_path)

    # Priority: ENV > Config File > System ID > UUID
    PROVIDER_ID = (
        os.environ.get("PROVIDER_ID", "")
        or (config.get("provider", "provider_id", fallback=None) if config.has_section("provider") else None)
    ) or None

    # Generate and persist a UUID if nothing is configured
    if not PROVIDER_ID:
        PROVIDER_ID = str(uuid.uuid4())
        if not config.has_section("provider"):
            config.add_section("provider")
        config.set("provider", "provider_id", PROVIDER_ID)
        with open(config_path, "w") as f:
            config.write(f)
        print(f"[WARN] No PROVIDER_ID configured. Generated new ID: {PROVIDER_ID}")

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
zero_eval_inspecting: bool = False     # Flag for zero-eval inspection
my_models: set = set()                 # names of all Ollama models on this machine
loaded_models: set = set()             # names of currently loaded/warm models
job_counter: int = 0                   # Counter for completed jobs since last reconnect
reconnect_requested: bool = False      # Flag to trigger immediate reconnect

# Priority model written by baseprovider.py (_managed_model_loop)
_PRIORITY_MODEL_PATH = os.path.expanduser("~/.thinkfarm/priority_model.txt")
_priority_model: str = ""

# GPU context limits discovered by the context prober { model_name: num_ctx }
gpu_context_limits: dict = {}

# ---------------------------------------------------------------------------
# Slope Monitor (Performance Tracking)
# ---------------------------------------------------------------------------
_slope_peer_thresholds: dict = {}            # { model_name: peak / 3 }
_consecutive_bad: dict[str, int] = {}        # { model_name: count }
_slopemon_data_path = os.path.expanduser("~/.thinkfarm/_slopemon.json")
_trigger_path = os.path.expanduser("~/.thinkfarm/solo_slope_trigger.json")
_ZERO_EVAL_FILE = os.path.expanduser("~/.thinkfarm/ZERO_EVAL")
_TRIGGER_COUNT = 3
_SHORT_JOB_SECS = 10

# ---------------------------------------------------------------------------
# Blacklist ({model} → True)
# ---------------------------------------------------------------------------
_BLACKLIST_FILE = os.path.expanduser("~/.thinkfarm/blacklisted_models.json")
_blacklisted_models: set = set()


def load_blacklist():
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
    _blacklisted_models.add(model_name)
    save_blacklist()
    print(f"[BLACKLIST] Added \"{model_name}\" to blacklist.")

async def _ensure_priority_model_loaded() -> str | None:
    """Load the priority model into VRAM if it hasn't been loaded yet.

    Returns the model name that was actually loaded (or is already loaded),
    or None if nothing was/would be loaded.
    """
    # Read the priority model from the file written by baseprovider.py
    try:
        with open(_PRIORITY_MODEL_PATH, "r") as f:
            model = f.read().strip()
    except (FileNotFoundError, PermissionError):
        return None

    if not model:
        return None

    # Get a fresh list of loaded models directly from Ollama to avoid stale cache issues
    try:
        loaded_models_check = await ollama.get_loaded_models()
        loaded_names = {m.get("name") if isinstance(m, dict) else m for m in loaded_models_check} if loaded_models_check else set()
    except Exception as e:
        print(f"[PRIORITY] Error checking loaded models: {e}")
        return None

    # If priority model is already loaded (by us or another process), just confirm
    if loaded_names and model in loaded_names:
        return model

    # If some other model is loaded and this isn't it, nothing to do right now
    if loaded_names:
        return None

    # 3. Actually load the priority model if nothing else is in use
    if not loaded_names:
        try:
            success = await ollama.load_model(model)
            if not success:
                raise Exception("load_model call returned False")
            print(f"[PRIORITY] Ensured {model} is loaded.")
            _priority_model = model  # Now safe to update
            return model
        except Exception as e:
            print(f"[PRIORITY] Could not load {model}: {e}")
            return None
    else:
        print(f"[PRIORITY] Another model already loaded ({list(loaded_names)}), skipping priority load.")


# For change-detection on status updates
_previous_status = None
_cached_full_models = []                # Original list of dicts from Ollama /api/tags

# Heartbeat: track the last model used and activity time for Ollama keep-alive
_last_model: str = ""
_heartbeat_task: asyncio.Task | None = None
_last_activity_time: float = 0.0

# Reference to the live WebSocket so execute_job can send chunks
_ws_ref = None

# Active inference tasks by job_id
active_tasks: dict = {}

# Event to trigger immediate status updates (heartbeats) and reset the timer
status_update_event: asyncio.Event | None = None




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

    adjusted_limit = int(limit * 0.9) if limit > 0 else limit

    requested = (
        job_body.get("options", {}).get("num_ctx")
        if isinstance(job_body.get("options"), dict)
        else None
    )
    if requested is None:
        return True
    if requested > adjusted_limit:
        print(f"Context check: {model} requested {requested:,} > limit {adjusted_limit:,} — declining.")
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
    global _ws_ref, reconnect_requested, job_counter
    while True:
        try:
            job_counter = 0  # Reset counter on new connection
            ws_url = f"{SERVER_WS_URL}{PROVIDER_ID}"
            async with websockets.connect(ws_url) as websocket:
                _ws_ref = websocket
                print(f"Connected to server: {ws_url}")
                await send_initial_status(websocket)
                asyncio.create_task(periodic_status_updates(websocket))
                await listen_for_server_messages(websocket)
        except Exception as e:
            if reconnect_requested:
                reconnect_requested = False
                print("[RECONNECT] Reconnecting WebSocket immediately (planned refresh after 30 jobs)...")
            else:
                print(f"WebSocket error: {e}, reconnecting in 5 s…")
                await asyncio.sleep(5)
        finally:
            _ws_ref = None


def _filter_by_gpu_limit(models: list) -> list:
    """Filter out models that have a GPU context limit of 0 or are blacklisted."""
    filtered = []
    for m in models:
        name = m.get("name") if isinstance(m, dict) else m
        if gpu_context_limits.get(name) != 0 and name not in _blacklisted_models:
            filtered.append(m)
    return filtered


async def get_provider_status():
    status = await ollama.get_models()
    loaded_list = await ollama.get_loaded_models()
    if status is None or loaded_list is None:
        return None

    status = _filter_by_gpu_limit(status)
    loaded_list = _filter_by_gpu_limit(loaded_list)

    adjusted_limits = {
        name: (int(lim * 0.9) if lim > 0 else lim)
        for name, lim in gpu_context_limits.items()
    }

    return {
        "provider_id": PROVIDER_ID,
        "connected_at": datetime.now().isoformat(),
        "models": status,
        "loaded_models": loaded_list,
        "context_limits": adjusted_limits,
        "is_busy": is_busy > 0,
    }


async def get_loaded_models_only():
    loaded_list = await ollama.get_loaded_models()
    if loaded_list is None:
        return None

    loaded_list = _filter_by_gpu_limit(loaded_list)

    adjusted_limits = {
        name: (int(lim * 0.9) if lim > 0 else lim)
        for name, lim in gpu_context_limits.items()
    }

    return {
        "provider_id": PROVIDER_ID,
        "connected_at": datetime.now().isoformat(),
        "loaded_models": loaded_list,
        "context_limits": adjusted_limits,
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


async def periodic_status_updates(websocket):
    global _previous_status
    last_full_update_time = asyncio.get_event_loop().time()
    while True:
        try:
            if status_update_event is not None:
                try:
                    await asyncio.wait_for(status_update_event.wait(), timeout=30.0)
                    status_update_event.clear()
                    is_event_triggered = True
                except asyncio.TimeoutError:
                    is_event_triggered = False
            else:
                await asyncio.sleep(30)
                is_event_triggered = False

            current_time = asyncio.get_event_loop().time()
            if is_event_triggered or (current_time - last_full_update_time >= 300.0):
                status = await get_provider_status()
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
                    last_full_update_time = current_time
                    print(
                        f"[STATUS] Sent update: "
                        f"{len(status.get('models', []))} models, "
                        f"{len(status.get('loaded_models', []))} loaded"
                    )
            else:
                # Send a simple ping ('hi') to keep the connection warm
                await websocket.send(json.dumps({"type": "ping"}))
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
        job_id = data.get("job_id")
        task = asyncio.create_task(execute_job_with_heartbeat_reset(websocket, data))
        if job_id:
            active_tasks[job_id] = task
    elif msg_type == "cancel_job":
        job_id = data.get("job_id")
        task = active_tasks.get(job_id)
        if task:
            print(f"Cancelling job {job_id} on request from server.")
            task.cancel()


async def handle_job_published(websocket, data: dict):
    if stopping or zero_eval_inspecting:
        return
    model = data.get("model", "")
    body = data.get("body", {})
    if model not in my_models:
        return
    if not _context_fits(model, body):
        return

    loaded = await ollama.get_loaded_models()
    loaded_names = {m.get("name") if isinstance(m, dict) else m for m in loaded}
    if model not in loaded_names:
        await asyncio.sleep(0.5)

    if model not in my_models:
        return
    await _try_accept_job(websocket, data)


# ---------------------------------------------------------------------------
# Job execution – stream from Ollama back to a server via WebSocket
# ---------------------------------------------------------------------------
async def execute_job(websocket, data: dict):
    global is_busy
    is_busy += 1
    job_id = data.get("job_id")
    print(f"Received job {job_id} (queued/active count: {is_busy})")
    try:
        async with execution_lock:
            print(f"Starting job {job_id}")
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

            # Route to the custom model with a fixed context size if limit > 0
            model_name = body.get("model")
            if model_name and endpoint_key != "show":
                limit = gpu_context_limits.get(model_name)
                if limit and limit > 0:
                    body["model"] = f"thinkfarm-{model_name}"
                    if "options" in body and isinstance(body["options"], dict):
                        body["options"].pop("num_ctx", None)
                    print(f"Routing job {job_id} to custom model thinkfarm-{model_name} (fixed context size: {limit})")

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
                        batch = []
                        sep = "\n\n" if endpoint_key.startswith("v1/") else "\n"

                        # Batch all tokens over a fixed interval. Fast models are pooled into fewer packets.
                        _flush_interval = 0.075  # 75ms

                        async def batch_flusher():
                            while True:
                                await asyncio.sleep(_flush_interval)
                                if batch:
                                    data_to_send = "".join(batch)
                                    batch.clear()
                                    await websocket.send(
                                        json.dumps({"type": "chunk", "job_id": job_id, "data": data_to_send})
                                    )

                        flusher_task = asyncio.create_task(batch_flusher())
                        try:
                            async with client.stream("POST", ollama_path, json=body) as response:
                                async for line in response.aiter_lines():
                                    if not line:
                                        continue
                                    batch.append(line + sep)

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
                        finally:
                            flusher_task.cancel()
                            try:
                                await flusher_task
                            except asyncio.CancelledError:
                                pass

                            # Drain any tokens still waiting in the batch (stream ended mid-interval)
                            if batch:
                                data_to_send = "".join(batch)
                                batch.clear()
                                await websocket.send(
                                    json.dumps({"type": "chunk", "job_id": job_id, "data": data_to_send})
                                )
                    else:
                        response = await client.post(ollama_path, json=body)
                        response_data = response.json()
                        if isinstance(response_data, dict) and "model" in response_data and isinstance(response_data["model"], str):
                            if response_data["model"].startswith("thinkfarm-"):
                                response_data["model"] = response_data["model"][len("thinkfarm-"):]
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
                            "is_busy": is_busy > 1,
                        }
                    )
                )
                print(f"Job {job_id} completed — sending job_done")

                # Zero-eval inspection check
                if eval_count == 0 and endpoint_key in ("chat", "generate", "v1/chat/completions", "v1/completions"):
                    asyncio.create_task(_run_zero_eval_inspection(model_name or _last_model))

                # Slope Monitor Check
                actual_throughput = eval_count / (total_duration / 1e9) if total_duration > 0 else 0.0
                
                if total_duration >= _SHORT_JOB_SECS * 1e9 and model_name in _slope_peer_thresholds:
                    threshold = _slope_peer_thresholds[model_name]
                    if actual_throughput < threshold:
                        _consecutive_bad[model_name] = _consecutive_bad.get(model_name, 0) + 1
                        print(f"[SLOPE-MON] {model_name}: throughput={actual_throughput:.1f} "
                              f"< threshold={threshold:.1f} ({_consecutive_bad[model_name]}/{_TRIGGER_COUNT})")
                    else:
                        if _consecutive_bad.get(model_name, 0) > 0:
                            print(f"[SLOPE-MON] {model_name}: recovered — throughput="
                                  f"{actual_throughput:.1f} >= threshold={threshold:.1f}")
                        _consecutive_bad[model_name] = 0
                
                if _consecutive_bad.get(model_name, 0) >= _TRIGGER_COUNT and model_name in _slope_peer_thresholds:
                    details = {
                        "model": model_name,
                        "throughput": round(actual_throughput, 2),
                        "threshold": round(threshold, 2)
                    }
                    with open(_trigger_path, "w") as f:
                        json.dump(details, f)
                    print(f"[SLOPE-MON] *** {model_name} triggered! Shutting down...")
                    global stopping
                    stopping = True

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
                            "is_busy": is_busy > 1,
                        }
                    )
                )
    finally:
        is_busy -= 1
        if status_update_event is not None:
            status_update_event.set()


_zero_eval_lock = asyncio.Lock()

async def _run_zero_eval_inspection(model_name: str):
    if _zero_eval_lock.locked():
        return
    async with _zero_eval_lock:
        global zero_eval_inspecting, is_busy
        zero_eval_inspecting = True
        is_busy += 1
        try:
            while True:
                if stopping:
                    break
                print(f"[ZERO-EVAL] Testing model {model_name} with 'hello' prompt...")
                try:
                    result = await ollama.generate(model_name, "hello", stream=False)
                    if result:
                        print("[ZERO-EVAL] Test succeeded, returning to operation.")
                        break
                except Exception as e:
                    print(f"[ZERO-EVAL] Test failed: {e}")
                
                print("[ZERO-EVAL] Test failed. Writing zero-eval file and waiting 60s...")
                try:
                    with open(_ZERO_EVAL_FILE, "w") as f:
                        json.dump({"model": model_name, "timestamp": datetime.now().isoformat()}, f)
                except Exception as e:
                    print(f"[ZERO-EVAL] Could not write trigger file: {e}")
                
                for _ in range(60):
                    if stopping:
                        break
                    await asyncio.sleep(1)
        finally:
            zero_eval_inspecting = False
            is_busy -= 1


async def execute_job_with_heartbeat_reset(websocket, data: dict):
    """Wrapper that resets the heartbeat timer before executing a job."""
    global _last_activity_time, _last_model, job_counter, reconnect_requested
    _last_activity_time = asyncio.get_event_loop().time()

    # If this is a chat/generate job, remember the model
    body = data.get("body", {})
    model = body.get("model", "")
    if model:
        _last_model = model

    job_id = data.get("job_id")
    try:
        await execute_job(websocket, data)
    finally:
        if job_id:
            active_tasks.pop(job_id, None)
        # Mark completion time so heartbeat waits another full interval
        _last_activity_time = asyncio.get_event_loop().time()

        # Increment job counter and check for scheduled reconnection
        job_counter += 1
        print(f"[RECONNECT] Job completed. Job count since last reconnect: {job_counter}/30. Active/queued jobs: {is_busy}")
        if job_counter >= 30:
            if is_busy == 0:
                reconnect_requested = True
                print("[RECONNECT] 30 jobs reached and provider is idle. Initiating WebSocket reconnection...")
                try:
                    await websocket.close()
                except Exception as e:
                    print(f"[RECONNECT] Error closing WebSocket: {e}")
            else:
                print(f"[RECONNECT] 30 jobs reached, but provider is busy ({is_busy} jobs). Reconnection deferred.")


async def _heartbeat_loop():
    """Periodically send a minimal request to Ollama to keep it warm."""
    heartbeat_interval = 1200  # 20 minutes in seconds
    _loaded_model = None  # Track which model we actually have in VRAM
    while True:
        await asyncio.sleep(120)  

        # Ensure the priority model is present (solo start gap)
        _loaded_model = await _ensure_priority_model_loaded() or _loaded_model

        elapsed = asyncio.get_event_loop().time() - _last_activity_time
        if elapsed >= heartbeat_interval and not stopping and is_busy == 0:
            # Probe whichever model is actually loaded in VRAM
            model = _loaded_model or _last_model
            if not model:
                print("[HEARTBEAT] No model available yet – skipping.")
                continue
            try:
                # Use generate endpoint with a minimal prompt
                result = await ollama.generate(model, "hello", stream=False)
                if result is not None:
                    print(f"[HEARTBEAT] Kept {model} alive.")
                else:
                    print(f"[HEARTBEAT] Heartbeat failed for {model}.")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[HEARTBEAT] Heartbeat error for {model}: {e}")


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
    global status_update_event, gpu_context_limits
    status_update_event = asyncio.Event()

    load_config()
    if not PROVIDER_ID:
        print("[ERROR] Cannot start provider: PROVIDER_ID is missing in config.ini (searched in ~/.thinkfarm/)")
        return False, 1

    # Load cached GPU context limits
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

    # Load blacklist
    load_blacklist()

    # Load slopemon data
    try:
        with open(_slopemon_data_path, "r") as f:
            data = json.load(f)
        for model_name, info in data.get("models", {}).items():
            peak = info.get("peak", 0)
            if peak > 0:
                _slope_peer_thresholds[model_name] = peak / 3
        print(f"[SLOPE-MON] Loaded {len(_slope_peer_thresholds)} thresholds.")
    except (FileNotFoundError, Exception):
        print("[SLOPE-MON] No slopemon data — monitoring disabled.")

    # Fresh Ollama client
    global ollama
    ollama = OllamaClient()

    # Start the WebSocket connection loop
    asyncio.create_task(connect_to_server())

    # Set initial activity time so the first heartbeat fires correctly
    global _last_activity_time
    _last_activity_time = asyncio.get_event_loop().time()

    # Start the background heartbeat task
    global _heartbeat_task
    _heartbeat_task = asyncio.create_task(_heartbeat_loop())

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
    
    if any(count >= _TRIGGER_COUNT for count in _consecutive_bad.values()):
        return False, 42
    return True, 0


if __name__ == "__main__":
    success, code = asyncio.run(main())
    if code == 42:
        sys.exit(42)
    sys.exit(0 if success else 1)
