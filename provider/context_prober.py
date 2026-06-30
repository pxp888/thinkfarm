"""
context_prober.py - GPU context-limit discovery for Ollama models.

Runs a binary search over num_ctx values for each local model to find the
largest context window that fits entirely in VRAM (no CPU spillover).

Detection method: after each probe call, poll /api/ps and compare
  size  vs  size_vram
If they are equal the model is fully on-GPU for that context length.

Embedding models (those without a chat template in /api/show) are probed via
/api/embed instead of /api/generate so they are never mis-classified.

Results are persisted to gpu_context_limits.json next to this file so that
  - the probe only runs once per model,
  - new models are probed automatically on the next startup,
  - a crash mid-probe does not discard already-found results.
"""

import asyncio
import json
import os
import platform
from typing import Dict, Optional, Any
from datetime import datetime, timezone

import httpx

# Detect Apple Silicon for unified memory handling
_IS_MACOS_UNIFIED = platform.system() == "Darwin" and "arm64" in platform.machine()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_FILE = os.path.expanduser("~/.thinkfarm/gpu_context_limits.json")
_BASELINES_FILE = os.path.expanduser("~/.thinkfarm/performance_baselines.json")

# Smallest context size we will probe (powers-of-two friendlier than 1)
_MIN_CTX = 512

# Fallback upper bound if /api/show doesn't report num_ctx_train.
# Set to the largest known training context across supported models (262144 = Gemma 3 / Qwen 2.5).
_DEFAULT_MAX_CTX = 262144

# Maximum binary-search iterations per model. 10 gives ~0.1% accuracy over a
# range of 0-262144 (worst-case uncertainty: ~256 tokens). We stop early and
# return the last confirmed GPU-OK value so we always land on the safe side.
_MAX_PROBE_ITERATIONS = 10

# How many times to poll /api/ps waiting for a model to show up after load
_PS_POLL_ATTEMPTS = 20
_PS_POLL_INTERVAL = 1.5  # seconds

# Tiny prompt - we only care about memory layout, not output quality
_PROBE_PROMPT = "Hi"

# Log prefix so probe lines stand out in mixed output
_PFX = "[CTX-PROBE]"


# macOS / Apple Silicon unified memory probe helpers
# Total RAM on the machine (used to estimate safe free memory threshold)

def _get_available_ram_mb() -> int:
    """Return available system RAM in MB.

    Returns -1 (N/A) when the platform isn't measurable.

    Order of preference:
      1. ``psutil.virtual_memory()`` (preferred, works everywhere)
      2. ``vm_stat`` CLI (macOS, still works on Sequoia+)
      3. ``/proc/meminfo MemAvailable`` (Linux)
    """
    def _round_page_to_mb(page_count: int, page_size: int = 4096) -> int:
        return page_count * page_size // (1024 * 1024)

    # 1. psutil -- preferred on all platforms
    try:
        from psutil import virtual_memory as _psutil_vm
        return int(_psutil_vm().available) // (1024 * 1024)
    except Exception:
        pass

    # 2. macOS: fall back to ``vm_stat`` CLI
    #    vm_stat still works on macOS 15+ even though the old sysctl OIDs don't.
    if platform.system() == "Darwin":
        try:
            import subprocess
            raw = subprocess.check_output(["vm_stat"]).decode()
            # Parse lines like ": Pages free:               12345"
            pages_free = pages_inactive = pages_speculative = 0
            for line in raw.splitlines():
                if line.startswith("Pages free:"):
                    pages_free = int(line.split(":")[1].strip())
                elif line.startswith("Pages inactive:"):
                    pages_inactive = int(line.split(":")[1].strip())
                elif line.startswith("Pages speculative:"):
                    pages_speculative = int(line.split(":")[1].strip())
            # macOS available ~= free + inactive + speculative
            avail_pages = pages_free + pages_inactive + pages_speculative
            return _round_page_to_mb(avail_pages)
        except Exception:
            pass

    # 3. Linux: ``/proc/meminfo MemAvailable``
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        kb = int(line.split(":")[1].strip())
                        return kb // 1024
        except Exception:
            pass

    return -1  # all methods failed


# Minimum free RAM (in MB) to leave after loading a model.
# macOS will thrash hard otherwise.
_MIN_FREE_RAM_MB = 2048  # 2 GB


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def load_context_limits() -> Dict[str, int]:
    """Load previously discovered limits from disk. Returns {} on first run."""
    if not os.path.exists(_CACHE_FILE):
        print(f"{_PFX} No cache file found at {_CACHE_FILE} - starting fresh.")
        return {}
    try:
        with open(_CACHE_FILE, "r") as f:
            data = json.load(f)
        print(f"{_PFX} Loaded {len(data)} cached context limit(s) from {_CACHE_FILE}.")
        for model, ctx in data.items():
            print(f"{_PFX}   {model}: {ctx:,} tokens (cached)")
        return data
    except Exception as e:
        print(f"{_PFX} WARNING: could not read cache file ({e}). Starting fresh.")
        return {}


def save_context_limits(limits: Dict[str, int]) -> None:
    """Persist the limits dict atomically-ish (write then rename)."""
    tmp = _CACHE_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(limits, f, indent=2)
        os.replace(tmp, _CACHE_FILE)
        print(f"{_PFX} Saved {len(limits)} context limit(s) to {_CACHE_FILE}.")
    except Exception as e:
        print(f"{_PFX} WARNING: could not save cache file ({e}).")


def load_performance_baselines() -> Dict[str, Any]:
    """Load previously discovered baselines from disk. Returns a default dict structure on first run."""
    if not os.path.exists(_BASELINES_FILE):
        return {"baselines": {}, "performance_alerts": {}}
    try:
        with open(_BASELINES_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"baselines": {}, "performance_alerts": {}}
        if "baselines" not in data:
            data["baselines"] = {}
        if "performance_alerts" not in data:
            data["performance_alerts"] = {}
        return data
    except Exception as e:
        print(f"{_PFX} WARNING: could not read baselines file ({e}). Starting fresh.")
        return {"baselines": {}, "performance_alerts": {}}


def save_performance_baselines(data: Dict[str, Any]) -> None:
    """Persist the baselines dict atomically-ish (write then rename)."""
    os.makedirs(os.path.dirname(_BASELINES_FILE), exist_ok=True)
    tmp = _BASELINES_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, _BASELINES_FILE)
        print(f"{_PFX} Saved performance baselines to {_BASELINES_FILE}.")
    except Exception as e:
        print(f"{_PFX} WARNING: could not save baselines file ({e}).")


# ---------------------------------------------------------------------------
# Ollama helper calls (standalone, not using OllamaClient to keep timeout control)
# ---------------------------------------------------------------------------


class OllamaConnectionError(RuntimeError):
    """Raised when the Ollama server is unreachable or crashed during probing."""
    pass


async def _is_server_alive(base_url: str) -> bool:
    """Check if Ollama server is responsive by calling /api/tags."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            url = base_url.rstrip("/") + "/api/tags"
            resp = await client.get(url)
            return resp.status_code == 200
    except Exception:
        return False


async def _wait_for_server(base_url: str, max_attempts: int = 15) -> bool:
    """Wait for the Ollama server to become responsive again."""
    print(f"{_PFX}     Checking if Ollama server at {base_url} is alive...")
    for attempt in range(1, max_attempts + 1):
        if await _is_server_alive(base_url):
            print(f"{_PFX}     Ollama server is responsive.")
            return True
        print(f"{_PFX}     Ollama server is unresponsive (attempt {attempt}/{max_attempts}). Waiting 3s...")
        await asyncio.sleep(3)
    return False


def _get_nvidia_gpu_vram_mb() -> tuple[int, int]:
    """Return (total_vram_mb, free_vram_mb) for NVIDIA GPU."""
    try:
        import subprocess
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        ).strip().split("\n")
        
        # SUM VRAM across visible or all GPUs
        import os
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            try:
                indices = [int(x.strip()) for x in visible.split(",") if x.strip().isdigit()]
                if indices:
                    total = 0
                    free = 0
                    for idx in indices:
                        if idx < len(res):
                            parts = res[idx].split(",")
                            total += int(parts[0].strip())
                            free += int(parts[1].strip())
                    return total, free
            except Exception:
                pass
        
        total = 0
        free = 0
        for line in res:
            parts = line.split(",")
            if len(parts) >= 2:
                total += int(parts[0].strip())
                free += int(parts[1].strip())
        return total, free
    except Exception as e:
        print(f"{_PFX}   WARNING: Failed to query GPU VRAM via nvidia-smi: {e}")
        return -1, -1


def _safe_int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def _estimate_kv_cache_bytes_per_token(model_info: Dict[str, Any]) -> float:
    """Estimate KV cache size per token in bytes."""
    layers = 0
    for k, v in model_info.items():
        if k.endswith(".block_count"):
            layers = _safe_int(v)
            break
            
    kv_heads = 0
    for k, v in model_info.items():
        if k.endswith(".head_count_kv"):
            kv_heads = _safe_int(v)
            break
            
    heads = 0
    for k, v in model_info.items():
        if k.endswith(".head_count"):
            heads = _safe_int(v)
            break
            
    if kv_heads == 0:
        kv_heads = heads if heads > 0 else 32
        
    head_dim = 0
    for k, v in model_info.items():
        if k.endswith(".key_length") or k.endswith(".value_length"):
            head_dim = _safe_int(v)
            break
            
    if head_dim == 0:
        emb_len = 0
        for k, v in model_info.items():
            if k.endswith(".embedding_length"):
                emb_len = _safe_int(v)
                break
        if emb_len > 0 and heads > 0:
            head_dim = emb_len // heads
        else:
            head_dim = 128
            
    if layers == 0:
        layers = 32
        
    # KV cache size per token = 2 (key & value) * layers * kv_heads * head_dim * 2 (bytes per float16)
    return 2 * layers * kv_heads * head_dim * 2


async def _get_model_size_from_tags(base_url: str, model_name: str) -> int:
    """Fetch the total model size in bytes from /api/tags."""
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            for m in data.get("models", []):
                if m.get("name") == model_name:
                    return int(m.get("size", 0))
    except Exception:
        pass
    return 0


async def _get_model_info(base_url: str, model_name: str) -> tuple[int, bool, bool]:
    """
    Query /api/show and return (max_ctx, is_embed, is_eligible).

    is_embed is True when the model has no chat template.
    is_eligible is False for cloud or custom models.

    Falls back to (_DEFAULT_MAX_CTX, False, True) on any error.
    """
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=15.0) as client:
            # Newer Ollama versions use {"model": ...}; older ones use {"name": ...}.
            resp = await client.post("/api/show", json={"model": model_name})
            if resp.status_code == 400:
                resp = await client.post("/api/show", json={"name": model_name})
            resp.raise_for_status()
            info = resp.json()

        # Guard against Ollama returning a plain string (e.g. an error message)
        if not isinstance(info, dict):
            raise ValueError(f"/api/show returned non-dict: {info!r}")

        # Eligibility check: skip cloud and custom models.
        # Allow official models that use internal blob paths/digests in parent_model.
        remote_host = info.get("remote_host")
        remote_model = info.get("remote_model")
        parent = (info.get("details", {}) or {}).get("parent_model", "")
        is_eligible = not (remote_host or remote_model)
        if parent and not (parent.startswith("/") or "sha256" in parent):
            is_eligible = False

        # Detect embedding-only models: they have no chat/generate template,
        # or they explicitly list 'embedding' in capabilities, or use a known architecture.
        template = info.get("template", "").strip()
        details = info.get("details") or {}
        families = details.get("families") or []
        capabilities = info.get("capabilities") or []
        model_info = info.get("model_info") or {}
        arch = str(model_info.get("general.architecture", "")).lower()

        is_embed = (
            template == "" or
            "embedding" in capabilities or
            any(f in ["bert", "nomic-bert"] for f in families) or
            "bert" in arch
        )

        # num_ctx_train lives inside model_info. Keys vary by architecture (llama, qwen, bert etc).
        ctx = (
            model_info.get("llama.context_length")
            or model_info.get("general.context_length")
            or next((v for k, v in model_info.items() if k.endswith(".context_length")), None)
            or _DEFAULT_MAX_CTX
        )
        # Attempt to find num_ctx in Modelfile parameters if not in model_info
        if ctx == _DEFAULT_MAX_CTX and "parameters" in info and isinstance(info["parameters"], str):
            for line in info["parameters"].splitlines():
                if line.strip().startswith("num_ctx"):
                    try:
                        ctx = int(line.split()[-1])
                        break
                    except:
                        pass
        ctx = int(ctx)

        # Estimate KV cache size per token and cap ctx analytically based on VRAM
        try:
            kv_bytes_per_token = _estimate_kv_cache_bytes_per_token(model_info)
            if kv_bytes_per_token > 0:
                model_size_bytes = await _get_model_size_from_tags(base_url, model_name)
                if model_size_bytes > 0:
                    total_vram_mb, _ = _get_nvidia_gpu_vram_mb()
                    if total_vram_mb > 0:
                        total_vram_bytes = total_vram_mb * 1024 * 1024
                        # Reserve 1.5 GB for system/UI overhead
                        overhead_bytes = 1536 * 1024 * 1024
                        available_kv_bytes = total_vram_bytes - model_size_bytes - overhead_bytes
                        if available_kv_bytes > 0:
                            analytical_max = int(available_kv_bytes / kv_bytes_per_token)
                            print(f"{_PFX}   Analytical KV-based max context: {analytical_max:,} tokens (KV size/token = {kv_bytes_per_token} bytes, model weights = {model_size_bytes / (1024*1024):,.1f} MB, VRAM = {total_vram_mb:,} MB)")
                        else:
                            print(f"{_PFX}   WARNING: Model size ({model_size_bytes / (1024*1024):,.1f} MB) + overhead exceeds VRAM ({total_vram_mb:,} MB).")
        except Exception as e:
            print(f"{_PFX}   WARNING: Could not calculate analytical KV limit: {e}")

        model_type = "embedding" if is_embed else "generative"
        print(f"{_PFX}   /api/show -> type={model_type}, max_ctx={ctx:,}")
        return ctx, is_embed, is_eligible

    except Exception as e:
        print(
            f"{_PFX}   WARNING: /api/show failed for {model_name} ({e}). "
            f"Assuming generative/eligible, upper bound {_DEFAULT_MAX_CTX:,}."
        )
        return _DEFAULT_MAX_CTX, False, True


async def _probe_performance_baseline(
    base_url: str,
    model_name: str,
    num_ctx: int,
) -> Optional[float]:
    """
    Establish a performance baseline by running 3 passes and measuring the slope.
    Saves the average slope in performance_baselines.json.
    """
    print(f"{_PFX}   Establishing performance baseline for {model_name} at num_ctx={num_ctx:,}...")
    slopes = []
    
    for i in range(1, 4):
        print(f"{_PFX}     Pass {i}/3...")
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=600.0) as client:
                payload = {
                    "model": model_name,
                    "prompt": "what is the history of Sweden?",
                    "stream": False,
                    "options": {
                        "num_ctx": num_ctx,
                        "temperature": 0.2
                    }
                }
                resp = await client.post("/api/generate", json=payload)
                resp.raise_for_status()
                gen_data = resp.json()
                
                prompt_eval_duration = gen_data.get("prompt_eval_duration", 0)
                eval_duration = gen_data.get("eval_duration", 0)
                eval_count = gen_data.get("eval_count", 0)
                prompt_eval_count = gen_data.get("prompt_eval_count", 0)
                
                compute_seconds = (prompt_eval_duration + eval_duration) / 1e9
                if compute_seconds <= 0:
                    print(f"{_PFX}       Warning: compute_seconds is 0. Skipping this pass.")
                    continue
                
                if compute_seconds <= 0:
                    print(f"{_PFX}       Warning: compute_seconds is 0. Skipping this pass.")
                    continue
                
                slope = (eval_count + 0.003 * prompt_eval_count) / compute_seconds
                print(f"{_PFX}       eval_count={eval_count}, prompt_eval_count={prompt_eval_count}, compute_seconds={compute_seconds:.3f}s, slope={slope:.3f} tokens/s")
                slopes.append(slope)
        except Exception as e:
            print(f"{_PFX}       Error in pass {i}: {e}")
            
    if not slopes:
        print(f"{_PFX}     Failed to measure any valid slopes for {model_name}.")
        return None
        
    avg_slope = sum(slopes) / len(slopes)
    print(f"{_PFX}     Average slope: {avg_slope:.3f} tokens/s")
    
    baselines_data = load_performance_baselines()
    baselines_data["baselines"][model_name] = {
        "slope": round(avg_slope, 2),
        "samples": len(slopes),
        "last_probed": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }
    save_performance_baselines(baselines_data)
    return avg_slope


async def _probe_at_ctx(
    base_url: str,
    model_name: str,
    num_ctx: int,
    is_embed: bool = False,
) -> bool:
    """
    Load the model at the given num_ctx by sending a tiny probe request,
    then check if it fits safely.

    On **Linux / non-unified-memory** systems: compares `size` vs `size_vram`.
      Returns True  -> model is fully in GPU VRAM (no spillover).
      Returns False -> VRAM spillover detected.

    On **macOS Apple Silicon (unified memory)**: there is no separate VRAM,
      so `size == size_vram` is always True. Here we check whether loading
      the model would leave enough *free system RAM*.

    Uses /api/embed for embedding models, /api/generate for all others.
    """

    if is_embed:
        # Embedding models: use /api/embed - /api/generate returns an error.
        print(f"{_PFX}     Sending probe embed (num_ctx={num_ctx:,}) ...", flush=True)
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=600.0) as client:
                payload = {
                    "model": model_name,
                    "input": _PROBE_PROMPT,
                    "options": {"num_ctx": num_ctx},
                }
                resp = await client.post("/api/embed", json=payload)
                resp.raise_for_status()
                embed_data = resp.json()
                n_embeddings = len(embed_data.get("embeddings", []))
                print(
                    f"{_PFX}     Embed OK - {n_embeddings} embedding vector(s) returned"
                )
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            if isinstance(e, httpx.HTTPStatusError):
                try:
                    body = e.response.text
                    err_msg += f" | Response body: {body}"
                except Exception:
                    pass
            print(f"{_PFX}     ERROR during probe embed: {err_msg}")
            
            # Check if the server is still alive
            if not await _is_server_alive(base_url):
                print(f"{_PFX}     WARNING: Ollama server appears to have crashed/stopped during embed probe.")
                is_alive = await _wait_for_server(base_url)
                if not is_alive:
                    raise OllamaConnectionError(f"Ollama server died during embed probing: {err_msg}")
            return False
    else:
        # Generative models: use /api/generate.
        print(
            f"{_PFX}     Sending probe generate (num_ctx={num_ctx:,}) ...", flush=True
        )
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=600.0) as client:
                payload = {
                    "model": model_name,
                    "prompt": _PROBE_PROMPT,
                    "stream": False,
                    "options": {"num_ctx": num_ctx, "num_predict": 1},
                }
                resp = await client.post("/api/generate", json=payload)
                resp.raise_for_status()
                gen_data = resp.json()
                print(
                    f"{_PFX}     Generate OK - "
                    f"eval_count={gen_data.get('eval_count', '?')}, "
                    f"total_duration={gen_data.get('total_duration', '?')}"
                )
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            if isinstance(e, httpx.HTTPStatusError):
                try:
                    body = e.response.text
                    err_msg += f" | Response body: {body}"
                except Exception:
                    pass
            print(f"{_PFX}     ERROR during probe generate: {err_msg}")
            
            # Check if the server is still alive
            if not await _is_server_alive(base_url):
                print(f"{_PFX}     WARNING: Ollama server appears to have crashed/stopped during generate probe.")
                is_alive = await _wait_for_server(base_url)
                if not is_alive:
                    raise OllamaConnectionError(f"Ollama server died during generate probing: {err_msg}")
            return False

    # Interpret the result based on platform.
    if _IS_MACOS_UNIFIED:
        # No separate VRAM on Apple Silicon - check free system RAM instead.
        remaining_free_mb = _get_available_ram_mb()
        if remaining_free_mb < 0:
            print(f"{_PFX}     Could not measure available RAM - defaulting to GPU-only check.")
            return True  # be permissive; fallback is safer
        print(
            f"{_PFX}     macOS free RAM: {remaining_free_mb:,} MB | "
            f"{'[OK - safe headroom]' if remaining_free_mb >= _MIN_FREE_RAM_MB else '[WARN - low free RAM]'}"
        )
        return remaining_free_mb >= _MIN_FREE_RAM_MB
    print(f"{_PFX}     Polling /api/ps for {model_name} ...", flush=True)
    for attempt in range(1, _PS_POLL_ATTEMPTS + 1):
        await asyncio.sleep(_PS_POLL_INTERVAL)
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
                ps_resp = await client.get("/api/ps")
                ps_resp.raise_for_status()
                ps_models = ps_resp.json().get("models", [])

            entry = next(
                (
                    m
                    for m in ps_models
                    if m.get("name", "").startswith(model_name.split(":")[0])
                ),
                None,
            )

            if entry is None:
                print(
                    f"{_PFX}     /api/ps poll {attempt}/{_PS_POLL_ATTEMPTS}: "
                    f"model not yet visible ..."
                )
                continue

            size = entry.get("size", -1)
            size_vram = entry.get("size_vram", -1)
            print(
                f"{_PFX}     /api/ps poll {attempt}/{_PS_POLL_ATTEMPTS}: "
                f"size={size:,}  size_vram={size_vram:,}  "
                f"{'[GPU-only]' if size == size_vram else '[CPU spillover]'}"
            )

            return size == size_vram

        except Exception as e:
            print(f"{_PFX}     /api/ps poll {attempt}/{_PS_POLL_ATTEMPTS}: error ({e})")

    print(
        f"{_PFX}     Model did not appear in /api/ps after "
        f"{_PS_POLL_ATTEMPTS} attempts - treating as failed probe."
    )
    return False


async def _unload_model(base_url: str, model_name: str, is_embed: bool = False) -> None:
    """
    Unload the model from VRAM by setting keep_alive=0.

    Embedding models must be unloaded via /api/embed (keep_alive=0) because
    Ollama ignores keep_alive on /api/generate for models that can't generate.
    """
    print(f"{_PFX}   Unloading {model_name} from VRAM ...", flush=True)
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            if is_embed:
                await client.post(
                    "/api/embed",
                    json={"model": model_name, "input": "", "keep_alive": 0},
                )
            else:
                await client.post(
                    "/api/generate",
                    json={"model": model_name, "prompt": "", "keep_alive": 0},
                )
        print(f"{_PFX}   Unload request sent.")
        await asyncio.sleep(1.0)  # brief pause before probing next model
    except Exception as e:
        print(f"{_PFX}   WARNING: unload request failed ({e}). Continuing anyway.")


# ---------------------------------------------------------------------------
# Binary search for a single model
# ---------------------------------------------------------------------------


async def _find_max_gpu_ctx(
    base_url: str,
    model_name: str,
    upper_bound: int,
    is_embed: bool = False,
) -> Optional[int]:
    """
    Binary-search for the largest num_ctx that keeps the model fully on GPU.

    upper_bound should be pre-fetched from _get_model_info by the caller.

    Returns the best context length found, or None if even _MIN_CTX spills.
    """
    lo, hi = _MIN_CTX, upper_bound
    best: Optional[int] = None

    probe_type = "embed" if is_embed else "generate"
    print(f"{_PFX}   Probe type     : {probe_type}")
    print(f"{_PFX}   Binary search range: [{lo:,} ... {hi:,}]")

    iteration = 0
    while lo <= hi:
        iteration += 1
        mid = (lo + hi) // 2
        print(
            f"{_PFX}   Iteration {iteration}/{_MAX_PROBE_ITERATIONS}: testing num_ctx={mid:,}  "
            f"(range [{lo:,} ... {hi:,}])"
        )

        gpu_ok = await _probe_at_ctx(base_url, model_name, mid, is_embed=is_embed)

        if gpu_ok:
            best = mid
            print(f"{_PFX}   -> GPU-OK at {mid:,}. Trying higher.")
            lo = mid + 1
        else:
            print(f"{_PFX}   -> Spillover at {mid:,}. Trying lower.")
            hi = mid - 1

        if iteration >= _MAX_PROBE_ITERATIONS:
            remaining = hi - lo + 1
            best_value = f"{best:,}" if best is not None else "0"
            print(
                f"{_PFX}   Reached {_MAX_PROBE_ITERATIONS}-iteration cap "
                f"(~{remaining:,} tokens of uncertainty remaining). "
                f"Stopping on safe side at {best_value}."
            )
            break

    return best


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


async def create_custom_model(base_url: str, original_model: str, max_ctx: int) -> bool:
    """
    Create a custom model in Ollama based on original_model with a fixed context window.
    The custom model is prefixed with 'thinkfarm-'.
    """
    custom_model_name = f"thinkfarm-{original_model}"
    adjusted_ctx = int(max_ctx * 0.9)
    print(f"{_PFX} Ensuring custom model {custom_model_name} exists with num_ctx={adjusted_ctx} (90% of discovered {max_ctx})...")
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
            resp = await client.post(
                "/api/create",
                json={
                    "model": custom_model_name,
                    "from": original_model,
                    "parameters": {
                        "num_ctx": adjusted_ctx
                    },
                    "stream": False,
                }
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("status") == "success":
                print(f"{_PFX} Successfully created/updated custom model {custom_model_name}.")
                return True
            else:
                print(f"{_PFX} Custom model creation status: {result}")
                return False
    except Exception as e:
        print(f"{_PFX} ERROR creating custom model {custom_model_name}: {e}")
        return False


async def run_context_probing(
    base_url: str,
    model_names: list,
    existing_limits: Dict[str, int],
) -> Dict[str, int]:
    """
    Probe any model in *model_names* that is NOT already in *existing_limits*.

    Saves incremental results to disk after each model so a crash doesn't
    discard already-discovered limits.

    Returns the updated limits dict (existing + newly discovered).
    """
    limits = dict(existing_limits)  # work on a copy

    return await _probe_new_models(
        base_url=base_url,
        model_names=model_names,
        existing_limits=limits,
    )


async def _probe_new_models(
    base_url: str,
    model_names: list,
    existing_limits: Dict[str, int],
) -> Dict[str, int]:
    """
    Probe models that are not yet in the cache or lack a performance baseline.

    This is the original probe logic, now separated for use by both the
    FastAPI app and the CLI entry point.
    """
    limits = dict(existing_limits)
    baselines_data = load_performance_baselines()
    baselines = baselines_data.get("baselines", {})

    to_probe = []
    for m in model_names:
        if m not in limits:
            to_probe.append(m)
        elif limits[m] > 0 and m not in baselines:
            to_probe.append(m)

    if not to_probe:
        print(
            f"{_PFX} All {len(model_names)} model(s) already have cached limits and baselines - "
            f"skipping probe."
        )
        # Ensure custom models exist for all eligible local models that have a valid limit
        for model_name in model_names:
            limit = limits.get(model_name)
            if limit is not None and limit > 0:
                await create_custom_model(base_url, model_name, limit)
        return limits

    try:
        for idx, model_name in enumerate(to_probe, 1):
            print(f"{_PFX} [{idx}/{len(to_probe)}] Probing: {model_name}")
            print(f"{_PFX} " + "-" * 56)

            # Check server liveness first to fail fast if Ollama is down
            if not await _is_server_alive(base_url):
                raise OllamaConnectionError("Ollama server is unresponsive at the start of probing a model.")

            # Detect model type once - drives probe and unload endpoint choice.
            # Also captures the declared max ctx to avoid a second /api/show call.
            upper_bound, is_embed, is_eligible = await _get_model_info(base_url, model_name)

            if not is_eligible:
                print(f"{_PFX}  Skipping {model_name}: cloud or custom model is not eligible for sharing.")
                # Set to 0 in limits so we don't try again next time, but it won't be announced by the client anyway
                limits[model_name] = 0
                save_context_limits(limits)
                continue

            if is_embed:
                print(f"{_PFX}  Model type: EMBEDDING - will probe via /api/embed")
            else:
                print(f"{_PFX}  Model type: GENERATIVE - will probe via /api/generate")

            if is_embed:
                # Embedding models are always "safe" even if they spill to CPU,
                # as they don't have the same context/OOM risks as generative models.
                print(f"{_PFX}  RESULT: {model_name} is an embedding model. Storing -1 (N/A).")
                limits[model_name] = -1
            else:
                if model_name in limits and limits[model_name] > 0:
                    best_ctx = limits[model_name]
                    print(f"{_PFX}  Using cached context limit: {best_ctx:,} tokens.")
                else:
                    best_ctx = await _find_max_gpu_ctx(
                        base_url, model_name, upper_bound=upper_bound, is_embed=False
                    )

                if best_ctx is None:
                    print(
                        f"{_PFX}  RESULT: Could not run {model_name} on GPU even at "
                        f"{_MIN_CTX:,} tokens. Storing 0 (GPU unavailable)."
                    )
                    limits[model_name] = 0
                else:
                    print(
                        f"{_PFX}  RESULT: Max GPU-safe context for {model_name} "
                        f"= {best_ctx:,} tokens [OK]"
                    )
                    limits[model_name] = best_ctx
                    
                    # Establish performance baseline
                    await _probe_performance_baseline(base_url, model_name, best_ctx)

            # Persist after each model in case of crash
            save_context_limits(limits)

            # Free VRAM before the next model
            await _unload_model(base_url, model_name, is_embed=is_embed)
            print()
    except OllamaConnectionError as e:
        print(f"{_PFX} ERROR: Probing aborted due to Ollama server connection loss: {e}")
        return limits

    # Ensure custom models exist for all eligible local models that have a valid limit
    for model_name in model_names:
        limit = limits.get(model_name)
        if limit is not None and limit > 0:
            await create_custom_model(base_url, model_name, limit)

    print(f"{_PFX} ----- ----- ----- ----- ----- ----- ----- ----- ----- -----")
    print(f"{_PFX} Probing complete. Summary:")
    for model, ctx in limits.items():
        cached_tag = "(cached)" if model not in to_probe else "(new)"
        print(f"{_PFX}   {model}: {ctx:,} tokens {cached_tag}")
    print(
        f"{_PFX} ----- -------- ----- ----------- ----- --- ----- ----- ----- ------- ---\n"
    )

    return limits


async def get_performance_data() -> Dict[str, Any]:
    """Fetch the global performance endpoint and return the parsed JSON."""
    perf_url = "https://www.thinkfarm.net/api/performance"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(perf_url)
            resp.raise_for_status()
            data = resp.json()
        print(f"{_PFX}  [PERF] Fetched global performance data for {len(data.get('data', []))} model(s).")
        return data
    except Exception as e:
        print(f"{_PFX}  [PERF] WARNING: could not fetch performance data from {perf_url} ({e}).")
        return {"data": []}


async def get_ollama_models(base_url: str) -> list:
    """
    Query Ollama's /api/tags endpoint and return a list of model names.

    Returns:
        List of model name strings (e.g., ["llama3.2", "mistral:7b"])
    """
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            names = [m.get("name", "").strip() for m in models if m.get("name")]
            names = [n for n in names if not n.startswith("thinkfarm-")]
            return names
    except Exception as e:
        print(f"{_PFX} WARNING: Failed to fetch tags from {base_url} ({e}).")
        return []


# ---------------"""""""""---CLI entry point
# ---------------""""""""""


def main(args=None):
    """Main CLI entry point for standalone context probing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Probe local Ollama models to discover GPU context limits."
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Ollama server URL (default: http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-probing all models, ignoring cached results",
    )
    args = parser.parse_args(args)

    url = args.ollama_url or "http://127.0.0.1:11434"
    asyncio.run(_run_cli_probe(url, args.force))


async def _run_cli_probe(base_url: str, force: bool = False):
    """
    Main probe workflow.

    1. Query /api/tags for all available model names
    2. Load any previously discovered limits
    3. Probe models that need probing (or all if --force was given)
    4. Save results and print summary
    """
    limits = load_context_limits()  # Load existing cache

    # If --force, clear all limits and start fresh
    if force:
        print(f"{_PFX} --force given - clearing {len(limits)} cached limits.")
        save_context_limits({})
        limits = {}
        save_performance_baselines({"baselines": {}, "performance_alerts": {}})

    # Fetch all models from Ollama
    print(f"{_PFX} Fetching model list from {base_url}/api/tags...")
    models = await get_ollama_models(base_url)

    if not models:
        print(f"{_PFX} No models found on Ollama server.")
        print(f"{_PFX} Make sure Ollama is running and models are pulled.")
        return

    print(f"{_PFX} Found {len(models)} model(s): {', '.join(models)}\n")

    # Run probing
    updated_limits = await _probe_new_models(
        base_url=base_url,
        model_names=models,
        existing_limits=limits,
    )

    print(f"\n{_PFX} Done!")
    print(f"{_PFX} Results saved to {_CACHE_FILE}")
    print(
        f"{_PFX} -------- ----------- --- -------- ---- ----- -------- ------ ----- --------- ---"
    )


if __name__ == "__main__":
    main()
