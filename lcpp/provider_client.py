import sys
sys.modules["websockets.speedups"] = None
import asyncio
import json
import logging
import time
import os
from datetime import datetime
from pathlib import Path
import httpx
import websockets
from config import ConfigManager

logger = logging.getLogger("thinkfarm.provider")

class LlamaCppClient:
    def __init__(self, base_url: str, config: ConfigManager = None):
        self.base_url = base_url.rstrip("/")
        self.config = config

    async def get_models(self) -> list:
        if self.config and self.config.model_name:
            m = {
                "name": self.config.model_name,
                "digest": (getattr(self.config, "model_digest", "") if self.config else "") or self.config.model_name
            }
            return [m]
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/v1/models")
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    models = []
                    for model in data:
                        model_id = model.get("id")
                        if model_id:
                            # Normalize path to filename
                            name = os.path.basename(model_id)
                            if name.endswith(".gguf"):
                                name = name[:-5]
                            m = {
                                "name": name,
                                "digest": (getattr(self.config, "model_digest", "") if self.config else "") or name
                            }
                            models.append(m)
                    return models
        except Exception:
            pass
        return []

    async def get_context_limit(self) -> int:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/props")
                if resp.status_code == 200:
                    props = resp.json()
                    n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
                    if n_ctx:
                        return int(n_ctx)
        except Exception:
            pass
        return 4096

    async def get_model_info(self, model: str) -> dict:
        return {
            "modelfile": f"FROM {model}",
            "parameters": "",
            "template": "",
            "details": {
                "format": "gguf",
                "family": "llama"
            }
        }

    async def delete_model(self, model: str) -> bool:
        return False

    async def pull_model(self, model: str):
        yield {"error": "Pulling models not supported by llama.cpp"}


class ProviderClient:
    def __init__(self, config: ConfigManager, status_callback=None, log_callback=None, restart_callback=None):
        self.config = config
        self.status_callback = status_callback  # Callback for connection/status updates
        self.log_callback = log_callback        # Callback for provider logs
        self.restart_callback = restart_callback
        self.running = False
        self.soft_stopping = False
        self.websocket = None
        self.current_jobs = 0  # Accepted jobs currently being processed
        self.active_jobs = {}           # Active Ollama execution tasks
        self.completed_jobs_count = 0
        self.blacklist_path = Path(os.path.expanduser("~/.thinkfarm/blacklisted_models.json"))
        self.blacklisted_models = self.load_blacklist()
        self.performance_data = {}
        self.loaded_models = set()
        self.model_job_history = {}  # model_name -> list of (throughput, duration)
        self.last_full_status_time = 0.0
        self.last_model_names = None
        self.last_optimization_time = 0.0
        self.context_limits = {}
        self.load_context_limits()
        self.performance_baselines = {}
        self.load_performance_baselines()
        self.last_inference_time = time.time()
        self.startup_model_loaded = False
        self.restarting_lcpp = False

    def load_context_limits(self):
        try:
            from context_prober import load_context_limits as load_limits
            self.context_limits = load_limits()
            if self.config.model_name and self.context_limits:
                for name, limit in list(self.context_limits.items()):
                    if name != self.config.model_name:
                        self.context_limits[self.config.model_name] = limit
                        break
        except Exception as e:
            self.log(f"Failed to load context limits: {e}", logging.ERROR)
            self.context_limits = {}

    def load_performance_baselines(self):
        try:
            from context_prober import load_performance_baselines as load_baselines
            baselines_data = load_baselines()
            self.performance_baselines = baselines_data.get("baselines", {})
            if self.config.model_name and self.performance_baselines:
                for name, baseline in list(self.performance_baselines.items()):
                    if name != self.config.model_name:
                        self.performance_baselines[self.config.model_name] = baseline
                        break
        except Exception as e:
            self.log(f"Failed to load performance baselines: {e}", logging.ERROR)
            self.performance_baselines = {}

    def log(self, message: str, level=logging.INFO):
        logger.log(level, message)
        if self.log_callback:
            self.log_callback(message)

    def load_blacklist(self):
        try:
            if self.blacklist_path.exists():
                with open(self.blacklist_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            self.log(f"Failed to load blacklist: {e}", logging.ERROR)
        return []

    def save_blacklist(self):
        try:
            self.blacklist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.blacklist_path, "w") as f:
                json.dump(self.blacklisted_models, f)
        except Exception as e:
            self.log(f"Failed to save blacklist: {e}", logging.ERROR)

    async def get_performance_data(self):
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.config.server_url.rstrip('/')}/api/performance"
                resp = await client.get(url)
                if resp.status_code == 200:
                    self.performance_data = resp.json()
                    self.log(f"Fetched global performance data: {self.performance_data}")
        except Exception as e:
            self.log(f"Failed to fetch global performance: {e}", logging.WARNING)

    async def get_local_models(self, only_probed=False, force=False):
        now = time.time()
        if force or not hasattr(self, "_cached_raw_models") or self._cached_raw_models is None or now - getattr(self, "_last_raw_models_time", 0.0) >= 30:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{self.config.local_lcpp_url.rstrip('/')}/v1/models")
                    if resp.status_code == 200:
                        data = resp.json().get("data", [])
                        models = []
                        for m in data:
                            model_id = m.get("id")
                            if model_id:
                                name = os.path.basename(model_id)
                                if name.endswith(".gguf"):
                                    name = name[:-5]
                                m_dict = {
                                    "name": name,
                                    "digest": getattr(self.config, "model_digest", "") or name
                                }
                                models.append(m_dict)
                        self._cached_raw_models = models
                        self._last_raw_models_time = now
                    else:
                        if not hasattr(self, "_cached_raw_models"):
                            self._cached_raw_models = []
            except Exception as e:
                self.log(f"Local llama.cpp connection failed: {e}", logging.WARNING)
                if not hasattr(self, "_cached_raw_models"):
                    self._cached_raw_models = []
        
        models = self._cached_raw_models
        active_models = []
        seen_names = set()
        for m in models:
            name = m.get("name")
            if not name:
                continue
            m_copy = m.copy()
            if self.config.model_name:
                name = self.config.model_name
                m_copy["name"] = name
            m_copy["digest"] = getattr(self.config, "model_digest", "") or name
            if name.startswith("thinkfarm-"):
                name = name[10:]
                m_copy["name"] = name
            if only_probed:
                limit = self.context_limits.get(name)
                if limit is None or limit == 0:
                    continue
            if name not in self.blacklisted_models:
                if name not in seen_names:
                    active_models.append(m_copy)
                    seen_names.add(name)
        return active_models

    async def get_loaded_models(self):
        local_models = await self.get_local_models()
        loaded = set(m.get("name") for m in local_models if m.get("name"))
        self.loaded_models = loaded
        return list(loaded)

    async def send_status(self, force_full: bool = False):
        if not self.websocket:
            return
        
        models = await self.get_local_models(only_probed=True, force=force_full)
        model_names = {m.get("name") for m in models if m.get("name")}
        
        now = time.time()
        models_changed = (self.last_model_names is None) or (model_names != self.last_model_names)
        time_expired = (now - self.last_full_status_time >= 1200)
        
        if force_full or models_changed or time_expired:
            loaded = await self.get_loaded_models()
            
            limits = self.context_limits
            cfg_limit = getattr(self.config, "lccp_context", 0) or 0
            context_limits = {}
            for m in models:
                name = m.get("name")
                if name:
                    # Config lccp_context takes precedence; -1 is unlimited
                    if cfg_limit == -1:
                        context_limits[name] = -1
                    elif cfg_limit > 0:
                        context_limits[name] = cfg_limit
                    else:
                        # Use 90% of cached limit if available and positive, else fallback to 8192
                        limit = limits.get(name)
                        if limit is not None and limit > 0:
                            context_limits[name] = int(limit * 0.9)
                        elif limit == -1:
                            context_limits[name] = -1
                        else:
                            context_limits[name] = 8192
            
            status_msg = {
                "type": "status",
                "provider_id": self.config.provider_id,
                "connected_at": datetime.utcnow().isoformat() + "Z",
                "models": models,
                "loaded_models": loaded,
                "context_limits": context_limits,
                "slots": getattr(self.config, "slots", 1),
                "is_busy": len(self.active_jobs) >= getattr(self.config, "slots", 1)
            }
            
            try:
                await self.websocket.send(json.dumps(status_msg))
                self.log("Sent provider status update")
                self.last_full_status_time = now
                self.last_model_names = model_names
            except Exception as e:
                self.log(f"Failed to send status update: {e}", logging.ERROR)
        else:
            try:
                await self.websocket.send(json.dumps({"type": "ping"}))
                # self.log("Sent ping")
            except Exception as e:
                self.log(f"Failed to send ping: {e}", logging.ERROR)

    async def run(self):
        self.running = True
        self.soft_stopping = False
        self.probing_triggered = False
        self.log(f"Starting Provider Client ({self.config.provider_id})...")
        await self.get_performance_data()
        
        while self.running:
            # Check if auto manage models is enabled and GB allowed is set
            if self.config.auto_manage_models and self.config.gb_allowed > 0:
                now = time.time()
                if now - self.last_optimization_time >= 3600:
                    self.last_optimization_time = now
                    self.log("Running scheduled portfolio optimization...")
                    try:
                        from model_manager import ModelManager
                        lcpp_client = LlamaCppClient(self.config.local_lcpp_url, self.config)
                        manager = ModelManager(lcpp_client, self.config.server_url)
                        newly_pulled, priority_model = await manager.optimize_portfolio(self.config.gb_allowed)
                        self.log(f"Portfolio optimization complete. Newly pulled: {newly_pulled}")
                    except Exception as e:
                        self.log(f"Error during portfolio optimization: {e}", logging.ERROR)

            # Check for new models first before connecting
            if await self.check_for_new_models():
                self.log("New models found. Running context probing routine...")
                if self.status_callback:
                    self.status_callback("Probing")
                
                try:
                    from context_prober import run_context_probing, load_context_limits
                    models = await self.get_local_models(only_probed=False)
                    model_names = [m.get("name") for m in models if m.get("name")]
                    limits = load_context_limits()
                    await run_context_probing(self.config.local_lcpp_url, model_names, limits)
                    self.load_context_limits()
                    self.load_performance_baselines()
                except Exception as e:
                    self.log(f"Error running context probing: {e}", logging.ERROR)
                
                self.soft_stopping = False
                self.probing_triggered = False
                continue

            if not self.startup_model_loaded:
                self.startup_model_loaded = True
                asyncio.create_task(self.load_most_desirable_model())

            try:
                ws_scheme = "wss" if self.config.server_url.startswith("https") else "ws"
                host = self.config.server_url.split("://")[-1]
                ws_url = f"{ws_scheme}://{host}/ws/provider/{self.config.provider_id}"
                
                self.log(f"Connecting to WebSocket: {ws_url}")
                if self.status_callback:
                    self.status_callback("Connecting")
                
                async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
                    self.websocket = ws
                    self.log("WebSocket connected successfully")
                    if self.status_callback:
                        self.status_callback("Connected")
                    
                    # Send initial status
                    await self.send_status(force_full=True)
                    
                    # Status loop
                    async def status_heartbeat():
                        while self.running and self.websocket == ws:
                            await asyncio.sleep(5)
                            if await self.check_for_new_models():
                                self.log("New model detected during heartbeat! Initiating soft stop for probing...")
                                asyncio.create_task(self.soft_disconnect_for_probing())
                                break
                            await self.send_status()
                            await self.run_heartbeat_check()

                    heartbeat_task = asyncio.create_task(status_heartbeat())
                    
                    try:
                        async for message in ws:
                            data = json.loads(message)
                            await self.handle_message(data)
                    finally:
                        heartbeat_task.cancel()
                        
            except Exception as e:
                self.websocket = None
                if self.status_callback:
                    if getattr(self, "restarting_lcpp", False):
                        pass # status is set by restart_lcpp
                    else:
                        self.status_callback("Disconnected")
                if self.running and not self.soft_stopping:
                    if getattr(self, "probing_triggered", False):
                        self.probing_triggered = False
                    elif getattr(self, "restarting_lcpp", False):
                        await asyncio.sleep(2)
                    else:
                        self.log(f"WebSocket connection error: {e}. Reconnecting in 5 seconds...", logging.ERROR)
                        await asyncio.sleep(5)
                else:
                    if not getattr(self, "probing_triggered", False) and not getattr(self, "restarting_lcpp", False):
                        self.running = False
                        self.log("Provider connection stopped.")
                    else:
                        self.probing_triggered = False
        
        self.websocket = None
        if self.status_callback:
            self.status_callback("Stopped")

    async def stop(self):
        self.running = False
        if self.websocket:
            await self.websocket.close()
        for job_id, task in list(self.active_jobs.items()):
            task.cancel()
        self.log("Provider Client stopped.")

    async def soft_stop(self):
        self.log("Initiating soft stop...")
        self.soft_stopping = True
        if self.status_callback:
            self.status_callback("Stopping")
        if self.current_jobs == 0:
            self.log("No active or pending jobs. Stopping provider immediately.")
            self.running = False
            if self.websocket:
                await self.websocket.close()
        else:
            self.log(f"Waiting for {self.current_jobs} jobs to complete before stopping.")

    async def check_for_new_models(self) -> bool:
        """Check if there are any local models that need context probing or performance baseline."""
        try:
            models = await self.get_local_models()
            model_names = [m.get("name") for m in models if m.get("name")]
            
            limits = self.context_limits
            baselines = self.performance_baselines
            
            for m in model_names:
                if m not in limits or limits[m] == 0:
                    return True
                if limits[m] > 0 and m not in baselines:
                    return True
        except Exception as e:
            self.log(f"Error checking for new models: {e}", logging.WARNING)
        return False

    async def soft_disconnect_for_probing(self):
        """Soft stop connection to WebSocket for context probing when new model is found."""
        self.log("Initiating soft stop for context probing...")
        self.probing_triggered = True
        self.soft_stopping = True
        if self.status_callback:
            self.status_callback("Stopping")
        
        while self.current_jobs > 0:
            self.log(f"Waiting for {self.current_jobs} jobs to complete before probing.")
            await asyncio.sleep(0.5)
            
        self.log("All active jobs completed. Closing WebSocket to run context prober.")
        if self.websocket:
            await self.websocket.close()

    async def handle_message(self, msg: dict):
        msg_type = msg.get("type")
        self.log(f"Received message of type: {msg_type}")
        
        if msg_type == "job_published":
            if self.soft_stopping:
                self.log("Ignoring new job advertisement during soft stop.")
                return
            job_id = msg.get("job_id")
            model = msg.get("model")
            digest = msg.get("digest")
            
            # Check if we support this model by digest and are not blacklisted
            probed_models = await self.get_local_models(only_probed=True)
            matched_model = next((m for m in probed_models if digest and m.get("digest") == digest), None)

            if matched_model:
                local_model_name = matched_model.get("name")
                # Check context limit if num_ctx is specified
                num_ctx = msg.get("num_ctx")
                if num_ctx is not None and num_ctx > 0:
                    cfg_limit = getattr(self.config, "lccp_context", 0) or 0
                    if cfg_limit == -1:
                        pass  # unlimited via config
                    elif cfg_limit > 0:
                        if num_ctx > cfg_limit:
                            self.log(f"Ignoring job {job_id} for digest {digest} ({local_model_name}): requested num_ctx {num_ctx} exceeds config lccp_context {cfg_limit}")
                            return
                    else:
                        limit = self.context_limits.get(local_model_name)
                        if limit is not None and limit > 0:
                            effective_limit = int(limit * 0.9)
                        elif limit == -1:
                            effective_limit = -1
                        else:
                            effective_limit = 8192
                        if effective_limit != -1:  # -1 is unlimited
                            if num_ctx > effective_limit:
                                self.log(f"Ignoring job {job_id} for digest {digest} ({local_model_name}): requested num_ctx {num_ctx} exceeds limit {effective_limit}")
                                return

                async def process_acceptance():
                    max_slots = max(1, getattr(self.config, "slots", 1))
                    current_active = len(self.active_jobs)
                    
                    delay = 0.0
                    if current_active >= max_slots:
                        # Fully busy -> 2s delay
                        delay += 2.0
                    # Partially busy -> no delay
                        
                    # Model unloaded -> 2s delay
                    if local_model_name not in self.loaded_models:
                        delay += 2.0
                    
                    if delay > 0:
                        self.log(f"Delaying acceptance of job {job_id} by {delay:.1f}s (active: {current_active}/{max_slots}, loaded: {local_model_name in self.loaded_models})")
                        await asyncio.sleep(delay)
                        
                    if self.soft_stopping or not self.websocket:
                        return
                        
                    self.log(f"Accepting job {job_id} for digest {digest} ({local_model_name})")
                    accept_msg = {
                        "type": "accept",
                        "job_id": job_id,
                        "provider_id": self.config.provider_id
                    }
                    try:
                        await self.websocket.send(json.dumps(accept_msg))
                        # Send status showing we are busy (optimistic)
                        await self.send_status()
                    except Exception as e:
                        self.log(f"Failed to send acceptance for job {job_id}: {e}", logging.WARNING)

                asyncio.create_task(process_acceptance())
                
        elif msg_type == "job_assigned":
            job_id = msg.get("job_id")
            endpoint = msg.get("endpoint")
            body = msg.get("body")
            digest = msg.get("digest")
            
            # Ensure the model gets loaded in the background if it's not already loaded
            if digest:
                is_embed_endpoint = endpoint in ("embed", "embeddings")
                probed_models = await self.get_local_models(only_probed=True)
                matched = next((m for m in probed_models if m.get("digest") == digest), None)
                if matched:
                    local_name = matched.get("name")
                    using_custom = not is_embed_endpoint
                    actual_model = f"thinkfarm-{local_name}" if using_custom and local_name else local_name
                    if local_name and local_name not in self.loaded_models and actual_model not in self.loaded_models:
                        asyncio.create_task(self.keep_model_loaded(actual_model, is_embed_endpoint))

            # Start job in background task
            self.current_jobs += 1
            task = asyncio.create_task(self.execute_job(job_id, endpoint, body, digest=digest))
            self.active_jobs[job_id] = task
            await self.send_status()
            
        elif msg_type == "cancel_job":
            job_id = msg.get("job_id")
            if job_id in self.active_jobs:
                self.log(f"Cancelling job {job_id}")
                self.active_jobs[job_id].cancel()
                del self.active_jobs[job_id]
                await self.send_status()

        elif msg_type == "error":
            self.log(f"Server error: {msg.get('detail')}", logging.ERROR)

    async def execute_job(self, job_id: str, endpoint: str, body: dict, digest: str = None):
        self.last_inference_time = time.time()
        # Determine local model mapping via digest
        requested_model = body.get("model") if body else None
        is_embed_endpoint = endpoint in ("embed", "embeddings")

        probed_models = await self.get_local_models(only_probed=True)
        matched = next((m for m in probed_models if digest and m.get("digest") == digest), None)
        local_name = matched.get("name") if matched else requested_model

        using_custom = not is_embed_endpoint
        actual_model = f"thinkfarm-{local_name}" if using_custom and local_name else local_name
        
        if using_custom and requested_model and actual_model:
            body["model"] = actual_model
            if actual_model != requested_model:
                self.log(f"Mapping requested model '{requested_model}' to local custom model '{actual_model}'")

        is_ollama_format = endpoint in ("chat", "generate", "embed", "embeddings", "show")
        
        # Translate endpoint
        if endpoint in ("chat", "v1/chat/completions"):
            lcpp_path = "/v1/chat/completions"
        elif endpoint in ("generate", "v1/completions"):
            lcpp_path = "/v1/completions"
        elif endpoint in ("embed", "embeddings", "v1/embeddings"):
            lcpp_path = "/v1/embeddings"
        elif endpoint == "show":
            mock_show = {
                "modelfile": f"FROM {requested_model}",
                "parameters": "",
                "template": "",
                "details": {
                    "format": "gguf",
                    "family": "llama"
                }
            }
            await self.websocket.send(json.dumps({
                "type": "chunk",
                "job_id": job_id,
                "data": json.dumps(mock_show)
            }))
            await self.websocket.send(json.dumps({
                "type": "job_done",
                "job_id": job_id,
                "eval_count": 0,
                "prompt_eval_count": 0,
                "is_busy": len(self.active_jobs) >= getattr(self.config, "slots", 1)
            }))
            return
        else:
            lcpp_path = f"/{endpoint.lstrip('/')}"

        url = f"{self.config.local_lcpp_url.rstrip('/')}{lcpp_path}"
        
        # Translate Ollama options/parameters to OpenAI parameters
        translated_body = {}
        if is_ollama_format:
            translated_body["model"] = requested_model or ""
            translated_body["stream"] = body.get("stream", False)
            
            if endpoint == "chat":
                translated_body["messages"] = body.get("messages", [])
            elif endpoint == "generate":
                translated_body["prompt"] = body.get("prompt", "")
            elif endpoint in ("embed", "embeddings"):
                translated_body["input"] = body.get("input", "")
            
            options = body.get("options", {})
            for k, val in options.items():
                if k == "temperature":
                    translated_body["temperature"] = val
                elif k == "top_p":
                    translated_body["top_p"] = val
                elif k == "top_k":
                    translated_body["top_k"] = val
                elif k == "seed":
                    translated_body["seed"] = val
                elif k == "num_predict":
                    translated_body["max_tokens"] = val
                elif k == "stop":
                    translated_body["stop"] = val
        else:
            translated_body = body
            
        if "keep_alive" in translated_body:
            del translated_body["keep_alive"]
        if "stream_options" in translated_body:
            del translated_body["stream_options"]

        is_stream = translated_body.get("stream", False)

        self.log(f"Executing job {job_id} on local llama.cpp: {lcpp_path}")
        start_time = time.time()
        eval_count = 0
        prompt_eval_count = 0
        has_output = False

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if is_stream:
                    # Stream response chunking
                    req = client.build_request("POST", url, json=translated_body)
                    resp = await client.send(req, stream=True)
                    
                    if resp.status_code != 200:
                        err_content = await resp.aread()
                        raise Exception(f"Local llama.cpp returned {resp.status_code}: {err_content}")
                    
                    # Accumulate and batch chunks in ~75ms windows
                    buffer = []
                    line_buffer = ""
                    last_send_time = time.time()
                    
                    separator = "\n\n" if "v1/" in lcpp_path else "\n"
                    
                    async for chunk_bytes in resp.aiter_bytes():
                        chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
                        buffer.append(chunk_str)
                        
                        # Process complete lines for JSON parsing
                        line_buffer += chunk_str
                        while separator in line_buffer:
                            line, line_buffer = line_buffer.split(separator, 1)
                            if not line.strip():
                                continue
                            try:
                                if "v1/" in lcpp_path:
                                    if line.startswith("data:"):
                                        line_clean = line[5:].strip()
                                        if line_clean != "[DONE]":
                                            chunk_json = json.loads(line_clean)
                                            if "usage" in chunk_json and chunk_json["usage"]:
                                                eval_count = chunk_json["usage"].get("completion_tokens", eval_count)
                                                prompt_eval_count = chunk_json["usage"].get("prompt_tokens", prompt_eval_count)
                                                has_output = True
                                            elif "choices" in chunk_json and chunk_json["choices"]:
                                                choice = chunk_json["choices"][0]
                                                if "delta" in choice and choice["delta"].get("content"):
                                                    has_output = True
                                                    eval_count += 1
                                else:
                                    chunk_json = json.loads(line)
                                    if "eval_count" in chunk_json:
                                        eval_count = chunk_json.get("eval_count", eval_count)
                                        prompt_eval_count = chunk_json.get("prompt_eval_count", prompt_eval_count)
                                    if chunk_json.get("response") or chunk_json.get("message", {}).get("content"):
                                        has_output = True
                            except Exception:
                                pass
                        
                        current_time = time.time()
                        if current_time - last_send_time >= 0.075:
                            combined_data = "".join(buffer)
                            
                            # If client requested Ollama format, translate back to Ollama format
                            if is_ollama_format and "v1/" in lcpp_path:
                                translated_chunks = []
                                # Split combined data by data: to process multiple chunks
                                for chunk_part in combined_data.split("data:"):
                                    chunk_part = chunk_part.strip()
                                    if not chunk_part or chunk_part == "[DONE]":
                                        continue
                                    try:
                                        chunk_json = json.loads(chunk_part)
                                        if "choices" in chunk_json and chunk_json["choices"]:
                                            choice = chunk_json["choices"][0]
                                            delta = choice.get("delta", {})
                                            content = delta.get("content", "")
                                            if content:
                                                if endpoint == "chat":
                                                    translated_chunks.append(json.dumps({
                                                        "model": requested_model,
                                                        "created_at": datetime.utcnow().isoformat() + "Z",
                                                        "message": {"role": "assistant", "content": content},
                                                        "done": False
                                                    }) + "\n")
                                                elif endpoint == "generate":
                                                    translated_chunks.append(json.dumps({
                                                        "model": requested_model,
                                                        "created_at": datetime.utcnow().isoformat() + "Z",
                                                        "response": content,
                                                        "done": False
                                                    }) + "\n")
                                    except Exception:
                                        pass
                                combined_data = "".join(translated_chunks)
                            
                            if combined_data:
                                await self.websocket.send(json.dumps({
                                    "type": "chunk",
                                    "job_id": job_id,
                                    "data": combined_data
                                }))
                            buffer = []
                            last_send_time = current_time
                            
                    # Parse remaining line buffer
                    if line_buffer.strip():
                        try:
                            line = line_buffer
                            if "v1/" in lcpp_path:
                                if line.startswith("data:"):
                                    line_clean = line[5:].strip()
                                    if line_clean != "[DONE]":
                                        chunk_json = json.loads(line_clean)
                                        if "usage" in chunk_json and chunk_json["usage"]:
                                            eval_count = chunk_json["usage"].get("completion_tokens", eval_count)
                                            prompt_eval_count = chunk_json["usage"].get("prompt_tokens", prompt_eval_count)
                                            has_output = True
                            else:
                                chunk_json = json.loads(line)
                                if "eval_count" in chunk_json:
                                    eval_count = chunk_json.get("eval_count", eval_count)
                                    prompt_eval_count = chunk_json.get("prompt_eval_count", prompt_eval_count)
                                if chunk_json.get("response") or chunk_json.get("message", {}).get("content"):
                                    has_output = True
                        except Exception:
                            pass
                            
                    # Send remaining buffer
                    if buffer:
                        combined_data = "".join(buffer)
                        if is_ollama_format and "v1/" in lcpp_path:
                            translated_chunks = []
                            for chunk_part in combined_data.split("data:"):
                                chunk_part = chunk_part.strip()
                                if not chunk_part or chunk_part == "[DONE]":
                                    continue
                                try:
                                    chunk_json = json.loads(chunk_part)
                                    if "choices" in chunk_json and chunk_json["choices"]:
                                        choice = chunk_json["choices"][0]
                                        delta = choice.get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            if endpoint == "chat":
                                                translated_chunks.append(json.dumps({
                                                    "model": requested_model,
                                                    "created_at": datetime.utcnow().isoformat() + "Z",
                                                    "message": {"role": "assistant", "content": content},
                                                    "done": False
                                                }) + "\n")
                                            elif endpoint == "generate":
                                                translated_chunks.append(json.dumps({
                                                    "model": requested_model,
                                                    "created_at": datetime.utcnow().isoformat() + "Z",
                                                    "response": content,
                                                    "done": False
                                                }) + "\n")
                                except Exception:
                                    pass
                            combined_data = "".join(translated_chunks)
                        
                        # Send final done chunk for Ollama format if streaming
                        if is_ollama_format and "v1/" in lcpp_path:
                            if endpoint == "chat":
                                combined_data += json.dumps({
                                    "model": requested_model,
                                    "created_at": datetime.utcnow().isoformat() + "Z",
                                    "done": True
                                }) + "\n"
                            elif endpoint == "generate":
                                combined_data += json.dumps({
                                    "model": requested_model,
                                    "created_at": datetime.utcnow().isoformat() + "Z",
                                    "done": True
                                }) + "\n"
                        
                        if combined_data:
                            await self.websocket.send(json.dumps({
                                "type": "chunk",
                                "job_id": job_id,
                                "data": combined_data
                            }))
                else:
                    # Non-streaming
                    resp = await client.post(url, json=translated_body)
                    if resp.status_code != 200:
                        raise Exception(f"Local llama.cpp returned {resp.status_code}: {resp.text}")
                    
                    resp_json = resp.json()
                    
                    if "usage" in resp_json and resp_json["usage"]:
                        eval_count = resp_json["usage"].get("completion_tokens", 0)
                        prompt_eval_count = resp_json["usage"].get("prompt_tokens", 0)
                    else:
                        eval_count = resp_json.get("eval_count", 0)
                        prompt_eval_count = resp_json.get("prompt_eval_count", 0)
                    
                    if eval_count > 0:
                        has_output = True
                    elif resp_json.get("response") or resp_json.get("message", {}).get("content"):
                        has_output = True
                    elif "choices" in resp_json and resp_json["choices"]:
                        choice = resp_json["choices"][0]
                        if choice.get("text") or choice.get("message", {}).get("content"):
                            has_output = True
                    
                    # Translate non-streaming response back to Ollama format if requested
                    if is_ollama_format and "v1/" in lcpp_path:
                        if endpoint == "chat":
                            content = ""
                            role = "assistant"
                            if "choices" in resp_json and resp_json["choices"]:
                                choice = resp_json["choices"][0]
                                message = choice.get("message", {})
                                content = message.get("content", "")
                                role = message.get("role", "assistant")
                            
                            translated_resp = {
                                "model": requested_model,
                                "created_at": datetime.utcnow().isoformat() + "Z",
                                "message": {"role": role, "content": content},
                                "done": True,
                                "eval_count": eval_count,
                                "prompt_eval_count": prompt_eval_count
                            }
                        elif endpoint == "generate":
                            content = ""
                            if "choices" in resp_json and resp_json["choices"]:
                                choice = resp_json["choices"][0]
                                content = choice.get("text", "")
                            
                            translated_resp = {
                                "model": requested_model,
                                "created_at": datetime.utcnow().isoformat() + "Z",
                                "response": content,
                                "done": True,
                                "eval_count": eval_count,
                                "prompt_eval_count": prompt_eval_count
                            }
                        elif endpoint in ("embed", "embeddings"):
                            embeddings = []
                            if "data" in resp_json:
                                for item in resp_json["data"]:
                                    embeddings.append(item.get("embedding", []))
                            
                            if endpoint == "embeddings":
                                translated_resp = {
                                    "embedding": embeddings[0] if embeddings else []
                                }
                            else:
                                translated_resp = {
                                    "embeddings": embeddings
                                }
                        else:
                            translated_resp = resp_json
                    else:
                        translated_resp = resp_json
                    
                    await self.websocket.send(json.dumps({
                        "type": "chunk",
                        "job_id": job_id,
                        "data": json.dumps(translated_resp)
                    }))
                    
                    if eval_count > 0:
                        has_output = True
                    elif resp_json.get("response") or resp_json.get("message", {}).get("content"):
                        has_output = True
                    elif "choices" in resp_json and resp_json["choices"]:
                        choice = resp_json["choices"][0]
                        if choice.get("text") or choice.get("message", {}).get("content"):
                            has_output = True

            duration_ns = int((time.time() - start_time) * 1e9)
            self.log(f"Job {job_id} executed successfully. Duration: {duration_ns / 1e9:.2f}s, Tokens: {eval_count}")
            
            # Send job_done
            done_msg = {
                "type": "job_done",
                "job_id": job_id,
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
                "total_duration": duration_ns,
                "is_busy": len(self.active_jobs) >= getattr(self.config, "slots", 1)
            }
            await self.websocket.send(json.dumps(done_msg))
            
            # Performance monitoring (Slope Monitor & Zero-Eval detection)
            await self.monitor_performance(requested_model, eval_count, duration_ns, endpoint, has_output)

        except asyncio.CancelledError:
            self.log(f"Job {job_id} was cancelled.")
            raise  # re-raise so the task is properly marked cancelled (finally still runs first)
        except Exception as e:
            self.log(f"Failed to execute job {job_id}: {e}", logging.ERROR)
        finally:
            self.last_inference_time = time.time()
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            self.current_jobs -= 1
            self.completed_jobs_count += 1
            await self.send_status(force_full=True)
            # Proactive reconnection after 30 jobs
            if not self.soft_stopping and self.completed_jobs_count >= 30 and len(self.active_jobs) == 0:
                self.log("Completed 30 jobs. Reconnecting WebSocket to refresh routing...")
                self.completed_jobs_count = 0
                if self.websocket:
                    await self.websocket.close()

            if self.soft_stopping and self.current_jobs == 0:
                self.log("All jobs completed during soft stop. Closing websocket and stopping provider.")
                self.running = False
                if self.websocket:
                    await self.websocket.close()

    async def monitor_performance(self, model: str, eval_count: int, duration_ns: int, endpoint: str, has_output: bool):
        # Skip performance/zero-eval monitoring for embedding endpoints
        if endpoint in ("embed", "embeddings"):
            return

        # 1. Zero-Evaluation Detection
        if eval_count == 0 and not has_output:
            self.log(f"Zero evaluation detected for model {model}. Running sanity check test...")
            sanity_success = await self.run_sanity_check(model)
            if not sanity_success:
                self.log(f"Sanity check failed for model {model}! Triggering llama.cpp restart...", logging.CRITICAL)
                asyncio.create_task(self.restart_lcpp())
            return
            
        # 2. Slope Monitor
        duration_s = duration_ns / 1e9
        if duration_s > 0:
            tps = eval_count / duration_s
            self.log(f"Model {model} throughput: {tps:.2f} tokens/s")
            
            history = self.model_job_history.setdefault(model, [])
            history.append((tps, duration_s))
            if len(history) > 5:
                history.pop(0)
                
            # Perform slope monitor blacklist check
            # Fetch /api/performance maps model to a peak performance.
            if self.performance_data and model in self.performance_data:
                global_peak = self.performance_data[model].get("peak", 0)
                threshold = global_peak / 3
                
                # Check 3 consecutive job completions with duration >= 10s falling below threshold
                slow_jobs = [h for h in history if h[1] >= 10.0]
                if len(slow_jobs) >= 3 and all(h[0] < threshold for h in slow_jobs[-3:]):
                    avg_slow_tps = sum(h[0] for h in slow_jobs[-3:]) / 3
                    self.log(f"Slow jobs detected for model {model} (avg throughput: {avg_slow_tps:.2f} t/s below threshold {threshold:.2f} t/s).")
                    asyncio.create_task(self.handle_slow_jobs_routine(model, avg_slow_tps))

    async def run_sanity_check(self, model: str) -> bool:
        try:
            url = f"{self.config.local_lcpp_url.rstrip('/')}/v1/completions"
            body = {
                "model": model,
                "prompt": "hello",
                "max_tokens": 1
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=body)
                return resp.status_code == 200
        except Exception:
            return False

    async def keep_model_loaded(self, model_name: str, is_embed: bool = False):
        self.loaded_models.add(model_name)

    async def get_raw_loaded_models(self) -> list:
        return list(await self.get_loaded_models())

    async def run_heartbeat_check(self):
        if not self.running or self.current_jobs > 0:
            return
        
        now = time.time()
        if now - self.last_inference_time >= 1200:  # 20 minutes
            self.last_inference_time = now
            self.log("Sending heartbeat check to local llama.cpp...")
            try:
                url = f"{self.config.local_lcpp_url.rstrip('/')}/v1/models"
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        self.log("Heartbeat check successful (llama.cpp online)")
                    else:
                        self.log(f"Heartbeat check failed (status {resp.status_code})", logging.WARNING)
            except Exception as e:
                self.log(f"Heartbeat check exception: {e}", logging.WARNING)

    async def load_most_desirable_model(self):
        if not self.running or self.current_jobs > 0 or getattr(self, "soft_stopping", False) or getattr(self, "probing_triggered", False):
            return
        if await self.check_for_new_models():
            return
        if await self.get_raw_loaded_models():
            return

        self.log("llama.cpp serves a single pre-loaded model. Querying local model status...")
        try:
            local_models = await self.get_local_models()
            for m in local_models:
                name = m.get("name")
                if name:
                    self.loaded_models.add(name)
            self.log(f"Active loaded models: {list(self.loaded_models)}")
        except Exception as e:
            self.log(f"Failed to query local models: {e}", logging.WARNING)

    async def restart_lcpp(self) -> str:
        self.log("Initiating llama.cpp restart procedure...")
        self.restarting_lcpp = True
        
        # 1) Disconnect the websocket
        if self.websocket:
            self.log("Disconnecting WebSocket...")
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None
            
        # 2) Run the restart command
        if self.config.managed_lcpp and hasattr(self, 'restart_callback') and self.restart_callback:
            self.log("Invoking managed llama.cpp restart callback...")
            try:
                if asyncio.iscoroutinefunction(self.restart_callback):
                    await self.restart_callback()
                else:
                    self.restart_callback()
            except Exception as e:
                self.log(f"Managed llama.cpp restart callback failed: {e}", logging.ERROR)
        else:
            cmd = self.config.lcpp_restart_cmd.strip()
            if not cmd:
                self.log("No llama.cpp restart command configured. Skipping command execution.", logging.WARNING)
            else:
                self.log(f"Running restart command: {cmd}")
                try:
                    # Run the command asynchronously
                    proc = await asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()
                    self.log(f"Restart command return code: {proc.returncode}")
                    if stdout:
                        self.log(f"Restart stdout: {stdout.decode().strip()}")
                    if stderr:
                        self.log(f"Restart stderr: {stderr.decode().strip()}", logging.ERROR if proc.returncode != 0 else logging.INFO)
                except Exception as e:
                    err_msg = f"Failed to run restart command: {e}"
                    self.log(err_msg, logging.ERROR)
                    if self.status_callback:
                        self.status_callback(f"Restart failed: {e}")
                    self.restarting_lcpp = False
                    self.running = False
                    return "error"
                
        # 3) Wait until the v1/models endpoint is responsive
        self.log("Waiting for local llama.cpp /v1/models endpoint to respond...")
        models_url = f"{self.config.local_lcpp_url.rstrip('/')}/v1/models"
        is_responsive = False
        # Poll every 5 seconds for up to 3 minutes (36 attempts)
        for attempt in range(36):
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(models_url)
                    if resp.status_code == 200:
                        is_responsive = True
                        break
            except Exception:
                pass
            await asyncio.sleep(5.0)
            
        if not is_responsive:
            err_msg = "llama.cpp endpoint is not responsive after restart."
            self.log(err_msg, logging.ERROR)
            if self.status_callback:
                self.status_callback("Error: llama.cpp offline")
            self.restarting_lcpp = False
            self.running = False
            return "offline"
            
        # 4) Run a basic test prompt
        priority_model = ""
        try:
            local_models = await self.get_local_models()
            if local_models:
                priority_model = local_models[0].get("name", "")
        except Exception as e:
            self.log(f"Could not determine model to test: {e}", logging.WARNING)
            
        if not priority_model:
            err_msg = "No local models available for performance testing."
            self.log(err_msg, logging.ERROR)
            if self.status_callback:
                self.status_callback(f"Error: {err_msg}")
            self.restarting_lcpp = False
            self.running = False
            return "error"
            
        self.log(f"Running performance test prompt on: {priority_model}")
        baseline_slope = None
        baseline_info = self.performance_baselines.get(priority_model)
        if baseline_info and "slope" in baseline_info:
            baseline_slope = baseline_info["slope"]
            self.log(f"Baseline performance for {priority_model}: {baseline_slope:.2f} t/s")
        else:
            self.log(f"No baseline performance recorded for {priority_model}.", logging.WARNING)
            
        test_url = f"{self.config.local_lcpp_url.rstrip('/')}/v1/completions"
        payload = {
            "model": priority_model,
            "prompt": "what is the history of Sweden?",
            "max_tokens": 50
        }
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(test_url, json=payload)
                if resp.status_code != 200:
                    raise Exception(f"llama-server returned status code {resp.status_code}")
                gen_data = resp.json()
                
                usage = gen_data.get("usage", {})
                eval_count = usage.get("completion_tokens", 0)
                prompt_eval_count = usage.get("prompt_tokens", 0)
                
                compute_seconds = time.time() - start_time
                if compute_seconds <= 0:
                    compute_seconds = 1.0
                    
                measured_slope = (eval_count + 0.003 * prompt_eval_count) / compute_seconds
                self.log(f"Measured test slope: {measured_slope:.2f} t/s")
        except Exception as e:
            err_msg = f"Performance test prompt failed: {e}"
            self.log(err_msg, logging.ERROR)
            if self.status_callback:
                self.status_callback("Error: Test prompt failed")
            self.restarting_lcpp = False
            self.running = False
            return "error"
            
        if baseline_slope is not None:
            threshold = 0.7 * baseline_slope
            if measured_slope < threshold:
                err_msg = f"Performance degraded: {measured_slope:.2f} t/s is below 70% of baseline ({baseline_slope:.2f} t/s)"
                self.log(err_msg, logging.ERROR)
                if self.status_callback:
                    self.status_callback(f"Error: {err_msg}")
                self.restarting_lcpp = False
                self.running = False
                return "performance_degraded"
                
        self.log("llama.cpp restart verification successful. Status: online")
        self.restarting_lcpp = False
        return "online"

    async def handle_slow_jobs_routine(self, model: str, slow_jobs_tps: float):
        self.log(f"Handling slow jobs detected for model {model}. Initiating graceful disconnect...")
        self.soft_stopping = True
        if self.status_callback:
            self.status_callback("Stopping")
            
        # Wait for current active jobs to finish
        while self.current_jobs > 0:
            await asyncio.sleep(0.5)
            
        if self.websocket:
            self.log("Closing WebSocket connection gracefully.")
            await self.websocket.close()
            self.websocket = None
            
        # Compare the slope value of the 'slow' jobs with baseline
        baseline_slope = None
        baseline_info = self.performance_baselines.get(model)
        if baseline_info and "slope" in baseline_info:
            baseline_slope = baseline_info["slope"]
            
        if baseline_slope is not None:
            threshold = 0.7 * baseline_slope
            if slow_jobs_tps >= threshold:
                self.log(f"Slow jobs performance ({slow_jobs_tps:.2f} t/s) is comparable to baseline ({baseline_slope:.2f} t/s). Blacklisting model {model}...")
                if model not in self.blacklisted_models:
                    self.blacklisted_models.append(model)
                if model in self.loaded_models:
                    self.loaded_models.remove(model)
                self.save_blacklist()
                await self.send_status(force_full=True)
                
                # Re-allow running and reconnect
                self.soft_stopping = False
            else:
                self.log(f"Slow jobs performance ({slow_jobs_tps:.2f} t/s) is worse than baseline ({baseline_slope:.2f} t/s). Hardware issue likely, initiating restart...", logging.ERROR)
                await self.restart_lcpp()
        else:
            self.log(f"No baseline data for {model} to compare slow jobs. Defaulting to blacklisting.", logging.WARNING)
            if model not in self.blacklisted_models:
                self.blacklisted_models.append(model)
            if model in self.loaded_models:
                self.loaded_models.remove(model)
            self.save_blacklist()
            await self.send_status(force_full=True)
            self.soft_stopping = False
