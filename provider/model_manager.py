import asyncio
import json
import os
import platform
import random
import subprocess
import httpx
from typing import Dict, List, Optional, Set
from pathlib import Path
import context_prober

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path.home() / ".thinkfarm"
_MANIFEST_NAME_FILE = _CONFIG_DIR / "managed_models_names.json"
_USER_MODELS_FILE = _CONFIG_DIR / "user_models.json"
_PFX = "[MODEL-MGMT]"

# Hysteresis: existing models get a boost to prevent rapid swapping.
# A new model must be > 20% better than an existing one to trigger a swap.
_STICKINESS_FACTOR = 1.2

class ModelManager:
    def __init__(self, ollama_client, server_url: str):
        self.ollama = ollama_client
        self.server_url = server_url.rstrip("/")
        self.managed_storage_gb = 0
        self.manifest: Set[str] = set()  # managed model names
        self.user_models: Set[str] = set()
        self._load_manifest_names()
        self._load_user_models()

    # Map of GGUF tensor type names → bytes per element
    # Accurate estimates based on GGUF block overheads (bits-per-weight / 8)
    _GGUF_TYPE_BYTES: Dict[str, float] = {
        # --- Float types ---
        "F32": 4.0,
        "FF32": 4.0,
        "F16": 2.0,
        "FF16": 2.0,
        "F8": 1.0,

        # --- Q4 series (approx 4.5 - 5.0 bpw) ---
        "Q4_0": 0.5625,
        "Q4_1": 0.6250,
        "Q4_K": 0.5625,
        "Q4_K_M": 0.6062,
        "Q4_K_S": 0.5312,

        # --- Q5 series (approx 5.5 - 6.0 bpw) ---
        "Q5_0": 0.6875,
        "Q5_1": 0.7500,
        "Q5_K": 0.6875,
        "Q5_K_M": 0.7187,
        "Q5_K_S": 0.6562,

        # --- Q6 series (approx 6.6 bpw) ---
        "Q6_K": 0.8250,

        # --- Q8 series (approx 8.5 bpw) ---
        "Q8_0": 1.0625,
        "Q8_K": 1.0625,

        # --- Q2 / Q3 series ---
        "Q2_K": 0.3281,
        "Q3_K_S": 0.4141,
        "Q3_K_M": 0.4688,
        "Q3_K_L": 0.5156,

        # --- Others ---
        "I8": 1.0,
        "I16": 2.0,
        "I32": 4.0,
        "BF16": 2.0,      # Missing in previous version
        "Q4_0_expert": 0.5625,
    }



    def _load_manifest_names(self):
        """Load the set of managed model names from disk."""
        if _MANIFEST_NAME_FILE.exists():
            try:
                with open(_MANIFEST_NAME_FILE, "r") as f:
                    self.manifest = set(json.load(f))
            except Exception as e:
                print(f"{_PFX} Error loading manifest names: {e}")
                self.manifest = set()

    def _estimate_vram_for_model(self, info: dict) -> int:
        """Walk the tensor list to estimate VRAM needed for this model."""
        # Ollama often nests tensor info inside 'model_info'
        tensors = info.get("tensors")
        if not tensors and "model_info" in info:
            tensors = info["model_info"].get("tensors", [])
        
        if not tensors:
            return 0

        total_bytes = 0
        for t in tensors:
            shape = t.get("shape", [])
            elem_count = 1
            for dim in shape:
                elem_count *= dim
            
            # Case-insensitive lookup for the type
            type_name = str(t.get("type", "F16")).upper()
            bpe = self._GGUF_TYPE_BYTES.get(type_name, 0.5)
            
            total_bytes += elem_count * bpe

        return int(total_bytes * 1.05)

    def _save_manifest(self):
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(_MANIFEST_NAME_FILE, "w") as f:
                json.dump(list(self.manifest), f, indent=2)
        except Exception as e:
            print(f"{_PFX} Error saving manifest names: {e}")

    def _load_user_models(self):
        """Load the set of user-model names from disk."""
        if _USER_MODELS_FILE.exists():
            try:
                with open(_USER_MODELS_FILE, "r") as f:
                    self.user_models = set(json.load(f))
            except Exception as e:
                print(f"{_PFX} Error loading user_models: {e}")
                self.user_models = set()

    def _save_user_models(self):
        """Persist the user_models set to disk."""
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(_USER_MODELS_FILE, "w") as f:
                json.dump(list(self.user_models), f, indent=2)
        except Exception as e:
            print(f"{_PFX} Error saving user_models: {e}")

    def get_total_vram(self) -> int:
        """Get total VRAM in bytes, supporting NVIDIA and AMD on Linux and Windows.

        Detection order:
            1. NVIDIA (nvidia-smi)             — Linux / Windows
            2. AMD ROCm (rocm-smi)             — Linux only
            3. AMD kernel sysfs (amdgpu)       — Linux only
            4. AMD Software (amdsmi)           — Windows only
            5. macOS unified memory              — system_profiler

        Returns 0 if no GPU VRAM can be detected.
        Returns a very large number on macOS to signal "infinite" VRAM
        (Metal can use all available system memory).
        """
        system = platform.system().lower()  # 'linux' | 'windows' | 'darwin'

        # --- NVIDIA (cross-platform) ---
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=5
            )
            mib = int(res.stdout.strip().split("\n")[0])
            return mib * 1024 * 1024
        except Exception:
            pass

        # --- AMD on Linux ---
        if system == "linux":
            # Try rocm-smi first
            try:
                res = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram"],
                    capture_output=True, text=True, check=True, timeout=5
                )
                for line in res.stdout.splitlines():
                    if "vram_total" in line:
                        kib = int(line.split(":")[1].strip().split()[0])
                        return kib * 1024
            except Exception:
                pass

            # Try kernel sysfs (amdgpu driver, no extra tools needed)
            for sys_path in (
                "/sys/class/drm/card0/device/gpu_info/vram_total",
                "/sys/class/drm/card0/device/mem_info_vram_total",
            ):
                try:
                    val = Path(sys_path).read_text().strip()
                    if val.isdigit():
                        return int(val)
                except Exception:
                    pass

        # --- AMD on Windows (requires AMD Software: Adren Edition) ---
        if system == "windows":
            try:
                res = subprocess.run(
                    ["amdsmi", "--showvraminfo"],
                    capture_output=True, text=True, check=True, timeout=5
                )
                for line in res.stdout.splitlines():
                    if "VRAM" in line and "MiB" in line:
                        mib = float(line.split(":")[1].strip().split("MiB")[0].strip())
                        return int(mib * 1024 * 1024)
            except Exception:
                pass

        # --- macOS unified memory (Apple Silicon) ---
        # Metal can use all available system RAM for GPU inference.
        if system == "darwin":
            try:
                res = subprocess.run(
                    ["sysctl", "hw.memsize"],
                    capture_output=True, text=True, timeout=5
                )
                # e.g. "hw.memsize: 34359738368"
                val = res.stdout.strip().split(":")[1].strip()
                return int(val)
            except Exception:
                pass

        return 0

    async def get_demand_chart(self) -> List[dict]:
        """Fetch demand chart from central server."""
        headers = {"User-Agent": "Mozilla/5.0 (thinkfarm-provider)"}
        try:
            # verify=False can be used if SSL issues persist, but better to keep it True by default
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, verify=True) as client:
                url = f"{self.server_url}/api/demandchart"
                if "thinkfarm.net" in self.server_url:
                    url = "https://app.thinkfarm.net/api/demandchart"
                
                print(f"{_PFX} Fetching demand chart from: {url}")
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                return resp.json().get("data", [])
        except httpx.HTTPStatusError as e:
            print(f"{_PFX} HTTP Error fetching demand chart: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            print(f"{_PFX} Error fetching demand chart: {e}")
            return []

    async def get_remote_model_info(self, model_name: str) -> Optional[dict]:
        """Fetch model info from central server's show endpoint."""
        headers = {"User-Agent": "Mozilla/5.0 (thinkfarm-provider)"}
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, verify=True) as client:
                url = f"{self.server_url}/api/show"
                if "thinkfarm.net" in self.server_url:
                    url = "https://app.thinkfarm.net/api/show"
                
                resp = await client.post(url, json={"name": model_name}, headers=headers)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            print(f"{_PFX} HTTP Error fetching remote info for {model_name}: {e.response.status_code}")
            return None
        except Exception as e:
            print(f"{_PFX} Error fetching remote info for {model_name}: {e}")
            return None

    async def optimize_portfolio(self, storage_limit_gb: float) -> List[str]:
        """
        Runs the optimization logic.
        Returns a list of models that were newly pulled and need probing.
        """
        self.managed_storage_gb = storage_limit_gb
        if storage_limit_gb <= 0:
            return []

        print(f"{_PFX} Starting portfolio optimization (Limit: {storage_limit_gb} GB)")
        
        # 0. Get current local models to identify User Models
        local_tags = await self.ollama.get_models()
        local_model_names = set()
        if local_tags:
            local_model_names = {m.get("name") for m in local_tags if m.get("name")}
        
        # User models are those local but NOT in our manifest, plus any previously recorded
        local_user_models = local_model_names - self.manifest
        all_user_models = local_user_models | self.user_models

        # Accumulate: never remove from the persistent set
        new_user_models = local_user_models - self.user_models
        if new_user_models:
            self.user_models.update(new_user_models)
            self._save_user_models()
            print(f"{_PFX} user_models updated ({len(self.user_models)} total): {sorted(self.user_models)}")

        # 1. Hardware Discovery
        total_vram = self.get_total_vram()
        print(f"{_PFX} Detected VRAM: {total_vram / (1024**3):.2f} GB")

        # Guard: without a GPU there is nothing to optimize
        if total_vram == 0:
            print(f"{_PFX} No GPU VRAM detected. Skipping portfolio optimization.")
            return []

        # 2. Get Demand Chart
        demand = await self.get_demand_chart()
        if not demand:
            return []

        # 3. Calculate Opportunity & Filter
        candidates = []
        for item in demand:
            model = item["model"]
            if model in all_user_models:
                continue

            revenue = item["revenue"]
            providers = max(1, item["providers"])
            opportunity = revenue / providers

            # Apply stickiness for existing managed models
            if model in self.manifest:
                opportunity *= _STICKINESS_FACTOR
            
            info = await self.get_remote_model_info(model)
            if not info:
                continue

            # Estimate VRAM by walking the tensor list
            estimated_vram = self._estimate_vram_for_model(info)
            print(f"{_PFX} Estimated VRAM for {model}: {estimated_vram / (1024**3):.2f} GB")
            if total_vram > 0 and estimated_vram > total_vram:
                continue

            candidates.append({
                "model": model,
                "opportunity": opportunity,
                "size_bytes": estimated_vram,  # reuse field for VRAM estimate
                "info": info
            })

        candidates.sort(key=lambda x: x["opportunity"], reverse=True)

        # 4. Fill Quota
        target_managed_models = {}
        current_usage_bytes = 0
        limit_bytes = storage_limit_gb * (1024**3)

        for c in candidates:
            if current_usage_bytes + c["size_bytes"] <= limit_bytes:
                target_managed_models[c["model"]] = c
                current_usage_bytes += c["size_bytes"]
            else:
                break

        # Optimized Plan
        plan_lines = [
            f"{_PFX} === Optimized Plan ===",
            f"{_PFX}   Storage: {current_usage_bytes / (1024**3):.2f} / {limit_bytes / (1024**3):.2f} GB",
            f"{_PFX}   Models:",
        ]
        for name, c in target_managed_models.items():
            plan_lines.append(f"{_PFX}     - {name} ({c['opportunity']:.2f} opp, {c['size_bytes'] / (1024**3):.2f} GB)")
        # Deletions & additions
        to_remove = self.manifest - set(target_managed_models.keys()) - self.user_models
        to_add = set(target_managed_models.keys()) - self.manifest
        # Models already on disk won't need pulling (user_models already excluded them)
        for model in to_remove:
            plan_lines.append(f"{_PFX}   ❌ REMOVE: {model}")
        for model in to_add:
            plan_lines.append(f"{_PFX}   ➕ ADD: {model}")
        plan_lines.append(f"{_PFX} === End Plan ===")
        print("\n".join(plan_lines))

        # 5. Execute Changes

        # Coin flip: if any models are to be removed, 80% chance to skip the cycle
        if to_remove:
            if random.random() < 0.8:
                print(f"{_PFX} Optimization cycle requires removals, but skipped due to 80% coin flip.")
                return []

        newly_pulled = []

        for model in to_remove:
            print(f"{_PFX} Removing managed model: {model}")
            success = await self.ollama.delete_model(model)
            if success:
                self.manifest.discard(model)
                self._save_manifest()

        for model in to_add:
            # Staggering
            # delay = random.randint(0, 600)
            # print(f"{_PFX} Staggering pull of {model} by {delay} seconds...")
            # await asyncio.sleep(delay)
            
            print(f"{_PFX} Pulling managed model: {model}")
            stream_ok = False
            try:
                async for _ in self.ollama.pull_model(model):
                    pass
                stream_ok = True
            except Exception as e:
                print(f"{_PFX} Pull stream error for {model}: {e}")

            # Ground truth: check if model actually landed on disk
            models = await self.ollama.get_models()
            local_names = {m.get("name") for m in models if m.get("name")}

            if model in local_names:
                self.manifest.add(model)
                self._save_manifest()
                newly_pulled.append(model)
                print(f"{_PFX} Successfully pulled {model}")
            else:
                print(f"{_PFX} Pull finished but model not found on disk for {model}.")

        return newly_pulled

    async def get_managed_models_set(self) -> Set[str]:
        """Return the set of managed model names."""
        return self.manifest

    async def get_user_models_list(self) -> List[str]:
        """Return the names of all models identified as user-owned."""
        return sorted(self.user_models)
