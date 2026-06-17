import asyncio
import configparser
import json
import os
import platform
import random
import re
import shutil
import subprocess
import httpx
from typing import Dict, List, Optional, Set
from pathlib import Path
import context_prober

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path.home() / ".thinkfarm"
_CONFIG_PATH = _CONFIG_DIR / "config.ini"
_SLOPEMON_DATA_PATH = _CONFIG_DIR / "_slopemon.json"
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

    def _get_gpu_bandwidth_from_name(self, gpu_name: str) -> Optional[float]:
        """Look up GPU memory bandwidth in GB/s based on the device name."""
        name_lower = gpu_name.lower()
        
        # Mapping table of common GPUs to their memory bandwidth in GB/s
        mappings = {
            # RTX 50 series
            "5090": 1700.0,
            "5080": 1000.0,
            
            # RTX 40 series
            "4090": 1008.0,
            "4080": 716.8,
            "4070 ti": 504.0,
            "4070": 504.0,
            "4060 ti": 288.0,
            "4060": 272.0,

            # RTX 30 series
            "3090 ti": 1008.0,
            "3090": 936.0,
            "3080 ti": 912.0,
            "3080": 760.0,
            "3070 ti": 608.0,
            "3070": 448.0,
            "3060 ti": 448.0,
            "3060": 360.0,
            
            # Enterprise / Data Center CUDA
            "a100": 1555.0,
            "a800": 1555.0,
            "h100": 3350.0,
            "h800": 3350.0,
            "a10g": 600.0,
            "a30": 933.0,
            "a40": 696.0,
            "t4": 320.0,
            "v100": 900.0,
            "p100": 732.0,

            # AMD ROCm / Radeon
            "mi300": 5300.0,
            "mi250": 3200.0,
            "mi210": 1600.0,
            "7900 xtx": 960.0,
            "7900 xt": 800.0,
            "7800 xt": 624.0,
            "7700 xt": 432.0,
            "6900 xt": 512.0,
            "6800 xt": 512.0,
            
            # Apple Silicon Unified Memory Bandwidths
            "m1 ultra": 800.0,
            "m2 ultra": 800.0,
            "m3 ultra": 800.0,
            "m1 max": 400.0,
            "m2 max": 400.0,
            "m3 max": 300.0,
            "m1 pro": 200.0,
            "m2 pro": 200.0,
            "m3 pro": 150.0,
            "m1": 68.25,
            "m2": 100.0,
            "m3": 100.0,
            "m4": 150.0,
        }

        for pattern, bw in mappings.items():
            if pattern in name_lower:
                return bw

        return None

    def get_gpu_specs(self) -> tuple[int, float]:
        """Detect total VRAM in bytes and estimated memory bandwidth in GB/s."""
        total_vram = self.get_total_vram()
        if total_vram == 0:
            return 0, 0.0

        system = platform.system().lower()
        gpu_name = ""

        # 1. Query NVIDIA GPU Name
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=5
            )
            gpu_name = res.stdout.strip().split("\n")[0]
        except Exception:
            pass

        # 2. Query AMD on Linux
        if not gpu_name and system == "linux":
            try:
                res = subprocess.run(
                    ["rocm-smi", "--showproductname"],
                    capture_output=True, text=True, timeout=5
                )
                for line in res.stdout.splitlines():
                    if "Card series" in line or "Product Name" in line:
                        gpu_name = line.split(":")[1].strip()
                        break
            except Exception:
                pass

        # 3. macOS (Apple Silicon) Model Name
        if not gpu_name and system == "darwin":
            try:
                res = subprocess.run(
                    ["sysctl", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5
                )
                gpu_name = res.stdout.strip().split(":")[1].strip()
            except Exception:
                pass

        # Parse bandwidth from name
        bandwidth = None
        if gpu_name:
            bandwidth = self._get_gpu_bandwidth_from_name(gpu_name)
            if bandwidth:
                print(f"{_PFX} Detected GPU Name: '{gpu_name}' -> Bandwidth: {bandwidth} GB/s")
            else:
                print(f"{_PFX} Detected GPU Name: '{gpu_name}' (No mapped bandwidth)")

        # Fallback heuristic based on VRAM size
        if bandwidth is None:
            vram_gb = total_vram / (1024**3)
            if vram_gb >= 24:
                bandwidth = 1000.0
            elif vram_gb >= 16:
                bandwidth = 500.0
            elif vram_gb >= 8:
                bandwidth = 300.0
            else:
                bandwidth = 150.0
            print(f"{_PFX} Fallback GPU Bandwidth heuristic: {bandwidth} GB/s (based on {vram_gb:.1f} GB VRAM)")

        return total_vram, bandwidth

    def _calculate_model_suitability(self, info: dict, context_len: int, gpu_vram_bytes: int, gpu_bandwidth_gb_s: float) -> dict:
        """
        Mathematically calculate if a model fits in VRAM and estimate its TPS.
        Based on GGUF metadata returned from central server / Ollama show endpoint.
        """
        model_info = info.get("model_info", {})
        if not model_info and "model_info" in info:
            model_info = info["model_info"].get("model_info", {})

        # 1. Parameter count
        param_count = model_info.get("general.parameter_count")
        if not param_count:
            param_count = next((v for k, v in model_info.items() if k.endswith("parameter_count")), None)
        if not param_count:
            size_bytes = float(info.get("size", 0))
            if size_bytes > 0:
                param_count = int(size_bytes * 1.5)
            else:
                param_count = 8000000000  # Default 8B

        # 2. Extract block/layer count
        block_count = model_info.get("llama.block_count") or model_info.get("general.block_count")
        if not block_count:
            block_count = next((v for k, v in model_info.items() if k.endswith(".block_count")), 32)
        
        # 3. Extract attention heads & dimension
        head_count = model_info.get("llama.attention.head_count")
        if not head_count:
            head_count = next((v for k, v in model_info.items() if k.endswith(".head_count")), 32)
        
        head_count_kv = model_info.get("llama.attention.head_count_kv")
        if not head_count_kv:
            head_count_kv = next((v for k, v in model_info.items() if k.endswith(".head_count_kv")), head_count)

        head_dim = model_info.get("llama.attention.key_length")
        if not head_dim:
            head_dim = next((v for k, v in model_info.items() if k.endswith(".key_length")), 128)

        # 4. Weights Size Calculation
        details = info.get("details", {})
        quant_type = str(details.get("quantization_level", details.get("parameter_size", "Q4_K_M"))).upper()
        
        bpe = self._GGUF_TYPE_BYTES.get(quant_type, 0.5)
        weight_bytes = param_count * bpe

        # 5. KV Cache Size Calculation (FP16 elements, 2 bytes each)
        # Formula: 2 (K and V) * num_layers * num_kv_heads * head_dim * context_len * 2 bytes
        kv_cache_bytes = 4 * int(block_count) * int(head_count_kv) * int(head_dim) * context_len

        # 6. Total VRAM footprint (Weights + KV Cache + 10% system overhead)
        total_required_bytes = int((weight_bytes + kv_cache_bytes) * 1.10)

        # 7. Check if fits in GPU VRAM
        fits_in_vram = total_required_bytes <= gpu_vram_bytes

        # 8. Estimate TPS using roofline decode phase calculation
        estimated_tps = 0.0
        if fits_in_vram and total_required_bytes > 0:
            gpu_bandwidth_bps = gpu_bandwidth_gb_s * (10**9)
            estimated_tps = 0.75 * (gpu_bandwidth_bps / total_required_bytes)

        return {
            "fits": fits_in_vram,
            "weight_bytes": weight_bytes,
            "kv_cache_bytes": kv_cache_bytes,
            "total_vram_bytes": total_required_bytes,
            "estimated_tps": round(estimated_tps, 2)
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

    async def optimize_portfolio(self, storage_limit_gb: float) -> tuple[List[str], str]:
        """
        Runs the optimization logic.
        Returns a list of models that were newly pulled and need probing.
        """
        self.managed_storage_gb = storage_limit_gb
        if storage_limit_gb <= 0:
            return ([], "")

        print(f"{_PFX} Starting portfolio optimization (Limit: {storage_limit_gb} GB)")
        
        # 0. Get current local models to identify User Models
        gpu_limits = context_prober.load_context_limits()
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
        total_vram, bandwidth = self.get_gpu_specs()

        # Guard: without a GPU there is nothing to optimize
        if total_vram == 0:
            print(f"{_PFX} No GPU VRAM detected. Skipping portfolio optimization.")
            return ([], "")

        # 2. Get Demand Chart
        demand = await self.get_demand_chart()
        if not demand:
            return ([], "")

        # Load slopemon thresholds (models with significant samples)
        slope_peer_thresholds = {}
        if _SLOPEMON_DATA_PATH.exists():
            try:
                with open(_SLOPEMON_DATA_PATH, "r") as f:
                    slopemon_data = json.load(f)
                for model_name, slopemon_info in slopemon_data.get("models", {}).items():
                    peak = slopemon_info.get("peak", 0)
                    if peak > 0:
                        slope_peer_thresholds[model_name] = peak / 3
                print(f"{_PFX} Loaded {len(slope_peer_thresholds)} network peak thresholds from slopemon.")
            except Exception as e:
                print(f"{_PFX} Error reading slopemon file: {e}")

        # Load min_acceptable_tps from config
        min_acceptable_tps = 15.0
        if _CONFIG_PATH.exists():
            try:
                config = configparser.ConfigParser()
                config.read(_CONFIG_PATH)
                if config.has_section("provider") and config.has_option("provider", "min_acceptable_tps"):
                    min_acceptable_tps = float(config.get("provider", "min_acceptable_tps"))
            except Exception as e:
                print(f"{_PFX} Error reading min_acceptable_tps from config: {e}")
        print(f"{_PFX} Default min_acceptable_tps: {min_acceptable_tps} TPS")

        # 3. Calculate Opportunity & Filter
        candidates = []
        for item in demand:
            model = item["model"]
            if model in all_user_models:
                continue

            # Skip models that failed GPU probing / cannot run on the GPU
            if gpu_limits.get(model) == 0:
                continue

            revenue = item["revenue"]
            providers = max(1, item["providers"])
            opportunity = revenue / providers

            # Apply stickiness for existing managed models
            if model in self.manifest:
                opportunity *= _STICKINESS_FACTOR
            
            # Fetch remote info
            info = await self.get_remote_model_info(model)
            if not info:
                continue

            # Check feasibility and speed via custom metadata suitability check
            try:
                suitability = self._calculate_model_suitability(
                    info,
                    context_len=8192,
                    gpu_vram_bytes=total_vram,
                    gpu_bandwidth_gb_s=bandwidth
                )
                fits_in_vram = suitability["fits"]
                estimated_tps = suitability["estimated_tps"]
                estimated_vram = suitability["total_vram_bytes"]
                
                # If calculations result in 0 or extremely low VRAM, fall back
                if estimated_vram < 100 * 1024 * 1024:
                    raise ValueError("Estimated VRAM is too low, metadata might be missing.")
                
                print(f"{_PFX} Suitability check for {model}: fits={fits_in_vram}, tps={estimated_tps:.1f} (target threshold: {slope_peer_thresholds.get(model, min_acceptable_tps):.1f} TPS, VRAM={estimated_vram / (1024**3):.2f} GB)")
            except Exception as e:
                # Fallback to existing logic if calculator fails or metadata is missing
                estimated_vram = self._estimate_vram_for_model(info)
                fits_in_vram = (total_vram > 0 and estimated_vram <= total_vram)
                # Compute a rough TPS fallback
                if fits_in_vram and estimated_vram > 0:
                    gpu_bandwidth_bps = bandwidth * (10**9)
                    estimated_tps = 0.75 * (gpu_bandwidth_bps / estimated_vram)
                else:
                    estimated_tps = 0.0
                print(f"{_PFX} [Fallback] Calculated suitability failed ({e}). Estimated VRAM for {model}: {estimated_vram / (1024**3):.2f} GB, fits={fits_in_vram}, tps={estimated_tps:.1f}")

            # Filter candidates based on VRAM fit and performance threshold
            target_threshold = slope_peer_thresholds.get(model, min_acceptable_tps)
            
            if not fits_in_vram:
                print(f"{_PFX} Skipping {model}: does not fit in VRAM.")
                continue
            
            if estimated_tps < target_threshold:
                print(f"{_PFX} Skipping {model}: estimated tps ({estimated_tps:.1f}) < target threshold ({target_threshold:.1f} TPS).")
                continue

            candidates.append({
                "model": model,
                "opportunity": opportunity,
                "size_bytes": estimated_vram,  # reuse field for VRAM estimate
                "info": info or {}
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

        # 5. Handle Displacement & Fill leftover quota with existing models
        manifest_sizes = {}
        for model in self.manifest:
            info = await self.ollama.get_model_info(model)
            manifest_sizes[model] = self._estimate_vram_for_model(info) if info else 0

        to_add = set(target_managed_models.keys()) - self.manifest
        candidates_for_removal = list(self.manifest - set(target_managed_models.keys()) - self.user_models)
        
        # Sort candidates for removal: prefer to keep those with higher opportunity
        demand_opportunity = {item["model"]: item["revenue"] / max(1, item["providers"]) for item in demand}
        for m in demand_opportunity:
            if m in self.manifest:
                demand_opportunity[m] *= _STICKINESS_FACTOR

        candidates_for_removal.sort(key=lambda m: demand_opportunity.get(m, 0), reverse=True)

        to_remove = set()
        if to_add:
            leftover_bytes = limit_bytes - current_usage_bytes
            for model in candidates_for_removal:
                model_size = manifest_sizes.get(model, 0)
                if model_size <= leftover_bytes:
                    leftover_bytes -= model_size
                    info = await self.ollama.get_model_info(model)
                    target_managed_models[model] = {
                        "model": model,
                        "opportunity": demand_opportunity.get(model, 0),
                        "size_bytes": model_size,
                        "info": info or {}
                    }
                    current_usage_bytes += model_size
                else:
                    to_remove.add(model)
        else:
            # Nothing is being added, keep all current manifest models to avoid useless removals
            for model in candidates_for_removal:
                model_size = manifest_sizes.get(model, 0)
                info = await self.ollama.get_model_info(model)
                target_managed_models[model] = {
                    "model": model,
                    "opportunity": demand_opportunity.get(model, 0),
                    "size_bytes": model_size,
                    "info": info or {}
                }
                current_usage_bytes += model_size

        # Optimized Plan
        plan_lines = [
            f"{_PFX} === Optimized Plan ===",
            f"{_PFX}   Storage: {current_usage_bytes / (1024**3):.2f} / {limit_bytes / (1024**3):.2f} GB",
            f"{_PFX}   Models:",
        ]
        for name, c in target_managed_models.items():
            plan_lines.append(f"{_PFX}     - {name} ({c['opportunity']:.2f} opp, {c['size_bytes'] / (1024**3):.2f} GB)")
        # Deletions & additions
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
                return ([], "")

        # Determine priority model: highest opportunity among all models available to the provider
        # (both user models and target managed models, and must have a positive value in gpu_context_limits)
        available_priorities = []
        for item in demand:
            model = item["model"]
            if model in all_user_models or model in target_managed_models:
                if gpu_limits.get(model, 0) <= 0:
                    continue
                revenue = item["revenue"]
                providers = max(1, item["providers"])
                opportunity = revenue / providers
                if model in self.manifest:
                    opportunity *= _STICKINESS_FACTOR
                available_priorities.append((model, opportunity))

        available_priorities.sort(key=lambda x: x[1], reverse=True)
        priority_model = available_priorities[0][0] if available_priorities else ""

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
                async for chunk in self.ollama.pull_model(model):
                    if isinstance(chunk, dict) and "error" in chunk:
                        raise Exception(chunk["error"])
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

        return (newly_pulled, priority_model)

    async def get_managed_models_set(self) -> Set[str]:
        """Return the set of managed model names."""
        return self.manifest

    async def get_user_models_list(self) -> List[str]:
        """Return the names of all models identified as user-owned."""
        return sorted(self.user_models)
