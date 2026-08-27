# llama.cpp Server Compatibility — File-by-File Analysis & Action List

## Scope Summary

Llama.cpp mode = stripped-down variant of thinkfarm. The external llama.cpp server handles all model loading/unloading. Unity only needs to:
1. Proxy API calls (chat, generate, embed) through to a user-specified llama.cpp server URL
2. Connect to the central server via WebSocket for job discovery/acknowledgement
3. Forward jobs and return results — **no** auto-loading, no VRAM scoring, no model desirability, no custom model creation

No file duplication is required for a minimum viable change. Instead: add a `backend_type` config flag, make the existing files conditionally enable/disable features per backend type.

---

## File-by-File Breakdown

### 1. `config.py` — MODIFY IN-PLACE
**What to KEEP:** All section handling (provider/consumer), persistence (`save`/`load`), all existing fields.

**What to ADD:**
```python
# In __init__ defaults:
self.backend_type: str = "ollama"  # "ollama" or "llama_cpp"

# In load(): read BACKEND_TYPE from [provider] section, defaulting to "ollama"
backend = v(self._SECTION_PROVIDER, "BACKEND_TYPE", "ollama") or "ollama"
self.backend_type = backend

# In save(): persist the field
raw.set(self._SECTION_PROVIDER, "BACKEND_TYPE", self.backend_type)
```

**What to REMOVE:** Nothing (all existing fields may still apply).

---

### 2. `client_server.py` — MODIFY IN-PLACE (minimal changes)
**What to KEEP:** Entire file mostly stays the same. All proxy endpoints are identical between Ollama and llama.cpp APIs (the OpenAI-compatible `/v1/` endpoints and Ollama's `/api/` endpoints work with both).

**What to CHANGE:**
- The `server_url` target remains the central thinkfarm server — no change needed for the proxy forwarder since it proxies through the central server to whatever the central server routes.
- **Minor:** In `filter_models()`, the whitelist filtering works regardless of backend type. No changes needed here.

**Verdict:** Effectively unchanged. May want to rename comment "Ollama Proxy" → "LLM Proxy" for clarity, but not required.

---

### 3. `model_manager.py` — ELIMINATE (do NOT copy)
**Scope of removal:** This entire file (~640 lines) is **entirely consumed by Ollama-specific features**:
- VRAM scoring with GPU bandwidth heuristics (`get_gpu_specs`, `_calculate_model_suitability`)
- Portfolio optimization (`optimize_portfolio`)
- Manifest tracking, user model tracking
- Demand chart desirability computation

For llama.cpp mode, **zero code from this file is needed**. Any remaining reference to `from model_manager import` in other files must be guarded by the backend_type flag or removed.

**Where references exist:**
- `provider_client.py:487-489` — `from model_manager import ModelManager; manager = ModelManager(...)` used in scheduled optimization block (lines 481-489)
- `provider_client.py:1053-1063` — Same usage in `restart_ollama`'s priority-model determination

**Action:** Guard both `import` statements with `if self.config.backend_type == "ollama":`.

---

### 4. `context_prober.py` — ELIMINATE (do NOT copy)
**Scope of removal:** This entire file (~730 lines) is Ollama-specific:
- GPU VRAM probing via `/api/ps` and `size_vram` (llama.cpp doesn't expose these endpoints)
- Custom model creation (`create_custom_model`, `_sync_recreate_custom_models`) — llama.cpp server doesn't support this API
- Performance baseline measurement via repeated probe calls
- Analytical KV-cache limit computation based on VRAM

For llama.cpp mode, **zero code from this file is needed**.

**Where references exist:**
- `provider_client.py:479` — `from context_prober import load_context_limits as load_limits; self.context_limits = load_limits()` — replace with `{}` (empty) for llama.cpp
- `provider_client.py:483` — `from context_prober import load_performance_baselines`; replace with empty dict for llama.cpp
- `provider_client.py:497-501` — `load_context_limits()` in probing block; guard with backend_type

**Action:** Guard all `import context_prober` statements behind `if self.config.backend_type == "ollama":`. Set `context_limits = {}` and `performance_baselines = {}` for llama.cpp mode.

---

### 5. `provider_client.py` — MODIFY IN-PLACE (significant trimming)
This is the core file that needs a feature-gate layer. Add helper method early in the class:

```python
@property
def _is_ollama_backend(self):
    return self.config.backend_type == "ollama"
```

Then gate the following sections behind this flag:

| Section (line range) | Feature | Action |
|---|---|---|
| `__init__` (~lines 45-70) | `context_limits`, `performance_baselines`, `probing_triggered` | Guard with `_is_ollama_backend`. Set defaults to `{}` for llama.cpp. |
| `load_context_limits()` (lines ~90-92) | Load context limits from disk | Guard: return `{}` if not ollama backend |
| `load_performance_baselines()` (lines ~94-97) | Load baselines from disk | Guard: return `{}` if not ollama backend |
| `get_loaded_models()` (lines ~180-195) | `/api/ps` query | Safe for llama.cpp? **NO** — llama.cpp doesn't expose this endpoint. For llama.cpp, track loaded models differently or assume all are loadable. |
| `send_status()` (lines ~200-240) | Status messages with `context_limits`, `loaded_models` | Strip `context_limits` field for llama.cpp; `loaded_models` may be unreliable |
| `run()` main loop (~line 250) | Scheduled portfolio optimization block (lines ~475-489) | Add `if self._is_ollama_backend and self.config.auto_manage_models and ...` guard |
| `run()` main loop — probing (lines ~493-506) | Context probing on new models | Add `if self._is_ollama_backend:` guard around the probing block |
| `run()` startup model load (line ~510) | `self.startup_model_loaded` check → `load_most_desirable_model` | Guard: only call `load_most_desirable_model()` for ollama backend |
| `handle_message()` job_assigned (lines ~570-585) | `keep_model_loaded()` call | Guard with `_is_ollama_backend` — no auto-load needed for llama.cpp |
| `execute_job()` (lines ~602-790+) | Model name mapping (`thinkfarm-` prefix) | Guard mapping: only apply when ollama backend. Forward model name as-is to llama.cpp server. |
| `monitor_performance()` (lines ~800-850) | Zero-eval detection, slope monitor | Guard with `_is_ollama_backend`. For llama.cpp mode: skip monitoring, just report raw eval counts. |
| `keep_model_loaded()` (~line 870) | Keep-alive keep_alive=-1 | Not needed for llama.cpp; guard entirely |
| `run_heartbeat_check()` (~line 900) | Heartbeat loaded models to keep them alive | Guard with `_is_ollama_backend`; no-op for llama.cpp |
| `load_most_desirable_model()` (~line 950) | Demand chart → desirability score model loading | Entire method only needed for ollama; guard or make stub |
| `restart_ollama()` (lines ~1030-1140+) | Ollama restart, priority model test prompt | Guard: the restart_callback path still works (for managed mode). The priority-model determination and test-prompt are unnecessary for llama.cpp. |

**Key behavioral change:** In llama.cpp mode, `execute_job()` must forward all requests directly without:
- Stripping the `thinkfarm-` prefix or re-adding it (model names pass through unchanged)
- Loading models implicitly (the external server manages its model pool)

---

### 6. `app_gui.py` — MODIFY IN-PLACE (UI mode toggle)
**What to CHANGE:**

#### a) Add backend-type selector in Provider Settings group
Add near the auto_manage_models checkbox area:
```python
# Backend type selector
backend_layout = QHBoxLayout()
self.backend_type_cb = QComboBox()  # PyQt6.QtWidgets import QComboBox needed
self.backend_type_cb.addItems(["Ollama (managed)", "llama.cpp server"])
if self.config_manager.backend_type == "ollama":
    self.backend_type_cb.setCurrentIndex(0)
else:
    self.backend_type_cb.setCurrentIndex(1)
self.backend_type_cb.currentIndexChanged.connect(self.on_backend_changed)
backend_layout.addWidget(QLabel("Inference Backend:"))
backend_layout.addWidget(self.backend_type_cb)
```

#### b) Hide/show Ollama-specific fields based on backend type
The `toggle_managed_ollama_fields()` method already exists. Extend it to also hide:
- "Local Ollama URL" — irrelevant for external llama.cpp; replace with generic "Server URL" label
- "Model Storage Path" — only relevant for Ollama
- "Auto Manage Models" checkbox — unnecessary feature flag
- Any Ollama path that references `/api/ps`, `/api/delete`, etc.

#### c) Kill Ollama-specific methods or gate them:
- `restart_ollama_server()` (~line 75): Only runs in managed mode (current behavior). In llama.cpp mode, this method is not called from `__init__`. The `managed_ollama` flag already gates it.
- `_kill_ollama_process()`: Already gated. No change needed.

#### d) Add QComboBox import:
Update the import: add `QComboBox` to the list.

---

### 7. `main.py` — MODIFY IN-PLACE (minimal)
**What to CHANGE:** Nothing functional. May want to add a comment noting that both backends share this entry point, but no code changes needed since `backend_type` is set via config.

---

### 8. `headless.py` — MODIFY IN-PLACE (minimal)
**What to CHANGE:** The ProviderClient initialization will now receive a config with `backend_type`. No code change needed as long as the guards are in place in provider_client.py. May want a CLI flag for `--backend <ollama|llama_cpp>` that overrides the config value.

---

## What NOT to Dup (Common Mistake)

Do **not** create:
- `config_llama.py` — unnecessary duplication; just add one field
- `provider_client_llama.py` — better to gate features than duplicate the entire websocket/forwarding loop
- `context_prober_llama.py` — context probing is meaningless for llama.cpp (no VRAM spillover to probe)
- `model_manager.py` copy — entirely irrelevant

---

## Minimal Viability Checklist

| # | Action | File | Lines Affected | Risk Level |
|---|---|---|---|---|
| 1 | Add `backend_type` config field | `config.py` | __init__, load(), save() (3 locations) | Low |
| 2 | Guard all context_prober imports | `provider_client.py` init, probing blocks | ~lines 475-506 | Medium |
| 3 | Guard all model_manager imports | `provider_client.py` optimization, restart | ~lines 481-489, ~1053-1063 | Low |
| 4 | Strip thinkfarm- model mapping for llama.cpp | `provider_client.py` execute_job | ~lines 620-630 | Medium (test with real llama.cpp) |
| 5 | Add backend type selector to GUI | `app_gui.py` provider group | +~20 lines UI code | Low |
| 6 | Hide/show Ollama-specific fields in GUI | `app_gui.py` toggle_managed_ollama_fields | ~lines 448-473 (extend) | Low |
| 7 | Gate /api/ps dependent logic | provider_client.py send_status, get_loaded_models | Multiple locations | Medium — llama.cpp may not have this endpoint |

---

## Critical Unknowns / Risks

1. **Does llama.cpp's OpenAI-compatible API support all the endpoints Unity proxies?** 
   - `/v1/chat/completions`, `/v1/completions`, `/v1/models` ✅ (standard)
   - `/v1/responses` ❓ — verify llama.cpp supports this endpoint
   - `/api/embeddings` ❓ — may differ from llama.cpp's embedding API

2. **Does the external llama.cpp server expose a model list compatible with Unity's provider protocol `loaded_models`?** 
   - Unity expects Ollama-style `/api/ps` responses. For llama.cpp, you'll get `{"data": [{"id": "..."}]}` from `/v1/models`. The format mismatch needs handling in the status reporter.

3. **Streaming behavior** — llama.cpp supports SSE for OpenAI endpoints but may differ on chunk formatting. The existing streaming code should work but needs real-world testing.

4. **No VRAM info means no context window estimation** — the provider protocol uses `context_limits` to tell clients what max `num_ctx` is safe. For llama.cpp, use a fixed sensible default (e.g., 8192 or read from `/v1/models` metadata).

---

## Recommended Implementation Order

1. **config.py** — Add `backend_type` field (5-min change)
2. **provider_client.py** — Gate model_manager/context_prober imports, add `_is_ollama_backend` property (~45 min)
3. **app_gui.py** — Add combobox + toggle logic for fields (~30 min)
4. Test Ollama mode still works (regression validation)
5. Test llama.cpp mode with a real llama.cpp server

Total estimated effort: **~1.5 hours of coding + testing time**.
