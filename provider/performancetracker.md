# Performance Tracker: Design Document

## Problem Statement

Thinkfarm providers execute inference jobs on Ollama. Jobs that run too slowly produce poor UX for customers and waste provider compute time. Currently the system relies purely on a global "peak throughput" benchmark to flag under-performing providers — this misses two cases:

1. **A provider's hardware is wrong for a model** — it consistently produces slow jobs but still passes the global floor.
2. **A provider's performance degrades** — the hardware was fine yesterday, but Ollama is leaking resources, thermal throttling, or GPU fragmentation has made the same model behave worse today.

The provider needs a **per-model self-check system** that answers:  
> *"Is this model unsuitable for my hardware, or is something else broken?"* — and act accordingly.

---

## Design Overview

A `PerformanceTracker` class monitors throughput per-model, compares to baselines, and triggers diagnostics when performance drops. The diagnostic either halts that model permanently, restarts the Ollama server, or confirms recovery — depending on what went wrong.

### Key Design Principles

| Principle | Rationale |
|---|---|
| **Per-model tracking** | Different models have different compute/memory profiles; one bad model shouldn't mask another good one. |
| **3 consecutive failures** | Eliminates one-off noise (GPU migration, scheduler hiccups, short-job jitter) while catching sustained degradation. |
| **Windowed observation, discrete decisions** | A rolling window of recent slopes is for observability; the actual decision trigger is the consecutive count. |
| **Restart as a first-order fix** | Many performance degradations are Ollama-side (resource leak, stale model cache). Restart before giving up. |
| **Baseline is the primary reference** | The global "peak" is a floor check; the provider's own baseline (learned by the context prober) is the signal check. |

---

## Components

### 1. `PerformanceTracker` (shared logic)

**Location:** `baseprovider.py` (shared between `baseprovider.py` and `solo.py`)

**Public API:**

```python
class PerformanceTracker:
    def __init__(self, restart_command: str = "")
    
    async def mark_job_result(self, model: str, slope: float) -> Optional[str]
    async def is_model_halted(self, model: str) -> bool
    async def run_self_check(self, model: str) -> str  # "halt" | "restart" | "recover"
    async def get_model_state(self, model: str) -> dict  # for telemetry / logging
```

**What it stores per-model:**

```python
@dataclass
class PerformanceState:
    baseline_slope: float = 0.0
    baseline_samples: int = 0
    baseline_bad: bool = False           # model hardware mismatch
    recent_slopes: deque[float] = deque(maxlen=20)
    consecutive_bad: int = 0
    state: str = "IDLE"                 # IDLE | WARNING | DEGRADED | Halted | WAITING_RESTART
    halted: bool = False
    last_degraded_at: float = 0.0
```

**Key method: `mark_job_result()`**

Called after each completed job. Implements the core state machine:

```
Input: (model, slope)

1. Compare slope against two thresholds:
   a) Local baseline (if learned): slope >= baseline × 0.5  →  OK
   b) Published peak (from server):  slope >= peak / 3     →  OK
   If neither passes → mark as bad

2. If not bad:
   consecutive_bad = 0
   If state == "WAITING_RESTART" → verify_restart_success()
   
3. If bad:
   consecutive_bad += 1
   
   If consecutive_bad == 1:
       state = "WARNING"  (log only, no action)
   If consecutive_bad == 2:
       state = "WARNING"  (log warning)
   If consecutive_bad >= 3:
       state = "DEGRADED"
       action = run_self_check(model)
       if action == "halt":
           halted = True
           return "halted"
       elif action == "restart":
           state = "WAITING_RESTART"
           return "degraded"
```

**Key method: `run_self_check()`**

```
Diagnostic sequence:

1. If no restart_command configured → "halt" (no way to fix)

2. Re-probe the model (3-pass slope measurement)
   
3. Compare to existing baseline:
   
   Baseline exists?
     ├─ Fresh slope >= baseline × 0.8 → baseline OK, restart
     └─ Fresh slope <  baseline × 0.8 → baseline bad → "halt"
                                          (model wrong for hardware)
   
   No baseline?
     → Record this as baseline, return "recover"

4. Restart Ollama (via restart_command)
   
5. Wait for Ollama to come back (poll /api/tags)
   
6. Re-measure slope (1-pass at known-good num_ctx)
   
7. Compare to baseline:
   
   slope >= baseline × 0.8 → restart succeeded
     consecutive_bad = 0
     state = "IDLE"
     return "restart"
   
   slope <  baseline × 0.8 → restart didn't help
     "halt" (model is bad for this hardware)
```

**Key method: `is_model_halted()`**

Called in `handle_job_published()`:

```python
if await self.tracker.is_model_halted(model):
    print(f"[PERF] Model {model} halted — skipping job")
    return  # decline to accept this job
```

**Key method: `get_model_state()`**

Returns the full state dict for model `model` — used for telemetry to the server and for logging.

---

### 2. Context Prober Integration

**Location:** `context_prober.py`

The context prober already measures per-model GPU capacity. It now also measures per-model **slope** (throughput) and stores baselines alongside context limits.

**Probe sequence for each eligible generative model:**

```
1. Binary search → find max GPU-safe num_ctx   (existing behavior)
2. At that num_ctx → run 3 passes + measure slope
3. Save both results together:

{
  "gemma4:31b": {
    "context_limit": 131072,
    "performance": {
      "slope": 128.4,
      "samples": 3,
      "last_probed": "2026-06-06T..."
    }
  }
}
```

**Stored in:** `~/.thinkfarm/performance_baselines.json` (separate file from `gpu_context_limits.json`)

**Slope calculation:**

```
slope = (eval_count + 0.003 × prompt_eval_count) / compute_seconds

where compute_seconds = (total_duration_ns - load_duration_ns) / 1e9
     → or total_duration_ns / 1e9 if load_duration unavailable
```

**Why subtract `load_duration`?** The model load is a one-time cost that happens once per job but represents memory bandwidth, not compute throughput. Including it inflates the denominator and deflates slope, creating a systematic error for small eval counts.

---

### 3. Provider Integration

**Location:** `baseprovider.py` and `solo.py`

```python
# Initialization
tracker = PerformanceTracker(
    restart_command=config.get("provider", "restart_command", fallback="")
)

# In execute_job():
slope = await _compute_slope_from_job_result(data)
result = await tracker.mark_job_result(
    model=model,
    slope=slope,
)
if result == "halted":
    print(f"[PERF] Model {model} halted — will not accept future jobs")

# In handle_job_published():
if await tracker.is_model_halted(model):
    # Decline the job
    if websocket and data.get("job_id"):
        await websocket.send(json.dumps({
            "type": "decline",
            "job_id": data["job_id"],
            "reason": "model_performance_halted"
        }))
    return
```

**Server reporting on halt:**

```python
await websocket.send(json.dumps({
    "type": "performance_alert",
    "model": model,
    "action": "halted",
    "slope": tracker.get_model_state(model)["last_slope"],
    "baseline": tracker.get_model_state(model)["baseline_slope"],
    "consecutive_bad": tracker.get_model_state(model)["consecutive_bad"],
}))
```

---

## Thresholds Reference

| Check | Threshold | Purpose |
|---|---|---|
| **Server-side floor** | slope ≥ peak / 3 | Is the provider viable at all? |
| **Self-check alarm** | slope < baseline × 0.5 | Something is seriously wrong |
| **Self-check alarm (no baseline)** | slope < peak / 3 | No baseline yet, use global floor |
| **Restart threshold** | slope ≥ baseline × 0.8 after restart | Restart fixed the problem? |
| **Slope floor** | computed_slope > 0 for eval_count ≥ 10 | Discard trivial/noisy jobs |

---

## State Machine Diagram

```
                    ┌─────────┐
                    │  IDLE   │  ← start state for each model
                    └────┬────┘
                         │ 1st bad slope
                    ┌────▼────┐
                    │ WARNING │  ← logged, no action
                    └────┬────┘
                         │ 2nd consecutive bad
                    ┌────▼─────┐
                    │ WARNING │  ← logged as warning
                    └────┬─────┘
                         │ 3rd consecutive bad
                    ┌──────▼───────┐
                    │   DEGRADED   │
                    └──────┬───────┘
                           │ run_self_check()
                           │
              ┌────────────┼────────────┐
              │            │            │
         no restart   baseline       baseline bad
              cmd?     still good    (model wrong)
              │         │
         halt      restart       halt permanently
              │         │           │
         ┌────▼────┐ ┌─▼────────┐ ┌──────────────┐
         │ STOPPED │ │WAITING_  │ │ HALTED       │
         │ (model) │ │ RESTART │ │ (disabled in │
         └─────────┘ └──────────┘ │  prober)    │
                                   └──────────────┘
                                        │
                                   Re-check slope
                                   slope OK?
                                  /          \
                               yes             no
                                │              │
                          IDLE  │           halt
                               │
                          (consecutive_bad = 0)
                               │
                               ▼
                          ┌─────────┐
                          │  IDLE   │
                          └─────────┘
```

---

## Configuration

**config.ini (`[provider]` section):**

```ini
restart_command = docker restart ollama
```

| Value | Meaning |
|---|---|
| `"docker restart ollama"` | Restart via Docker |
| `"systemctl restart ollama"` | Restart via systemd |
| `"kill -TERM $(pgrep -f \"ollama serve\") && ollama serve &"` | Restart as subprocess |
| `""` (empty) | No restart capability — provider will halt the model but not attempt recovery |
| *(missing)* | Same as empty |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                        Provider Startup                  │
│  ┌───────────┐   ┌───────────────┐   ┌───────────────┐ │
│  │ context_  │──▶│ 3-pass slope  │──▶│ Save to       │ │
│  │ prober    │   │ measure +     │   │ performance_  │ │
│  │ runs +    │   │ save to       │   │ baselines.json│ │
│  │ GPU probe │   │ gpu_context_  │   │               │ │
│  └───────────┘   │ limits.json   │   └──────┬────────┘ │
└─────────────────┴───────────────┘          │
                    ┌───────────────┐          │
                    │  Performance  │          │
                    │  Tracker      │          │
                    │  initialized  │          │
                    └──────┬────────┘          │
                           │                   │
                 ┌─────────▼──────────┐  ┌────▼───────────┐
                 │ handle_job_published│  │ execute_job    │
                 │ is_model_halted()? │  │ compute slope  │
                 └─────┬──────────────┘  │ mark_job_result│
                       │                └──────┬──────────┘
                       │ (decline if halted)   │
                       │                  ┌────▼────────────┐
                       │                  │ Is degraded?    │
                       │                  │ run_self_check? │
                       └──────────────────┴────────────────┘
```

---

## Persistence

Performance baselines are stored in `~/.thinkfarm/performance_baselines.json` alongside existing config:

```json
{
  "baselines": {
    "gemma4:31b": {
      "slope": 128.4,
      "samples": 3,
      "last_probed": "2026-06-06T12:00:00.000000"
    },
    "qwen3.6:35b-a3b-q4_K_M": {
      "slope": 97.2,
      "samples": 3,
      "last_probed": "2026-06-06T12:01:00.000000"
    }
  },
  "performance_alerts": {
    "qwen3.6:35b-a3b-q4_K_M": {
      "last_alert": "2026-06-06T14:30:00.000000",
      "last_state": "degraded",
      "consecutive_bad": 3,
      "current_slope": 38.7
    }
  }
}
```

**What persists across restart:** baselines, alert history.  
**What does NOT persist:** `recent_slopes` and `consecutive_bad` (reset on restart, which is correct — the tracker is the live monitor, the file is the learned knowledge).

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Restart command fails | Log error, keep halted, do NOT retry automatically (could be a real Docker crash) |
| Ollama doesn't come back after restart | Stay in DEGRADED; next job will re-trigger self-check |
| No restart command configured | Halt immediately, no recovery |
| Baseline not yet learned | Compare to global published peak only; don't halt on first bad job |
| Very short jobs (eval < 10) | Ignored — slope is meaningless |
| Jobs with `load_duration` unavailable | Use `total_duration` (acceptable approximation) |

---

## Open Questions

1. **How does the provider get `published_peak` values?**  
   Currently the provider has no mechanism to fetch them. Options: (a) include in server status API, (b) provider polls server's performance endpoint, (c) accept it from the peer when `announce` is received. This needs clarification before implementation.

2. **Should `performance_baselines.json` also persist `baseline_bad`?**  
   Yes — if model+hardware is a known mismatch, we don't want to re-learn the wrong thing. The prober's "model is wrong for hardware" result should be persisted too. This suggests merging the two files or adding a shared schema version field.

3. **How to handle thermal throttling vs. permanent degradation?**  
   Thermal throttling is time-dependent — it recovers after cool-down. A single restart won't always fix it. The current design treats both the same (halt), which is conservative. A future enhancement could track `current_slope` over time with exponential decay to auto-resume once performance recovers.

---

## Implementation Checklist

- [ ] Add `PerformanceTracker` to `baseprovider.py` with full state machine
- [ ] Add slope measurement to `context_prober.py` (3-pass probe at GPU-safe context)
- [ ] Add `performance_baselines.json` persistence (independent of `gpu_context_limits.json` or merged)
- [ ] Add `restart_command` to config parser
- [ ] Hook tracker into `execute_job()` (slope compute + mark)
- [ ] Hook tracker into `handle_job_published()` (halt check)
- [ ] Add server telemetry message on halt/restart actions
- [ ] Add `is_model_halted()` check in all accept paths
- [ ] Wire `restart_command` to `subprocess` execution in tracker
- [ ] Add `_wait_for_ollama()` utility for post-restart polling
