# Task for scout

The project at /var/home/pxperrine/Desktop/2llamashare/unity is a PyQt6 + FastAPI app for managing multiple Ollama AI instances (sharding models across GPUs, auto-loading/unloading models based on demand). The main files are:

- **app_gui.py** — PyQt6 tray app, full UI with settings/model selection
- **client_server.py** — FastAPI reverse-proxy that intercepts/transforms Ollama API calls
- **config.py** — TOML config reading/writing
- **provider_client.py** — Core business loop (model loading/unloading via `load_most_desirable_model`, performance monitoring, health checks)
- **model_manager.py** — VRAM scoring, GPU specs, model manifest sync, desirability optimization
- **context_prober.py** — Hardware probing (RAM/GPU/VRAM), custom model creation (Modelfile), performance baselines
- **headless.py** / **main.py** — Entry points

The user wants to add a *llama.cpp server* compatibility version. Key changes:
1. **Model management is ELIMINATED** — no desirability optimization, no auto-load/unload of models (the llama.cpp server handles that externally)
2. Features stripped: `model_manager.py` logic, custom model creation in `context_prober.py`, load/desire optimization in `provider_client.py`
3. The proxy client and config can largely stay the same — just target a regular llama.cpp URL instead of localhost:11434

Your task: Determine which files need to be **duplicated** vs **modified**, and what the minimum viable change is. Be specific — call out each file, whether it becomes a new copy or gets modified in-place, and what parts are cut/kept/adapted for llama.cpp. Give a concrete action list.

---
**Output:**
Write your findings to exactly this path: /var/home/pxperrine/Desktop/2llamashare/unity/.pi-subagents/artifacts/outputs/c618a30c/context.md
This path is authoritative for this run.
Ignore any other output filename or output path mentioned elsewhere, including output destinations in the base agent prompt, system prompt, or task instructions.

## Acceptance Contract
Acceptance level: checked
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope

Required evidence: changed-files, tests-added, commands-run, residual-risks, no-staged-files

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```