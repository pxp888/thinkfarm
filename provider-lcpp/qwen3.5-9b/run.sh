#!/usr/bin/env bash
docker run -d \
  --name llama-qwen35 \
  --device=nvidia.com/gpu=0 \
  --security-opt=label=disable \
  -p 8086:8080 \
  -v "$(dirname "$0")":/models:z \
  --entrypoint /app/llama-server \
  ghcr.io/ggml-org/llama.cpp:full-cuda13 \
  -m /models/Qwen3.5-9B-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -fa on \
  --parallel 1 \
  --spec-type draft-mtp \
  --spec-draft-n-max 6
