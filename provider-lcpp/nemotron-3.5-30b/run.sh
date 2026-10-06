#!/usr/bin/env bash
docker run -d \
  --name llamaspark \
  --device=nvidia.com/gpu=1 \
  --security-opt=label=disable \
  -p 8086:8080 \
  -v "$(dirname "$0")":/models:z \
  --entrypoint /app/llama-server \
  ghcr.io/ggml-org/llama.cpp:full-cuda13 \
  -m /models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  --parallel 1
