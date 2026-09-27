#!/usr/bin/env bash
docker run -d \
  --name llamaspark \
  --device=nvidia.com/gpu=1 \
  --security-opt=label=disable \
  -p 8086:8080 \
  -v "$(dirname "$0")":/models:z \
  --entrypoint /app/llama-server \
  ghcr.io/ggml-org/llama.cpp:full-cuda13 \
  -m /models/Spark-X2.5-4B-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -fa on \
  --parallel 1
