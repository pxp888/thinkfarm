podman run -d \
  --name llama-qwen38 \
  --device=nvidia.com/gpu=all \
  --security-opt=label=disable \
  -p 8086:8080 \
  -v /home/pxperrine/Documents/code/lcpp:/models:z \
  --entrypoint /app/llama-server \
  ghcr.io/ggml-org/llama.cpp:full-cuda13 \
  -m /models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --mmproj /models/mmproj-BF16.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  --parallel 2
  
