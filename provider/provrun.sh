#!/bin/bash

# Edit these values to configure your provider
PROVIDER_ID="618e5f4f-988d-44ca-80b4-3ccd6ad541b7"
AUTO_MANAGE="True"
MANAGED_STORAGE_GB="30"

# Remove existing container if it exists
podman rm -f thinkfarm-provider 2>/dev/null

echo "Starting thinkfarm-provider container..."
echo "Provider ID: $PROVIDER_ID"
echo "Auto Manage: $AUTO_MANAGE"
echo "Managed Storage: $MANAGED_STORAGE_GB GB"

# Run the container
podman run -d \
  --name thinkfarm-provider \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --restart always \
  -e PROVIDER_ID="$PROVIDER_ID" \
  -e OLLAMA_KEEP_ALIVE=-1 \
  -e AUTO_MANAGE="$AUTO_MANAGE" \
  -e MANAGED_STORAGE_GB="$MANAGED_STORAGE_GB" \
  -v "$HOME/ollama:/root/.ollama:Z" \
  -v "$HOME/.thinkfarm:/root/.thinkfarm:Z" \
  thinkfarm-provider

echo "Container started in background."
echo "Use 'podman logs -f thinkfarm-provider' to see output."
