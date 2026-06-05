#!/bin/bash

# Edit these values to configure your provider
PROVIDER_ID="56bee686-d2e5-4a99-9a81-886f232a3378"
AUTO_MANAGE="True"
MANAGED_STORAGE_GB="30"

# Remove existing container if it exists

echo "Starting thinkfarm-provider container..."
echo "Provider ID: $PROVIDER_ID"
echo "Auto Manage: $AUTO_MANAGE"
echo "Managed Storage: $MANAGED_STORAGE_GB GB"

# Run the container
docker run -d \
  --name thinkfarm-provider \
  --gpus=all \
  --security-opt=label=disable \
  --restart always \
  -e PROVIDER_ID="$PROVIDER_ID" \
  -e OLLAMA_KEEP_ALIVE=-1 \
  -e AUTO_MANAGE="$AUTO_MANAGE" \
  -e MANAGED_STORAGE_GB="$MANAGED_STORAGE_GB" \
  -v "/tank/testthink:/root/.ollama" \
  thinkfarm:latest

echo "Container started in background."
echo "Use 'docker logs -f thinkfarm-provider' to see output."
