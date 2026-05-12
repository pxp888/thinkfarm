#!/bin/bash
set -e

# Start Ollama in the background
echo "Starting Ollama..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
while ! curl -s http://127.0.0.1:11434/api/tags > /dev/null; do
  if ! kill -0 $OLLAMA_PID 2>/dev/null; then
    echo "Error: Ollama process died during startup."
    exit 1
  fi
  sleep 1
done
echo "Ollama is ready."

# Setup config.ini if not exists or if environment variables are provided
mkdir -p ~/.thinkfarm
CONFIG_FILE=~/.thinkfarm/config.ini

# Only create if it doesn't exist, or we could overwrite it every time if we want env vars to take precedence
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating initial config.ini..."
    cat <<EOF > "$CONFIG_FILE"
[provider]
provider_id = ${PROVIDER_ID:-container-provider}
ollama_url = http://127.0.0.1:11434
managed_storage_gb = ${MANAGED_STORAGE_GB:-30}
auto_manage = ${AUTO_MANAGE:-False}
EOF
else
    echo "Using existing config.ini at $CONFIG_FILE"
fi

# Run the provider
# We use 'exec' so signals (like SIGTERM) are passed to the python process
echo "Starting Provider..."
exec python3 -u provider/headless.py start
