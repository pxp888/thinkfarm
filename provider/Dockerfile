# Use the official Ollama image as the base
FROM ollama/ollama:latest

# Install Python and dependencies
# We use python3-pip and curl (for health checks)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up app directory
WORKDIR /app

# Copy requirements and install
# We remove customtkinter as it's a GUI library and not needed for a headless container
COPY provider/requirements.txt .
RUN sed -i '/customtkinter/d' requirements.txt && \
    pip3 install --no-cache-dir -r requirements.txt --break-system-packages

# Copy the provider source
# We copy everything in provider/ to /app/provider/
# solo.py and headless.py will be at /app/provider/
COPY provider/ /app/provider/

# Copy entrypoint script and make it executable
COPY provider/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Environment variables with defaults
ENV PYTHONUNBUFFERED=1
ENV PROVIDER_ID=""
ENV MANAGED_STORAGE_GB=30
ENV AUTO_MANAGE=False
ENV SERVER_URL=https://app.thinkfarm.net

# The entrypoint script starts both Ollama and the Provider app
ENTRYPOINT ["/app/entrypoint.sh"]
