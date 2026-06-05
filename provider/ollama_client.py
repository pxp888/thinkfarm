"""
Ollama client to query local Ollama server
Provides methods to fetch model information and current state
"""

from os import getenv
from pathlib import Path
from typing import Any, Dict, List, Optional

import configparser
import json
import httpx


class OllamaClient:
    """Client for interacting with local Ollama server."""

    def __init__(self, ollama_url: Optional[str] = None):
        # Priority: passed arg > config.ini > env var > hardcoded default
        if not ollama_url:
            config_dir = Path.home() / ".thinkfarm"
            config_path = config_dir / "config.ini"
            config = configparser.ConfigParser()
            config.read(str(config_path))
            try:
                ollama_url = config.get("provider", "ollama_url").strip()
            except (configparser.NoSectionError, configparser.NoOptionError):
                ollama_url = None
        if not ollama_url:
            ollama_url = getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        self.base_url = ollama_url
        self.httpx_client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)

    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Get list of all available models from Ollama.
        Filters out:
          - Custom/derived models that cannot be pulled from the official registry.
          - Cloud models that are routed to remote servers.
        Returns list of model dicts.
        """
        try:
            response = await self.httpx_client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
        except Exception as e:
            print(f"[OllamaClient] Error fetching models from /api/tags: {e}")
            return None

        # Filter out custom and cloud models
        filtered: List[Dict[str, Any]] = []
        async with httpx.AsyncClient(base_url=self.base_url, timeout=30.0) as client:
            for model in models:
                name = model.get("name", "")
                try:
                    resp = await client.post("/api/show", json={"name": name})
                    if resp.status_code == 404:
                        continue
                    info = resp.json()

                    # Filter out cloud models (those with remote_host or remote_model)
                    if info.get("remote_host") or info.get("remote_model"):
                        print(f"[OllamaClient] Skipping cloud model: {name}")
                        continue

                    # parent_model is empty for official/base models;
                    # non-empty means it was created/fine-tuned from another model.
                    # Some official models (e.g. vision) point to internal blobs.
                    parent = (info.get("details", {}) or {}).get("parent_model", "")
                    if parent and not (parent.startswith("/") or "sha256" in parent):
                        print(f"[OllamaClient] Skipping custom model (parent: {parent}): {name}")
                        continue
                    filtered.append(model)
                except Exception as e:
                    print(f"[OllamaClient] /api/show error for {name}: {e}. Skipping.")
                    continue

        return filtered

    async def get_loaded_models(self) -> List[Dict[str, Any]]:
        """
        Get list of currently running/loaded models.
        Returns list of dicts with model info from /api/ps endpoint.
        """
        try:
            response = await self.httpx_client.get("/api/ps")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            print(f"[OllamaClient] Error fetching loaded models from /api/ps: {e}")
            return None



    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get info about a specific model via /api/show.
        Returns model details or None if not found.
        """
        try:
            response = await self.httpx_client.post("/api/show", json={"name": model_name})
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[OllamaClient] Error fetching model info for {model_name}: {e}")
            return None

    async def pull_model(self, model_name: str):
        """
        Pull a model from the Ollama library.
        Returns a generator for progress updates.
        """
        async with self.httpx_client.stream("POST", "/api/pull", json={"name": model_name}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    yield json.loads(line)

    async def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from the local Ollama server.
        """
        try:
            response = await self.httpx_client.request("DELETE", "/api/delete", json={"name": model_name})
            if response.status_code == 404:
                print(f"[OllamaClient] Model {model_name} not found on server; treating as deleted.")
                return True
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"[OllamaClient] Error deleting model {model_name}: {e}")
            return False

    async def generate(self, model: str, prompt: str, stream: bool = False) -> Any:
        """
        Generate response from a model.
        Used for streaming or non-streaming requests.
        """
        try:
            payload = {"model": model, "prompt": prompt, "stream": stream}
            endpoint = "/api/generate"
            response = await self.httpx_client.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Ollama generate error: {e}")
            return None

    async def chat(self, model: str, messages: List[Dict], stream: bool = False) -> Any:
        """
        Chat with a model.
        Used for chat-style interactions.
        """
        try:
            payload = {"model": model, "messages": messages, "stream": stream}
            endpoint = "/api/chat"
            response = await self.httpx_client.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Ollama chat error: {e}")
            return None

    async def create_embedding(self, model: str, prompt: str) -> Optional[List[float]]:
        """
        Get embedding vector for a prompt.
        Returns embedding or None if failed.
        """
        try:
            payload = {"model": model, "prompt": prompt}
            response = await self.httpx_client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("embedding")
        except Exception as e:
            print(f"Ollama embedding error: {e}")
            return None

    async def close(self):
        """Close the HTTPX client connection."""
        await self.httpx_client.aclose()
