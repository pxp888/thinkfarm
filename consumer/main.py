"""
Consumer FastAPI app

Runs on port 11435 and presents itself as a local Ollama server.
Forwards all requests to the central thinkfarm server, including
streaming inference responses.
"""

import configparser
import json
import os
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Imports
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------
SERVER_URL = None

def load_config():
    """Load configuration from .env and config.ini."""
    global SERVER_URL
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle
        bundle_dir = sys._MEIPASS
        env_path = os.path.join(bundle_dir, '.env')
        load_dotenv(env_path)
    else:
        # Running in normal Python
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(env_path)
    SERVER_URL = os.environ.get("SERVER_URL", "https://app.thinkfarm.net")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure config is fresh on every startup
    load_config()
    
    # Load CONSUMER_ID from config.ini in ~/.thinkfarm/
    config_dir = os.path.expanduser("~/.thinkfarm")
    os.makedirs(config_dir, exist_ok=True)
    cfg_path = os.path.join(config_dir, "config.ini")
    
    config = configparser.ConfigParser()
    config.read(cfg_path)
    
    cid = config.get("consumer", "consumer_id", fallback=None)
    if cid:
        app.state.consumer_id = cid
        print(f"[lifespan] Loaded CONSUMER_ID: {cid} from {cfg_path}")
    else:
        print(f"Error: CONSUMER_ID not found in {cfg_path}")
        # In GUI context, we don't necessarily want to exit the whole process,
        # but the lifespan failure might prevent the server from starting.
        # For now, we'll allow it but health check/inference will fail.
        app.state.consumer_id = None

    yield

# Initial load for module-level variables
load_config()



app = FastAPI(title="thinkfarm Consumer", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health(request: Request):
    """Health check endpoint."""
    return {"status": "ok", "consumer_id": request.app.state.consumer_id}


@app.get("/version")
@app.get("/api/version")
async def version():
    """Return Ollama-compatible version from the central server."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SERVER_URL}/api/version")
            response.raise_for_status()
            return response.json()
        except Exception:
            return {"version": "0.18.0"}


# ---------------------------------------------------------------------------
# Model catalog endpoints (simple passthrough)
# ---------------------------------------------------------------------------

def get_whitelist_settings():
    """Read whitelist settings from config.ini."""
    try:
        config_dir = os.path.expanduser("~/.thinkfarm")
        cfg_path = os.path.join(config_dir, "config.ini")
        if not os.path.exists(cfg_path):
            return False, []
            
        config = configparser.ConfigParser()
        config.read(cfg_path)
        
        enabled = config.getboolean("consumer", "whitelist_enabled", fallback=False)
        models_str = config.get("consumer", "whitelist_models", fallback="")
        models = [m.strip() for m in models_str.split(",") if m.strip()]
        
        return enabled, models
    except Exception:
        return False, []


@app.get("/api/tags")
async def get_tags():
    """Return list of all available models from the central server."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SERVER_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            
            enabled, whitelist = get_whitelist_settings()
            if enabled and whitelist:
                filtered_models = [
                    m for m in data.get("models", [])
                    if (m.get("name") if isinstance(m, dict) else m) in whitelist
                ]
                data["models"] = filtered_models
            
            return data
        except Exception:
            return {"models": []}


@app.get("/api/ps")
async def get_ps():
    """Return list of loaded models from the central server."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SERVER_URL}/api/ps")
            response.raise_for_status()
            data = response.json()
            
            enabled, whitelist = get_whitelist_settings()
            if enabled and whitelist:
                filtered_models = [
                    m for m in data.get("models", [])
                    if (m.get("name") if isinstance(m, dict) else m) in whitelist
                ]
                data["models"] = filtered_models
                
            return data
        except Exception:
            return {"models": []}


# ---------------------------------------------------------------------------
# Inference endpoints – streaming passthrough
# ---------------------------------------------------------------------------

async def _stream_to_server(request: Request, endpoint: str) -> StreamingResponse:
    """
    Forward an inference request to the central server and stream the
    response back to the caller.

    Adds X-Consumer-ID so the server can log which consumer made the request.
    """
    
    # Get consumer_id from FastAPI app state (loaded on startup)
    cid = request.app.state.consumer_id
    if not cid:
        raise HTTPException(status_code=500, detail="Consumer ID not loaded")
    
    body_bytes = await request.body()
    url = f"{SERVER_URL}/{endpoint if endpoint.startswith('v1') else 'api/' + endpoint}"

    client = httpx.AsyncClient(timeout=None)
    try:
        # We use a context manager for the request to ensure we can read headers first
        req = client.build_request(
            "POST",
            url,
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Consumer-ID": cid,
            }
        )
        response = await client.send(req, stream=True)
        
        if response.status_code >= 400:
            # If server returned an error, read it and raise
            error_detail = await response.aread()
            await response.aclose()
            await client.aclose()
            try:
                detail = error_detail.decode()
                # Try to parse as JSON if possible
                detail_json = json.loads(detail)
                if isinstance(detail_json, dict) and "detail" in detail_json:
                    detail = detail_json["detail"]
            except:
                detail = error_detail.decode() or f"Server returned {response.status_code}"
            
            raise HTTPException(status_code=response.status_code, detail=detail)

        async def generate():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return StreamingResponse(
            generate(),
            status_code=response.status_code,
            media_type=response.headers.get("Content-Type", "application/json")
        )
    except Exception as e:
        if not isinstance(e, HTTPException):
            await client.aclose()
            raise HTTPException(status_code=500, detail=str(e))
        raise e


@app.post("/api/chat")
async def chat(request: Request):
    """Forward a chat request to the central server (streaming)."""
    return await _stream_to_server(request, "chat")


@app.post("/api/generate")
async def generate(request: Request):
    """Forward a generate request to the central server (streaming)."""
    return await _stream_to_server(request, "generate")


@app.post("/api/embeddings")
async def embeddings(request: Request):
    """Forward an embeddings request to the central server."""
    return await _stream_to_server(request, "embeddings")


@app.post("/api/embed")
async def embed(request: Request):
    """Forward a batch embed request to the central server."""
    return await _stream_to_server(request, "embed")


@app.post("/api/show")
async def show(request: Request):
    """Forward a model info request to the central server."""
    return await _stream_to_server(request, "show")


# ---------------------------------------------------------------------------
# OpenAI Compatibility Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    """Forward an OpenAI chat request to the central server."""
    return await _stream_to_server(request, "v1/chat/completions")


@app.post("/v1/completions")
async def openai_completions(request: Request):
    """Forward an OpenAI completions request to the central server."""
    return await _stream_to_server(request, "v1/completions")


@app.post("/v1/responses")
async def openai_responses(request: Request):
    """Forward an OpenAI-style responses request to the central server."""
    return await _stream_to_server(request, "v1/responses")


@app.get("/v1/models")
async def openai_models():
    """Return list of models in OpenAI-compatible format."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SERVER_URL}/v1/models")
            response.raise_for_status()
            data = response.json()
            
            enabled, whitelist = get_whitelist_settings()
            if enabled and whitelist:
                filtered_data = [
                    m for m in data.get("data", [])
                    if m.get("id") in whitelist
                ]
                data["data"] = filtered_data
                
            return data
        except Exception:
            return {"object": "list", "data": []}
