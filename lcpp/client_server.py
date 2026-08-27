import json
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from config import ConfigManager

logger = logging.getLogger("thinkfarm.client")

def create_client_app(config: ConfigManager):
    app = FastAPI(title="thinkfarm Client Proxy")
    app.state.config = config

    # Add CORS middleware to handle preflight OPTIONS requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    # Initialize shared HTTPX client
    @app.on_event("startup")
    async def startup_event():
        app.state.client = httpx.AsyncClient(timeout=60.0)
        logger.info(f"Client Server started on port {config.port} forwarding to {config.server_url}")

    @app.on_event("shutdown")
    async def shutdown_event():
        await app.state.client.aclose()
        logger.info("Client Server stopped")

    def filter_models(data: dict, endpoint: str) -> dict:
        if not config.whitelist_enabled:
            return data
        
        whitelist = set(config.whitelist_models)
        if endpoint in ("/api/tags", "/api/ps"):
            if "models" in data and isinstance(data["models"], list):
                data["models"] = [m for m in data["models"] if m.get("name") in whitelist]
        elif endpoint == "/v1/models":
            if "data" in data and isinstance(data["data"], list):
                data["data"] = [m for m in data["data"] if m.get("id") in whitelist]
        return data

    def inject_num_ctx(body: dict) -> dict:
        options = body.setdefault("options", {})
        num_ctx = options.get("num_ctx")
        if num_ctx is None or num_ctx <= 0:
            text_length = 0
            if "prompt" in body and isinstance(body["prompt"], str):
                text_length += len(body["prompt"])
            if "messages" in body and isinstance(body["messages"], list):
                for msg in body["messages"]:
                    if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], str):
                        text_length += len(msg["content"])
            
            estimated_prompt_tokens = text_length // 3
            buffer = 2048
            if "num_predict" in options and isinstance(options["num_predict"], int) and options["num_predict"] > 0:
                buffer = options["num_predict"]
            elif "max_tokens" in body and isinstance(body["max_tokens"], int) and body["max_tokens"] > 0:
                buffer = body["max_tokens"]
                
            calculated = estimated_prompt_tokens + buffer
            final_num_ctx = 4096 if calculated < 4096 else ((calculated + 1023) // 1024) * 1024
            options["num_ctx"] = final_num_ctx
            logger.debug(f"Injected num_ctx={final_num_ctx} (estimated prompt tokens: {estimated_prompt_tokens}, buffer: {buffer})")
        return body

    async def proxy_request(method: str, path: str, request: Request, transform_body: bool = False):
        url = f"{config.server_url.rstrip('/')}{path}"
        cid = config.consumer_id
        
        # Read request body if present
        body = None
        body_bytes = b""
        if method == "POST":
            try:
                body = await request.json()
                if transform_body:
                    body = inject_num_ctx(body)
                body_bytes = json.dumps(body).encode("utf-8")
            except Exception as e:
                logger.error(f"Failed to parse request JSON: {e}")
                raise HTTPException(status_code=400, detail="Invalid JSON body")
        
        logger.info(f"Proxying {method} {path} to central server...")
        
        # Determine if streaming response is requested
        stream_requested = False
        if body and isinstance(body, dict):
            stream_requested = body.get("stream", False)
            
        try:
            if stream_requested:
                req = app.state.client.build_request(
                    method,
                    url,
                    content=body_bytes,
                    headers={
                        "Content-Type": "application/json",
                        "X-Consumer-ID": cid,
                    }
                )
                resp = await app.state.client.send(req, stream=True)
                if resp.status_code >= 400:
                    await resp.aread()
                    try:
                        err_detail = resp.json()
                    except Exception:
                        err_detail = {"detail": resp.text}
                    logger.error(f"Central server returned error {resp.status_code}: {err_detail}")
                    return JSONResponse(status_code=resp.status_code, content=err_detail)
                
                # Streaming response back to client
                async def stream_generator():
                    try:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                    finally:
                        await resp.aclose()
                        
                media_type = "text/event-stream" if "v1/" in path else "application/x-ndjson"
                return StreamingResponse(stream_generator(), status_code=resp.status_code, media_type=media_type)
            else:
                req = app.state.client.build_request(
                    method,
                    url,
                    content=body_bytes,
                    headers={
                        "Content-Type": "application/json",
                        "X-Consumer-ID": cid,
                    }
                )
                resp = await app.state.client.send(req)
                if resp.status_code >= 400:
                    try:
                        err_detail = resp.json()
                    except Exception:
                        err_detail = {"detail": resp.text}
                    logger.error(f"Central server returned error {resp.status_code}: {err_detail}")
                    return JSONResponse(status_code=resp.status_code, content=err_detail)
                
                response_data = resp.json()
                response_data = filter_models(response_data, path)
                return JSONResponse(status_code=resp.status_code, content=response_data)
        except Exception as e:
            logger.error(f"Proxy connection failed: {e}")
            raise HTTPException(status_code=502, detail="Failed to connect to central thinkfarm server")

    # GET Endpoints
    @app.get("/")
    async def read_root():
        return {"status": "ok", "consumer_id": config.consumer_id}

    @app.get("/version")
    @app.get("/api/version")
    async def get_version():
        return {"version": "0.1.32"}

    @app.get("/api/tags")
    async def get_tags(request: Request):
        return await proxy_request("GET", "/api/tags", request)

    @app.get("/api/ps")
    async def get_ps(request: Request):
        return await proxy_request("GET", "/api/ps", request)

    @app.get("/v1/models")
    async def get_v1_models(request: Request):
        return await proxy_request("GET", "/v1/models", request)

    # POST Endpoints
    @app.post("/api/chat")
    async def post_chat(request: Request):
        return await proxy_request("POST", "/api/chat", request, transform_body=True)

    @app.post("/api/generate")
    async def post_generate(request: Request):
        return await proxy_request("POST", "/api/generate", request, transform_body=True)

    @app.post("/api/embeddings")
    async def post_embeddings(request: Request):
        return await proxy_request("POST", "/api/embeddings", request, transform_body=True)

    @app.post("/api/embed")
    async def post_embed(request: Request):
        return await proxy_request("POST", "/api/embed", request, transform_body=True)

    @app.post("/api/show")
    async def post_show(request: Request):
        # /api/show does NOT transform body according to Section 8
        return await proxy_request("POST", "/api/show", request, transform_body=False)

    @app.post("/v1/chat/completions")
    async def post_v1_chat(request: Request):
        # OpenAI endpoints do NOT transform body
        return await proxy_request("POST", "/v1/chat/completions", request, transform_body=False)

    @app.post("/v1/completions")
    async def post_v1_completions(request: Request):
        return await proxy_request("POST", "/v1/completions", request, transform_body=False)

    @app.post("/v1/responses")
    async def post_v1_responses(request: Request):
        return await proxy_request("POST", "/v1/responses", request, transform_body=False)

    return app
