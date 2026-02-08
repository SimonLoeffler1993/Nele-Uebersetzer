import os
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
from dotenv import load_dotenv

import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeleOpenAIProxy")

# Add file handler for debugging
fh = logging.FileHandler("proxy_debug.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)

NELE_API_KEY = os.getenv("NELE_API_KEY", "YOUR_NELE_API_KEY_HERE")
NELE_BASE_URL = "https://api.aieva.io/api:v1"

app = FastAPI(title="Nele.ai OpenAI Proxy")

@app.get("/v1/models")
async def list_models():
    """
    List models from Nele.ai and transform to OpenAI format.
    """
    nele_url = f"{NELE_BASE_URL}/models"
    headers = {
        "Authorization": f"Bearer {NELE_API_KEY}",
        "Accept": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(nele_url, headers=headers, timeout=30.0)
            if response.status_code != 200:
                logger.error(f"Error fetching models: {response.status_code}")
                return Response(content=response.content, status_code=response.status_code)
            
            nele_data = response.json()
            logger.info(f"Raw Nele.ai models response: {nele_data}")
            
            # Nele.ai returns: {"status": "success", "data": [{"id": "...", "name": "..."}]}
            # Or sometimes: {"models": [...]}
            nele_models = nele_data.get("data", []) or nele_data.get("models", [])
            
            openai_models = []
            for m in nele_models:
                # Handle cases where m is a string or a dict
                if isinstance(m, str):
                    m_id = m
                else:
                    m_id = m.get("id") or m.get("name")
                
                if m_id:
                    openai_models.append({
                        "id": m_id,
                        "object": "model",
                        "created": int(__import__("time").time()),
                        "owned_by": "nele-ai"
                    })
            
            logger.info(f"Transformed to {len(openai_models)} OpenAI models")
            return {
                "object": "list",
                "data": openai_models
            }
        except Exception as e:
            logger.error(f"Exception listing models: {str(e)}")
            # Fallback to a basic list if Nele.ai API is down
            fallback_ids = ["azure-o4-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o-mini"]
            return {
                "object": "list",
                "data": [{"id": fid, "object": "model", "created": 0, "owned_by": "nele-ai"} for fid in fallback_ids]
            }

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_openai(path: str, request: Request):
    """
    Proxy OpenAI API requests to Nele.ai.
    """
    logger.info(f"--- Incoming Request: {request.method} {path} ---")
    logger.info(f"Headers: {dict(request.headers)}")
    
    if path == "models" and request.method == "GET":
        return await list_models()


    # Map OpenAI paths to Nele.ai paths
    path_mapping = {
        "chat/completions": "chat-completion",
    }
    
    nele_path = path_mapping.get(path, path)
    nele_url = f"{NELE_BASE_URL}/{nele_path}"
    
    # Get request body
    body = await request.body()
    import json as json_module
    
    # Process payload to only include supported parameters for Nele.ai
    effective_payload = None
    client_wants_stream = False
    
    if body:
        logger.info(f"Request body: {body.decode('utf-8')}")
        try:
            request_data = json_module.loads(body)
            client_wants_stream = request_data.get("stream", False)
            
            # Nele.ai is EXTREMELY sensitive. Whitelist ONLY what's in their schema.
            # Schema: model (req), messages (req), max_tokens (opt), temperature (opt)
            if path == "chat/completions":
                effective_payload = {
                    "model": request_data.get("model"),
                    "messages": request_data.get("messages")
                }
                
                # Copy optional supported fields
                for field in ["temperature", "max_tokens"]:
                    if field in request_data:
                        effective_payload[field] = request_data[field]
                
                logger.info(f"Filtered payload for Nele (removed stream field): {json_module.dumps(effective_payload)}")
        except Exception as e:
            logger.error(f"Error parsing request body: {e}")
    
    # Use effective_payload if we processed it, otherwise fall back to raw body
    # We prefer using dict/json for httpx to handle headers automatically
    
    # Use the appropriate Nele endpoint
    nele_url = f"{NELE_BASE_URL}/chat-completion"
    
    # Forward headers, add auth
    # We use a MINIMAL set of headers to avoid nginx 400 Errors
    headers = {
        "Authorization": f"Bearer {NELE_API_KEY}",
        "Accept": "text/event-stream"
    }
    
    logger.info(f"Forwarding to Nele: {nele_url} | Headers: {headers}")
    
    # Make request to Nele.ai (always streams)
    async with httpx.AsyncClient() as client:
        # Prepare stream arguments
        stream_kwargs = {
            "method": request.method,
            "url": nele_url,
            "headers": headers,
            "timeout": 60.0
        }
        
        if effective_payload:
            stream_kwargs["json"] = effective_payload
        else:
            stream_kwargs["content"] = body
            headers["Content-Type"] = "application/json" # Manual content needs type
            
        if client_wants_stream:
            # Client wants streaming - pass through
            async def stream_response():
                try:
                    async with client.stream(**stream_kwargs) as response:
                        if response.status_code != 200:
                            error_body = await response.aread()
                            logger.error(f"Nele API Error ({response.status_code}): {error_body.decode('utf-8')}")
                            yield json_module.dumps({"error": error_body.decode('utf-8')}).encode('utf-8')
                            return

                        async for chunk in response.aiter_bytes():
                            yield chunk
                except Exception as e:
                    logger.error(f"Streaming Exception: {e}")
                    yield json_module.dumps({"error": str(e)}).encode('utf-8')
            
            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            # Client wants non-streaming - aggregate the stream
            full_content = ""
            model = ""
            finish_reason = None
            
            try:
                async with client.stream(**stream_kwargs) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        logger.error(f"Nele API Error ({response.status_code}): {error_body.decode('utf-8')}")
                        return Response(
                            content=error_body,
                            status_code=response.status_code,
                            headers=dict(response.headers)
                        )
                    
                    # Aggregate stream chunks
                    async for line in response.aiter_lines():
                        if line.strip():
                            stripped = line.strip()
                            
                            if stripped == "data: [DONE]":
                                continue
                            
                            if stripped.startswith("data: "):
                                try:
                                    chunk_json = json_module.loads(stripped[6:])
                                    
                                    # Skip usage chunks
                                    if chunk_json.get("object") == "usage.credits":
                                        continue
                                    
                                    # Extract data
                                    if chunk_json.get("object") == "chat.completion.chunk":
                                        model = chunk_json.get("model", model)
                                        choices = chunk_json.get("choices", [])
                                        if choices:
                                            delta = choices[0].get("delta", {})
                                            content = delta.get("content", "")
                                            full_content += content
                                            finish_reason = choices[0].get("finish_reason") or finish_reason
                                except Exception as e:
                                    logger.warning(f"Error parsing chunk: {e} | Line: {line}")
                                    pass
            except Exception as e:
                logger.error(f"Chat Exception: {e}")
                return Response(content=str(e), status_code=500)

            
            # Build standard OpenAI response
            openai_response = {
                "id": "chatcmpl-proxy",
                "object": "chat.completion",
                "created": int(__import__("time").time()),
                "model": model or (request_data.get("model") if body else "unknown"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_content
                    },
                    "finish_reason": finish_reason or "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
            return Response(
                content=json_module.dumps(openai_response),
                media_type="application/json"
            )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
