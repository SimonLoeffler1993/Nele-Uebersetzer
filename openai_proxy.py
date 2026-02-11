import os
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
from dotenv import load_dotenv
import json as json_module
import re

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

NELE_API_KEY = os.getenv("NELE_API_KEY")
NELE_BASE_URL = os.getenv("NELE_BASE_URL", "https://api.aieva.io/api:v1")

app = FastAPI(title="Nele.ai OpenAI Proxy")

def get_auth_key(request: Request = None, fallback_key: str = NELE_API_KEY) -> str:
    """
    Extract API key from Authorization header or fallback to env var.
    """
    if request:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header.replace("Bearer ", "").strip()
    return fallback_key

def inject_tool_instructions(messages: list, tools: list) -> list:
    """
    Inject tool definitions and formatting instructions into the prompt.
    Forces the model to output specific JSON format for tool calls.
    """
    if not tools:
        return messages
    
    tool_descriptions = []
    for tool in tools:
        if tool.get("type") == "function":
            f = tool["function"]
            desc = f.get("description", "No description")
            params = json_module.dumps(f.get("parameters", {}))
            tool_descriptions.append(f"- {f['name']}: {desc}. Arguments: {params}")
    
    instruction = (
        "\n\nIMPORTANT: You have tools available. If you need to use a tool, respond ONLY with a JSON object "
        "in this format: {\"name\": \"tool_name\", \"arguments\": { ... }}\n"
        "Do not include any conversational text BEFORE or AFTER the JSON if you are calling a tool.\n"
        "Available tools:\n" + "\n".join(tool_descriptions)
    )
    
    # Create a copy to avoid mutating the original
    new_messages = [m.copy() for m in messages]
    
    # Try to find a system message to append to
    system_msg = next((m for m in new_messages if m.get("role") == "system"), None)
    if system_msg:
        logger.info("Appending tool instructions to existing system message.")
        system_msg["content"] += instruction
    else:
        logger.info("No system message found, prepending one for tool instructions.")
        new_messages.insert(0, {"role": "system", "content": instruction})
    
    return new_messages

def sanitize_messages_for_nele(messages: list) -> list:
    """
    Maps messages with roles Nele.ai doesn't support (like 'tool') to standard roles.
    Removes fields like 'tool_calls' that Nele.ai rejects.
    """
    sanitized = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        tool_calls = m.get("tool_calls")
        
        if role == "tool":
            # Map tool result to user message
            logger.info(f"Mapping 'tool' message to 'user' for tool_id: {m.get('tool_call_id')}")
            sanitized.append({
                "role": "user",
                "content": f"System (Tool Output): {content}"
            })
        elif role == "assistant" and tool_calls:
            # Map assistant tool call back to JSON text so LLM sees it as plain text history
            tc = tool_calls[0]["function"]
            json_text = json_module.dumps({"name": tc["name"], "arguments": tc["arguments"]})
            logger.info(f"Mapping 'assistant' tool_calls back to JSON text: {tc['name']}")
            sanitized.append({
                "role": "assistant",
                "content": json_text
            })
        else:
            # Strip extra fields and ensure role/content are valid
            sanitized.append({
                "role": role,
                "content": content or ""
            })
    return sanitized

def extract_tool_call(content: str) -> dict:
    """
    Extracts a JSON tool call from string content.
    Supports raw JSON or Markdown-wrapped JSON.
    Handles 'id' or 'name' as function identifier.
    """
    if not content:
        return None
        
    content_trimmed = content.strip()
    
    def parse_if_tool(data_str):
        try:
            data = json_module.loads(data_str)
            # Support both 'id' (Nele) and 'name' (Standard)
            name = data.get("id") or data.get("name")
            # Nele might send 'arguments' as a dict or a stringified JSON
            if name and ("arguments" in data or "args" in data):
                args = data.get("arguments") or data.get("args") or {}
                if not isinstance(args, str):
                    args = json_module.dumps(args)
                return {"name": name, "arguments": args}
        except:
            pass
        return None

    # 1. Direct JSON (with potential prefix/suffix cleanup)
    content_clean = content_trimmed.strip()
    if content_clean.startswith("{") and content_clean.endswith("}"):
        res = parse_if_tool(content_clean)
        if res: return res

    # 2. Markdown Block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_clean, re.DOTALL)
    if match:
        res = parse_if_tool(match.group(1))
        if res: return res
        
    # 3. Fuzzy search for first JSON object
    # Greedy .* to match the last closing brace in case of nesting
    match = re.search(r"(\{.*\})", content_clean, re.DOTALL)
    if match:
        res = parse_if_tool(match.group(1))
        if res: return res
        
    return None

@app.get("/")
async def root():
    return {"message": "Nele.ai OpenAI Proxy"}

@app.get("/v1/models")
async def list_models(request: Request, alle: bool = False):
    """
    List models from Nele.ai and transform to OpenAI format.
    """
    api_key = get_auth_key(request)
    nele_url = f"{NELE_BASE_URL}/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
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
            
            # User requirement: Only use models from the "models" category
            nele_models = nele_data.get("models", [])
            
            # Fallback for older structure if "models" is missing but "data" exists
            if not nele_models and "data" in nele_data:
                nele_models = nele_data.get("data", [])
            
            openai_models = []
            for m in nele_models:
                # Handle cases where m is a string or a dict
                # if isinstance(m, str):
                #     m_id = m
                #     is_usable = True
                # else:
                m_id = m.get("id") or m.get("title") or m.get("name")
                is_usable = m.get("is_usable", True)
                
                # Filter by is_usable unless alle is True
                if not alle and not is_usable:
                   continue 
                
                if m_id:
                    openai_models.append({
                        "id": m_id,
                        "object": "model",
                        "created": int(__import__("time").time()),
                        "owned_by": "nele-ai"
                    })
            
            logger.info(f"Transformed to {len(openai_models)} OpenAI models (alle={alle})")
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
        return await list_models(request)


    # Map OpenAI paths to Nele.ai paths
    path_mapping = {
        "chat/completions": "chat-completion",
    }
    
    nele_path = path_mapping.get(path, path)
    nele_url = f"{NELE_BASE_URL}/{nele_path}"
    
    # Get request body
    body = await request.body()
    
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
                messages = request_data.get("messages", [])
                tools = request_data.get("tools")
                
                # If tools are present, inject them into the prompt
                if tools:
                    logger.info(f"Injecting {len(tools)} tools into the prompt.")
                    messages = inject_tool_instructions(messages, tools)
                
                # Sanitize ALL messages for Nele compatibility (maps tool role to user)
                messages = sanitize_messages_for_nele(messages)
                
                effective_payload = {
                    "model": request_data.get("model"),
                    "messages": messages
                }
                
                # Copy optional supported fields (strictly Nele-supported)
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
    api_key = get_auth_key(request)
    # We use a MINIMAL set of headers to avoid nginx 400 Errors
    headers = {
        "Authorization": f"Bearer {api_key}",
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
                        full_buffer = ""
                        potential_tool_call = False
                        
                        async with client.stream(**stream_kwargs) as response:
                            if response.status_code != 200:
                                error_body = await response.aread()
                                logger.error(f"Nele API Error ({response.status_code}): {error_body.decode('utf-8')}")
                                yield json_module.dumps({"error": error_body.decode('utf-8')}).encode('utf-8')
                                return

                            async for chunk in response.aiter_bytes():
                                decoded = chunk.decode('utf-8', errors='ignore')
                                
                                # Heuristic: if first real data looks like JSON start, buffer it
                                if not full_buffer:
                                    stripped = decoded.strip()
                                    if stripped.startswith("data: {") or stripped.startswith("data: ```"):
                                        potential_tool_call = True
                                        logger.info("Stream looks like a tool call, starting buffer.")
                                
                                if potential_tool_call:
                                    full_buffer += decoded
                                    # We wait until the end of the stream to extract the tool call
                                else:
                                    yield chunk
                                    
                        if potential_tool_call:
                            # Aggregate the full response content
                            aggregated_content = ""
                            model_name = "unknown"
                            for line in full_buffer.splitlines():
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    try:
                                        j = json_module.loads(line[6:])
                                        if j.get("object") == "chat.completion.chunk":
                                            model_name = j.get("model", model_name)
                                            choices = j.get("choices", [])
                                            if choices:
                                                aggregated_content += choices[0].get("delta", {}).get("content", "")
                                    except: pass
                            
                            tool_call = extract_tool_call(aggregated_content)
                            if tool_call:
                                logger.info(f"Detected tool call in stream: {tool_call['name']}")
                                # Yield a single OpenAI tool_call chunk
                                chunk = {
                                    "id": "chatcmpl-proxy",
                                    "object": "chat.completion.chunk",
                                    "created": int(__import__("time").time()),
                                    "model": model_name,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [{
                                                "index": 0,
                                                "id": f"call_{int(__import__('time').time())}",
                                                "type": "function",
                                                "function": tool_call
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }]
                                }
                                yield f"data: {json_module.dumps(chunk)}\n\n".encode('utf-8')
                                yield b"data: [DONE]\n\n"
                            else:
                                # Not a tool call after all, dump the buffer
                                yield full_buffer.encode('utf-8')
                    except Exception as e:
                        logger.error(f"Stream error: {e}")
                        yield f"data: {json_module.dumps({'error': str(e)})}\n\n".encode('utf-8')

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

            
            # Check for tool calls in the content
            logger.info(f"Full aggregated content: {full_content}")
            tool_call = extract_tool_call(full_content)
            
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
                        "content": None if tool_call else full_content
                    },
                    "finish_reason": "tool_calls" if tool_call else (finish_reason or "stop")
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
            if tool_call:
                logger.info(f"Detected tool call in aggregate: {tool_call['name']}")
                openai_response["choices"][0]["message"]["tool_calls"] = [{
                    "id": f"call_{int(__import__('time').time())}",
                    "type": "function",
                    "function": tool_call
                }]
            
            return Response(
                content=json_module.dumps(openai_response),
                media_type="application/json"
            )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
