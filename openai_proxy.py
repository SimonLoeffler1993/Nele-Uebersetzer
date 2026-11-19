import json
import logging
import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nele_files import AttachmentError, ImageUploader, extract_attachments, has_file_parts
from nele_stream import (
    TOOL_MODE_NATIVE,
    TOOL_MODE_PROMPT,
    StreamTranslator,
    UpstreamStreamError,
    aggregate_events,
    build_nele_payload,
    parse_sse_lines,
)

# Load environment variables
load_dotenv()

# Setup logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("NeleOpenAIProxy")

LOG_FILE = os.getenv("LOG_FILE", "proxy_debug.log")
if LOG_FILE:
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(LOG_LEVEL)
    logger.addHandler(fh)

NELE_API_KEY = os.getenv("NELE_API_KEY")
NELE_BASE_URL = os.getenv("NELE_BASE_URL", "https://api.aieva.io/api:v1").rstrip("/")
# "prompt": Tools werden in den System-Prompt geschrieben und aus der Antwort gelesen (bisheriges Verhalten)
# "native": Tools/tool_choice gehen direkt an nele.ai
NELE_TOOL_MODE = os.getenv("NELE_TOOL_MODE", TOOL_MODE_PROMPT).lower()
# Lange Antworten (z. B. Reasoning-Modelle) brauchen einen großzügigen Read-Timeout
NELE_READ_TIMEOUT = float(os.getenv("NELE_READ_TIMEOUT", "300"))

app = FastAPI(title="Nele.ai OpenAI Proxy")


def get_auth_key(request: Request = None, fallback_key: str = None) -> str:
    """
    Extract API key from Authorization header or fallback to env var.
    """
    if request:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header.replace("Bearer ", "").strip()
    return fallback_key if fallback_key is not None else NELE_API_KEY


def redact_headers(headers) -> dict:
    return {k: ("***" if k.lower() in ("authorization", "api-key", "x-api-key") else v) for k, v in headers.items()}


def openai_error(status_code: int, message: str, err_type: str = "upstream_error", code=None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": err_type, "param": None, "code": code}},
    )


def upstream_error_response(status_code: int, body: bytes) -> JSONResponse:
    """Fehler von nele.ai in das OpenAI-Fehlerformat übersetzen (Status bleibt erhalten)."""
    text = body.decode("utf-8", errors="replace")
    message = text
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                message = err.get("message") or json.dumps(err)
            elif err:
                message = str(err)
            else:
                message = data.get("message") or text
    except Exception:
        pass
    err_type = {
        401: "authentication_error",
        402: "insufficient_quota",
        403: "permission_error",
        404: "not_found_error",
        422: "invalid_request_error",
        429: "rate_limit_error",
    }.get(status_code, "upstream_error")
    return openai_error(status_code, f"nele.ai: {message}", err_type, code=status_code)


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(NELE_READ_TIMEOUT, connect=15.0))


@app.get("/")
async def root():
    return {"message": "Nele.ai OpenAI Proxy"}


@app.get("/v1/models")
@app.get("/models")
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
                return upstream_error_response(response.status_code, response.content)

            nele_data = response.json()
            logger.debug(f"Raw Nele.ai models response: {nele_data}")

            # User requirement: Only use models from the "models" category
            nele_models = nele_data.get("models", [])

            # Fallback for older structure if "models" is missing but "data" exists
            if not nele_models and "data" in nele_data:
                nele_models = nele_data.get("data", [])

            openai_models = []
            for m in nele_models:
                if isinstance(m, str):
                    m_id, is_usable = m, True
                else:
                    m_id = m.get("id") or m.get("title") or m.get("name")
                    is_usable = m.get("is_usable", True)

                # Filter by is_usable unless alle is True
                if not alle and not is_usable:
                    continue

                if m_id:
                    openai_models.append({
                        "id": m_id,
                        "object": "model",
                        "created": int(time.time()),
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


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-kompatibler Chat-Endpunkt.
    `stream: true`  -> Server-Sent Events im OpenAI-Format, Token für Token
    `stream: false` -> eine vollständige `chat.completion`
    """
    logger.info("--- Incoming Request: POST chat/completions ---")
    logger.debug(f"Headers: {redact_headers(request.headers)}")

    try:
        request_data = await request.json()
        if not isinstance(request_data, dict):
            raise ValueError("Body muss ein JSON-Objekt sein")
    except Exception as e:
        return openai_error(400, f"Ungültiger JSON-Body: {e}", "invalid_request_error")

    if not request_data.get("model"):
        return openai_error(400, "Feld 'model' fehlt.", "invalid_request_error")
    if not request_data.get("messages"):
        return openai_error(400, "Feld 'messages' fehlt oder ist leer.", "invalid_request_error")

    client_wants_stream = bool(request_data.get("stream", False))
    include_usage = bool((request_data.get("stream_options") or {}).get("include_usage"))
    tools_in_prompt = bool(request_data.get("tools")) and NELE_TOOL_MODE != TOOL_MODE_NATIVE

    api_key = get_auth_key(request)

    # Dateien/Bilder -> nele.ai-Attachments (Bilder werden vorher hochgeladen)
    if has_file_parts(request_data["messages"]):
        async with make_client() as upload_client:
            try:
                request_data["messages"] = await extract_attachments(
                    request_data["messages"],
                    ImageUploader(NELE_BASE_URL, api_key, upload_client),
                    upload_client,
                )
            except AttachmentError as e:
                return openai_error(e.status_code, str(e), "invalid_request_error")
            except httpx.HTTPError as e:
                return openai_error(502, f"Datei-Upload zu nele.ai fehlgeschlagen: {e}")

    payload = build_nele_payload(request_data, NELE_TOOL_MODE)
    logger.debug(f"Payload for Nele: {json.dumps(payload, ensure_ascii=False)[:5000]}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream",
    }
    nele_url = f"{NELE_BASE_URL}/chat-completion"

    # nele.ai streamt immer. Wir öffnen die Verbindung, BEVOR wir dem Client
    # antworten, damit Fehler (401, 402, 422 ...) als echter HTTP-Status ankommen.
    client = make_client()
    try:
        upstream = await client.send(
            client.build_request("POST", nele_url, headers=headers, json=payload),
            stream=True,
        )
    except httpx.HTTPError as e:
        await client.aclose()
        logger.error(f"Verbindung zu nele.ai fehlgeschlagen: {e}")
        return openai_error(502, f"Verbindung zu nele.ai fehlgeschlagen: {e}")

    if upstream.status_code != 200:
        body = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        logger.error(f"Nele API Error ({upstream.status_code}): {body.decode('utf-8', errors='replace')}")
        return upstream_error_response(upstream.status_code, body)

    events = parse_sse_lines(upstream.aiter_lines())

    if not client_wants_stream:
        try:
            result = await aggregate_events(events, request_data.get("model"), tools_in_prompt)
        except UpstreamStreamError as e:
            return openai_error(502, f"nele.ai: {e.error.get('message')}")
        except httpx.HTTPError as e:
            return openai_error(502, f"Stream von nele.ai abgebrochen: {e}")
        finally:
            await upstream.aclose()
            await client.aclose()
        return JSONResponse(content=result)

    translator = StreamTranslator(
        model=request_data.get("model"),
        tools_in_prompt=tools_in_prompt,
        include_usage=include_usage,
    )

    async def stream_response():
        try:
            for out in translator.start():
                yield out
            async for ev in events:
                for out in translator.feed(ev):
                    yield out
            for out in translator.finish():
                yield out
        except httpx.HTTPError as e:
            logger.error(f"Stream von nele.ai abgebrochen: {e}")
            yield f"data: {json.dumps({'error': {'message': f'Stream von nele.ai abgebrochen: {e}', 'type': 'upstream_error'}})}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
        finally:
            # Läuft auch, wenn der Client die Verbindung trennt
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx & Co. sollen nicht puffern
        },
    )


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def unsupported(path: str, request: Request):
    logger.warning(f"Nicht unterstützter Endpunkt: {request.method} /v1/{path}")
    return openai_error(
        404,
        f"Endpunkt /v1/{path} wird von diesem Proxy nicht unterstützt. "
        "Verfügbar: POST /v1/chat/completions, GET /v1/models",
        "invalid_request_error",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
