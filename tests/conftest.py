"""
Test-Setup: ein simulierter nele.ai-Server und der echte Proxy laufen
jeweils mit uvicorn auf einem freien Port.
"""
import asyncio
import json
import os
import socket
import sys
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHUNK_DELAY = 0.3


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def nele_chunk(content=None, finish_reason=None, tool_calls=None, model="gpt-4o"):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {
        "id": "nele-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


class FakeNele:
    """Simulierter nele.ai-Server. `script` bestimmt die nächste Antwort."""

    def __init__(self):
        self.app = FastAPI()
        self.requests = []
        self.script = {"status": 200, "events": [], "delay": CHUNK_DELAY}

        @self.app.post("/chat-completion")
        async def chat(request: Request):
            body = await request.json()
            self.requests.append({"body": body, "headers": dict(request.headers)})
            script = self.script
            if script["status"] != 200:
                return JSONResponse(status_code=script["status"], content=script.get("body", {"message": "error"}))

            async def gen():
                for ev in script["events"]:
                    await asyncio.sleep(script["delay"])
                    if ev == "[DONE]":
                        yield b"data: [DONE]\n\n"
                    else:
                        yield f"data: {json.dumps(ev)}\n\n".encode()

            return StreamingResponse(gen(), media_type="text/event-stream")

        self.uploads = []

        @self.app.post("/image-attachment")
        async def upload(request: Request):
            form = await request.form()
            f = form["file"]
            data = await f.read()
            self.uploads.append({"filename": f.filename, "content_type": f.content_type, "size": len(data)})
            return {"path": f"attachments/{len(self.uploads)}-{f.filename}", "expires_at": int(time.time()) + 3600}

        @self.app.get("/models")
        async def models():
            return {"models": [{"id": "gpt-4o", "is_usable": True}, {"id": "old", "is_usable": False}]}


def run_server(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            return server
        time.sleep(0.05)
    raise RuntimeError("server did not start")


@pytest.fixture(scope="session")
def fake_nele():
    fake = FakeNele()
    port = free_port()
    server = run_server(fake.app, port)
    fake.url = f"http://127.0.0.1:{port}"
    yield fake
    server.should_exit = True


@pytest.fixture(scope="session")
def proxy_module(fake_nele):
    os.environ["NELE_BASE_URL"] = fake_nele.url
    os.environ["LOG_FILE"] = ""
    import openai_proxy
    openai_proxy.NELE_BASE_URL = fake_nele.url
    return openai_proxy


@pytest.fixture(scope="session")
def proxy_url(proxy_module):
    port = free_port()
    server = run_server(proxy_module.app, port)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True


@pytest.fixture
def nele(fake_nele, proxy_module):
    fake_nele.requests.clear()
    fake_nele.uploads.clear()
    fake_nele.script = {"status": 200, "events": [], "delay": 0.01}
    proxy_module.NELE_TOOL_MODE = "prompt"
    return fake_nele
