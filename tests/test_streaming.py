import json
import time

import httpx
import pytest
from openai import OpenAI

from conftest import nele_chunk


def client(proxy_url):
    return OpenAI(base_url=proxy_url, api_key="test-key")


def text_events(*parts):
    evs = [nele_chunk(p) for p in parts]
    evs.append(nele_chunk(finish_reason="stop"))
    evs.append({"object": "usage.credits", "credits": 3})
    evs.append("[DONE]")
    return evs


def test_stream_delivers_tokens_incrementally(nele, proxy_url):
    nele.script = {"status": 200, "events": text_events("Hal", "lo", " Welt"), "delay": 0.4}

    start = time.monotonic()
    stream = client(proxy_url).chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Hi"}], stream=True
    )
    arrivals, texts, finish = [], [], None
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            arrivals.append(time.monotonic() - start)
            texts.append(chunk.choices[0].delta.content)
        if chunk.choices and chunk.choices[0].finish_reason:
            finish = chunk.choices[0].finish_reason

    assert "".join(texts) == "Hallo Welt"
    assert finish == "stop"
    # Erster Token muss deutlich vor dem letzten ankommen (kein Puffern)
    assert arrivals[0] < 0.7
    assert arrivals[-1] - arrivals[0] > 0.6


def test_stream_chunk_format_is_openai_like(nele, proxy_url):
    nele.script = {"status": 200, "events": text_events("A", "B"), "delay": 0.01}
    with httpx.stream(
        "POST", f"{proxy_url}/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}], "stream": True},
        headers={"Authorization": "Bearer k"},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["x-accel-buffering"] == "no"
        lines = [l for l in r.iter_lines() if l]

    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    # nele-spezifische usage.credits darf nicht durchrutschen
    assert "usage.credits" not in json.dumps(chunks)


def test_stream_include_usage(nele, proxy_url):
    events = [nele_chunk("ok"), nele_chunk(finish_reason="stop"),
              {"object": "usage.credits", "prompt_tokens": 5, "completion_tokens": 2}, "[DONE]"]
    nele.script = {"status": 200, "events": events, "delay": 0.01}
    stream = client(proxy_url).chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Hi"}], stream=True,
        stream_options={"include_usage": True},
    )
    chunks = list(stream)
    assert chunks[-1].choices == []
    assert chunks[-1].usage.prompt_tokens == 5
    assert chunks[-1].usage.total_tokens == 7


def test_non_stream_still_works(nele, proxy_url):
    nele.script = {"status": 200, "events": text_events("Hallo", " du"), "delay": 0.01}
    resp = client(proxy_url).chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Hi"}]
    )
    assert resp.choices[0].message.content == "Hallo du"
    assert resp.choices[0].finish_reason == "stop"


def test_upstream_error_returns_http_status(nele, proxy_url):
    nele.script = {"status": 401, "body": {"message": "Unauthenticated."}}
    with pytest.raises(Exception) as exc:
        client(proxy_url).chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "Hi"}], stream=True
        )
    assert getattr(exc.value, "status_code", None) == 401
    assert "Unauthenticated" in str(exc.value)


def test_prompt_mode_tool_call_in_stream(nele, proxy_url):
    events = [nele_chunk('{"name": "wetter", '), nele_chunk('"arguments": {"stadt": "Berlin"}}'),
              nele_chunk(finish_reason="stop"), "[DONE]"]
    nele.script = {"status": 200, "events": events, "delay": 0.01}
    tools = [{"type": "function", "function": {"name": "wetter", "parameters": {"type": "object"}}}]
    stream = client(proxy_url).chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Wetter?"}], stream=True, tools=tools
    )
    calls, finish, content = [], None, ""
    for c in stream:
        d = c.choices[0].delta if c.choices else None
        if d and d.tool_calls:
            calls += d.tool_calls
        if d and d.content:
            content += d.content
        if c.choices and c.choices[0].finish_reason:
            finish = c.choices[0].finish_reason
    assert finish == "tool_calls"
    assert content == ""
    assert calls[0].function.name == "wetter"
    assert json.loads(calls[0].function.arguments) == {"stadt": "Berlin"}
    # Tools wurden im Prompt-Modus in den System-Prompt geschrieben
    body = nele.requests[-1]["body"]
    assert "tools" not in body
    assert "wetter" in body["messages"][0]["content"]


def test_prompt_mode_plain_text_with_tools_streams_immediately(nele, proxy_url):
    nele.script = {"status": 200, "events": text_events("Es ", "ist ", "sonnig"), "delay": 0.4}
    tools = [{"type": "function", "function": {"name": "wetter", "parameters": {"type": "object"}}}]
    start = time.monotonic()
    stream = client(proxy_url).chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Hi"}], stream=True, tools=tools
    )
    arrivals, text = [], ""
    for c in stream:
        if c.choices and c.choices[0].delta.content:
            arrivals.append(time.monotonic() - start)
            text += c.choices[0].delta.content
    assert text == "Es ist sonnig"
    assert arrivals[-1] - arrivals[0] > 0.6


def test_native_tool_mode_passes_tools_through(nele, proxy_url, proxy_module):
    proxy_module.NELE_TOOL_MODE = "native"
    tc = [{"index": 0, "id": "call_1", "type": "function",
           "function": {"name": "wetter", "arguments": "{\"stadt\":"}}]
    tc2 = [{"index": 0, "function": {"arguments": "\"Berlin\"}"}}]
    events = [nele_chunk(tool_calls=tc), nele_chunk(tool_calls=tc2),
              nele_chunk(finish_reason="tool_calls"), "[DONE]"]
    nele.script = {"status": 200, "events": events, "delay": 0.01}
    tools = [{"type": "function", "function": {"name": "wetter", "parameters": {"type": "object"}}}]

    resp = client(proxy_url).chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Wetter?"}], tools=tools, tool_choice="auto"
    )
    call = resp.choices[0].message.tool_calls[0]
    assert call.function.name == "wetter"
    assert json.loads(call.function.arguments) == {"stadt": "Berlin"}
    body = nele.requests[-1]["body"]
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"


def test_payload_mapping(nele, proxy_url):
    nele.script = {"status": 200, "events": text_events("x"), "delay": 0.01}
    client(proxy_url).chat.completions.create(
        model="gpt-5", temperature=1.5, max_completion_tokens=100, reasoning_effort="low",
        messages=[
            {"role": "developer", "content": "Sei kurz."},
            {"role": "user", "content": [{"type": "text", "text": "Hallo"}]},
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                                                  "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "42"},
        ],
        extra_body={"documentCollectionId": "abc", "web_search": None},
    )
    body = nele.requests[-1]["body"]
    assert body["temperature"] == 1.0
    assert body["max_tokens"] == 100
    assert body["modelConfiguration"] == {"reasoning_effort": "low"}
    assert body["documentCollectionId"] == "abc"
    assert "web_search" not in body
    assert "stream" not in body
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert body["messages"][1]["content"] == "Hallo"
    assert nele.requests[-1]["headers"]["authorization"] == "Bearer test-key"


def test_models(nele, proxy_url):
    models = client(proxy_url).models.list()
    assert [m.id for m in models.data] == ["gpt-4o"]
