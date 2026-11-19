"""
Übersetzung zwischen OpenAI Chat Completions und nele.ai.

Diese Datei enthält nur reine Logik (kein FastAPI, kein HTTP), damit sie
leicht testbar ist:

- build_nele_payload(): OpenAI-Request -> nele.ai-Request
- parse_sse_lines():    nele.ai-SSE-Zeilen -> JSON-Events
- StreamTranslator:     nele.ai-Events -> OpenAI `chat.completion.chunk`s
- aggregate_events():   nele.ai-Events -> eine OpenAI `chat.completion`
"""

import json
import logging
import re
import time
import uuid
from typing import AsyncIterator, Dict, Iterable, List, Optional

logger = logging.getLogger("NeleOpenAIProxy")

# Rollen, die nele.ai akzeptiert
NELE_ROLES = {"system", "user", "assistant"}

# Felder, die zusätzlich zu model/messages 1:1 an nele.ai gehen dürfen.
# (nele.ai-spezifische Erweiterungen kommen bei OpenAI-Clients über `extra_body`.)
NELE_PASSTHROUGH_FIELDS = [
    "documentCollectionId",
    "documentCollectionOptions",
    "web_search",
]

TOOL_MODE_PROMPT = "prompt"
TOOL_MODE_NATIVE = "native"


# --------------------------------------------------------------------------- #
# Request: OpenAI -> nele.ai
# --------------------------------------------------------------------------- #

def flatten_content(content) -> str:
    """OpenAI erlaubt `content` als Liste von Parts; nele.ai erwartet einen String."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                parts.append(part.get("text", ""))
            # Bilder/Audio etc. werden von dieser Schnittstelle nicht unterstützt
        return "\n".join(p for p in parts if p)
    return str(content)


def inject_tool_instructions(messages: List[dict], tools: List[dict]) -> List[dict]:
    """
    Tool-Definitionen als Anweisung in den System-Prompt schreiben (Prompt-Modus).
    Das Modell soll bei einem Tool-Aufruf nur JSON im Format
    {"name": ..., "arguments": {...}} antworten.
    """
    if not tools:
        return messages

    tool_descriptions = []
    for tool in tools:
        if tool.get("type") == "function":
            f = tool["function"]
            desc = f.get("description", "No description")
            params = json.dumps(f.get("parameters", {}))
            tool_descriptions.append(f"- {f['name']}: {desc}. Arguments: {params}")

    instruction = (
        "\n\nIMPORTANT: You have tools available. If you need to use a tool, respond ONLY with a JSON object "
        "in this format: {\"name\": \"tool_name\", \"arguments\": { ... }}\n"
        "Do not include any conversational text BEFORE or AFTER the JSON if you are calling a tool.\n"
        "Available tools:\n" + "\n".join(tool_descriptions)
    )

    new_messages = [m.copy() for m in messages]
    system_msg = next((m for m in new_messages if m.get("role") == "system"), None)
    if system_msg:
        system_msg["content"] = flatten_content(system_msg.get("content")) + instruction
    else:
        new_messages.insert(0, {"role": "system", "content": instruction.lstrip()})
    return new_messages


def sanitize_messages_for_nele(messages: List[dict]) -> List[dict]:
    """
    nele.ai kennt nur system/user/assistant und nur einfache Felder.
    - `tool`-Nachrichten werden zu `user`-Nachrichten
    - `assistant` mit `tool_calls` wird zu JSON-Text
    - `developer` (neuere OpenAI-Clients) wird zu `system`
    """
    sanitized = []
    for m in messages:
        role = m.get("role")
        content = flatten_content(m.get("content"))
        tool_calls = m.get("tool_calls")

        if role == "tool":
            sanitized.append({"role": "user", "content": f"System (Tool Output): {content}"})
        elif role == "assistant" and tool_calls:
            texts = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                texts.append(json.dumps({"name": fn.get("name"), "arguments": fn.get("arguments")}))
            text = "\n".join(texts)
            if content:
                text = content + "\n" + text
            sanitized.append({"role": "assistant", "content": text})
        else:
            if role == "developer":
                role = "system"
            if role not in NELE_ROLES:
                role = "user"
            msg = {"role": role, "content": content}
            if m.get("name"):
                msg["name"] = m["name"]
            sanitized.append(msg)
    return sanitized


def build_nele_payload(request_data: dict, tool_mode: str = TOOL_MODE_PROMPT) -> dict:
    """Erzeugt aus einem OpenAI-Chat-Request den Body für POST /chat-completion."""
    messages = request_data.get("messages", []) or []
    tools = request_data.get("tools") or None

    if tools and tool_mode == TOOL_MODE_PROMPT:
        messages = inject_tool_instructions(messages, tools)

    payload = {
        "model": request_data.get("model"),
        "messages": sanitize_messages_for_nele(messages),
    }

    if tools and tool_mode == TOOL_MODE_NATIVE:
        payload["tools"] = tools
        tool_choice = request_data.get("tool_choice")
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    # max_tokens (neuere Clients senden max_completion_tokens)
    max_tokens = request_data.get("max_tokens") or request_data.get("max_completion_tokens")
    if max_tokens:
        payload["max_tokens"] = max_tokens

    # nele.ai erlaubt temperature nur zwischen 0 und 1 (OpenAI bis 2)
    if request_data.get("temperature") is not None:
        try:
            payload["temperature"] = max(0.0, min(1.0, float(request_data["temperature"])))
        except (TypeError, ValueError):
            pass

    # reasoning_effort (OpenAI) -> modelConfiguration.reasoning_effort (nele.ai)
    model_config = dict(request_data.get("modelConfiguration") or {})
    if request_data.get("reasoning_effort") and "reasoning_effort" not in model_config:
        model_config["reasoning_effort"] = request_data["reasoning_effort"]
    if model_config:
        payload["modelConfiguration"] = model_config

    for field in NELE_PASSTHROUGH_FIELDS:
        if request_data.get(field) is not None:
            payload[field] = request_data[field]

    return payload


# --------------------------------------------------------------------------- #
# Tool-Call-Erkennung im Prompt-Modus
# --------------------------------------------------------------------------- #

def extract_tool_call(content: str) -> Optional[dict]:
    """
    Sucht einen JSON-Tool-Aufruf im Text (roh oder in einem Markdown-Block).
    Akzeptiert `name` oder `id` als Funktionsname.
    """
    if not content:
        return None

    def parse_if_tool(data_str):
        try:
            data = json.loads(data_str)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        name = data.get("id") or data.get("name")
        if name and ("arguments" in data or "args" in data):
            args = data.get("arguments") or data.get("args") or {}
            if not isinstance(args, str):
                args = json.dumps(args)
            return {"name": name, "arguments": args}
        return None

    content_clean = content.strip()

    if content_clean.startswith("{") and content_clean.endswith("}"):
        res = parse_if_tool(content_clean)
        if res:
            return res

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_clean, re.DOTALL)
    if match:
        res = parse_if_tool(match.group(1))
        if res:
            return res

    match = re.search(r"(\{.*\})", content_clean, re.DOTALL)
    if match:
        res = parse_if_tool(match.group(1))
        if res:
            return res

    return None


# --------------------------------------------------------------------------- #
# SSE parsen
# --------------------------------------------------------------------------- #

DONE = object()  # Marker für `data: [DONE]`


async def parse_sse_lines(lines: AsyncIterator[str]) -> AsyncIterator[object]:
    """
    Liest SSE-Zeilen und liefert pro Event das geparste JSON (dict) oder DONE.
    Mehrzeilige `data:`-Felder werden nach SSE-Spezifikation zusammengefügt.
    """
    data_lines: List[str] = []

    def flush():
        if not data_lines:
            return None
        data = "\n".join(data_lines)
        data_lines.clear()
        if data.strip() == "[DONE]":
            return DONE
        try:
            return json.loads(data)
        except Exception:
            logger.warning(f"Kann SSE-Daten nicht parsen: {data[:200]}")
            return None

    async for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            ev = flush()
            if ev is not None:
                yield ev
            continue
        if line.startswith(":"):
            continue  # Kommentar / Keep-Alive
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            # Manche Server trennen Events nicht mit Leerzeilen: dann ist jede
            # data-Zeile bereits ein vollständiges JSON-Event.
            if data_lines and _is_complete_json(data_lines[-1]):
                ev = flush()
                if ev is not None:
                    yield ev
            data_lines.append(value)
        # event:/id:/retry: werden ignoriert
    ev = flush()
    if ev is not None:
        yield ev


def _is_complete_json(s: str) -> bool:
    s = s.strip()
    if s == "[DONE]":
        return True
    try:
        json.loads(s)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Usage
# --------------------------------------------------------------------------- #

def usage_from_event(ev: dict) -> Optional[dict]:
    """Versucht Token-Zahlen aus einem nele.ai-Event zu lesen."""
    src = ev.get("usage") if isinstance(ev.get("usage"), dict) else None
    if src is None and ev.get("object") == "usage.credits":
        src = ev
    if not src:
        return None
    prompt = src.get("prompt_tokens", src.get("input_tokens"))
    completion = src.get("completion_tokens", src.get("output_tokens"))
    if prompt is None and completion is None:
        return None
    prompt = int(prompt or 0)
    completion = int(completion or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(src.get("total_tokens") or prompt + completion),
    }


EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# --------------------------------------------------------------------------- #
# Stream: nele.ai -> OpenAI
# --------------------------------------------------------------------------- #

def sse(data) -> bytes:
    if data is DONE:
        return b"data: [DONE]\n\n"
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


class StreamTranslator:
    """
    Wandelt nele.ai-Events in saubere OpenAI-Stream-Chunks um.

    - Gleiche `id` und `created` über den ganzen Stream
    - Erstes Delta enthält `role: assistant`
    - Text wird sofort weitergereicht
    - Im Prompt-Tool-Modus wird nur gepuffert, wenn der Antworttext wie ein
      JSON-Tool-Aufruf beginnt (`{` oder ```)
    - Am Ende: Chunk mit `finish_reason`, optional Usage-Chunk, `[DONE]`
    """

    def __init__(self, model: str, tools_in_prompt: bool = False, include_usage: bool = False):
        self.id = f"chatcmpl-{uuid.uuid4().hex}"
        self.created = int(time.time())
        self.model = model or "unknown"
        self.tools_in_prompt = tools_in_prompt
        self.include_usage = include_usage

        self.role_sent = False
        self.finish_reason: Optional[str] = None
        self.usage: Optional[dict] = None
        self.native_tool_calls = False

        # Prompt-Tool-Modus: "undecided" -> "stream" | "buffer"
        self.mode = "undecided" if tools_in_prompt else "stream"
        self.pending = ""  # zurückgehaltener Text

    # -- Helfer ------------------------------------------------------------- #

    def _chunk(self, delta: dict, finish_reason=None) -> dict:
        if not self.role_sent:
            delta = {"role": "assistant", **delta}
            self.role_sent = True
        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }],
        }

    def _content(self, text: str) -> Iterable[bytes]:
        if not text:
            return
        if self.mode == "stream":
            yield sse(self._chunk({"content": text}))
            return

        self.pending += text
        if self.mode == "buffer":
            return
        stripped = self.pending.lstrip()
        if not stripped:
            return  # nur Whitespace, noch keine Entscheidung
        if stripped.startswith("{") or stripped.startswith("`"):
            logger.info("Antwort sieht wie ein Tool-Aufruf aus, puffere bis zum Ende.")
            self.mode = "buffer"
        else:
            self.mode = "stream"
            text, self.pending = self.pending, ""
            yield sse(self._chunk({"content": text}))

    # -- Öffentliche API ---------------------------------------------------- #

    def start(self) -> Iterable[bytes]:
        """Erster Chunk mit der Rolle, damit Clients sofort etwas sehen."""
        yield sse(self._chunk({"content": ""}))

    def feed(self, ev) -> Iterable[bytes]:
        if ev is DONE or not isinstance(ev, dict):
            return

        usage = usage_from_event(ev)
        if usage:
            self.usage = usage

        if ev.get("object") == "usage.credits":
            return

        if ev.get("error"):
            err = ev["error"]
            if not isinstance(err, dict):
                err = {"message": str(err), "type": "upstream_error"}
            yield sse({"error": err})
            return

        choices = ev.get("choices")
        if not choices:
            return  # z. B. Wissensdatenbank-Ergebnisse oder andere Meta-Events

        if ev.get("model"):
            self.model = ev["model"]

        choice = choices[0]
        delta = choice.get("delta") or choice.get("message") or {}

        if delta.get("tool_calls"):
            # Natives Tool-Calling von nele.ai: im OpenAI-Format durchreichen
            self.native_tool_calls = True
            yield from self._flush_pending()
            tool_calls = []
            for i, tc in enumerate(delta["tool_calls"]):
                tc = dict(tc)
                tc.setdefault("index", i)
                tool_calls.append(tc)
            yield sse(self._chunk({"tool_calls": tool_calls}))

        content = delta.get("content")
        if content:
            yield from self._content(content)

        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]

    def _flush_pending(self) -> Iterable[bytes]:
        if self.pending:
            text, self.pending = self.pending, ""
            self.mode = "stream"
            yield sse(self._chunk({"content": text}))

    def finish(self) -> Iterable[bytes]:
        finish_reason = self.finish_reason or "stop"

        if self.mode == "buffer":
            tool_call = extract_tool_call(self.pending)
            if tool_call:
                logger.info(f"Tool-Aufruf im Stream erkannt: {tool_call['name']}")
                self.pending = ""
                yield sse(self._chunk({
                    "tool_calls": [{
                        "index": 0,
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": tool_call,
                    }]
                }))
                finish_reason = "tool_calls"
            else:
                yield from self._flush_pending()
        else:
            yield from self._flush_pending()

        if self.native_tool_calls:
            finish_reason = "tool_calls"

        yield sse(self._chunk({}, finish_reason=finish_reason))

        if self.include_usage:
            yield sse({
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [],
                "usage": self.usage or EMPTY_USAGE,
            })

        yield sse(DONE)


# --------------------------------------------------------------------------- #
# Nicht-Stream: Events zu einer Antwort zusammenfassen
# --------------------------------------------------------------------------- #

async def aggregate_events(events: AsyncIterator[object], model: str, tools_in_prompt: bool) -> dict:
    content = ""
    finish_reason = None
    usage = None
    native_calls: Dict[int, dict] = {}

    async for ev in events:
        if ev is DONE or not isinstance(ev, dict):
            continue
        u = usage_from_event(ev)
        if u:
            usage = u
        if ev.get("error"):
            raise UpstreamStreamError(ev["error"])
        choices = ev.get("choices")
        if not choices:
            continue
        if ev.get("model"):
            model = ev["model"]
        choice = choices[0]
        delta = choice.get("delta") or choice.get("message") or {}
        content += delta.get("content") or ""
        for i, tc in enumerate(delta.get("tool_calls") or []):
            idx = tc.get("index", i)
            acc = native_calls.setdefault(idx, {
                "id": None, "type": "function", "function": {"name": "", "arguments": ""},
            })
            if tc.get("id"):
                acc["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                acc["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                args = fn["arguments"]
                acc["function"]["arguments"] += args if isinstance(args, str) else json.dumps(args)
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    message = {"role": "assistant", "content": content}
    tool_calls = []
    if native_calls:
        for idx in sorted(native_calls):
            tc = native_calls[idx]
            tc["id"] = tc["id"] or f"call_{uuid.uuid4().hex[:24]}"
            tool_calls.append(tc)
    elif tools_in_prompt:
        tc = extract_tool_call(content)
        if tc:
            logger.info(f"Tool-Aufruf in Antwort erkannt: {tc['name']}")
            tool_calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function", "function": tc})
            message["content"] = None

    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or "unknown",
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": None,
            "finish_reason": finish_reason or "stop",
        }],
        "usage": usage or EMPTY_USAGE,
    }


class UpstreamStreamError(Exception):
    def __init__(self, error):
        self.error = error if isinstance(error, dict) else {"message": str(error), "type": "upstream_error"}
        super().__init__(self.error.get("message", "upstream error"))
