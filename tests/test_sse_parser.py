import asyncio

from nele_stream import DONE, StreamTranslator, parse_sse_lines


async def _lines(items):
    for i in items:
        yield i


def parse(lines):
    async def run():
        return [e async for e in parse_sse_lines(_lines(lines))]
    return asyncio.run(run())


def test_parser_handles_blank_line_separated_events():
    evs = parse(['data: {"a": 1}', "", ": keep-alive", "", "data: [DONE]", ""])
    assert evs == [{"a": 1}, DONE]


def test_parser_handles_events_without_blank_lines():
    evs = parse(['data: {"a": 1}', 'data: {"b": 2}', "data: [DONE]"])
    assert evs == [{"a": 1}, {"b": 2}, DONE]


def test_translator_whitespace_before_text_is_kept():
    t = StreamTranslator("m", tools_in_prompt=True)
    out = b"".join(t.feed({"choices": [{"delta": {"content": "\n"}}]}))
    assert out == b""
    out = b"".join(t.feed({"choices": [{"delta": {"content": "Hallo"}}]}))
    assert b'"content": "\\nHallo"' in out


def test_translator_markdown_answer_that_is_not_a_tool_is_flushed():
    t = StreamTranslator("m", tools_in_prompt=True)
    list(t.feed({"choices": [{"delta": {"content": "```python\nprint(1)\n```"}}]}))
    out = b"".join(t.finish())
    assert b"print(1)" in out
    assert b'"finish_reason": "stop"' in out
