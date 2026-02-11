import json
import re

def extract_tool_call(content: str) -> dict:
    if not content:
        return None
        
    content_trimmed = content.strip()
    
    def parse_if_tool(data_str):
        try:
            data = json.loads(data_str)
            name = data.get("id") or data.get("name")
            if name and "arguments" in data:
                args = data["arguments"]
                if not isinstance(args, str):
                    args = json.dumps(args)
                return {"name": name, "arguments": args}
        except:
            pass
        return None

    if content_trimmed.startswith("{") and content_trimmed.endswith("}"):
        res = parse_if_tool(content_trimmed)
        if res: return res

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_trimmed, re.DOTALL)
    if match:
        res = parse_if_tool(match.group(1))
        if res: return res
        
    match = re.search(r"(\{.*\})", content_trimmed, re.DOTALL)
    if match:
        res = parse_if_tool(match.group(1))
        if res: return res
        
    return None

# Test cases based on user report and typical LLM behaviors
tests = [
    {
        "name": "Raw JSON with id (User case)",
        "content": '{ "id": "Skiservice abfragen", "arguments": {} }',
        "expected": "Skiservice abfragen"
    },
    {
        "name": "Markdown wrapped JSON",
        "content": 'Hier ist das Tool:\n```json\n{"name": "Wetter", "arguments": {"stadt": "Berlin"}}\n```',
        "expected": "Wetter"
    },
    {
        "name": "JSON with arguments as object",
        "content": '{"id": "Search", "arguments": {"q": "test"}}',
        "expected": "Search"
    }
]

for t in tests:
    result = extract_tool_call(t["content"])
    if result and result["name"] == t["expected"]:
        print(f"PASS: {t['name']}")
        print(f"  Result: {result}")
    else:
        print(f"FAIL: {t['name']}")
        print(f"  Got: {result}")
