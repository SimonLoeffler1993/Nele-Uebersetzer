import json
import time
import re

def extract_json_tool_call(content: str) -> dict:
    if not content:
        return None
    content_trimmed = content.strip()
    if content_trimmed.startswith("{") and content_trimmed.endswith("}"):
        try:
            data = json.loads(content_trimmed)
            if "name" in data and "arguments" in data:
                return data
        except:
            pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_trimmed, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "name" in data and "arguments" in data:
                return data
        except:
            pass
    match = re.search(r"(\{.*\})", content_trimmed, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "name" in data and "arguments" in data:
                return data
        except:
            pass
    return None

def simulate_streaming_transformation(chunks_content):
    full_content_buffer = ""
    is_buffering = False
    
    # Process chunks like the proxy does
    for content in chunks_content:
        test_buffer = (full_content_buffer + content).strip()
        if not is_buffering:
            if test_buffer.startswith("{") or test_buffer.startswith("```"):
                is_buffering = True
                print("DEBUG: Stream entered buffer mode")
        
        if is_buffering:
            full_content_buffer += content
        else:
            # Yielding normally
            pass
            
    # At the end
    if is_buffering:
        tool_call_data = extract_json_tool_call(full_content_buffer)
        if tool_call_data:
            print(f"DEBUG: Detected tool call: {tool_call_data['name']}")
            arguments = tool_call_data["arguments"]
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            
            # Synthetic OpenAI chunk
            return {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"name": tool_call_data["name"], "arguments": arguments}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }
    return None

# Test Streaming Case
streaming_chunks = ['{"name": ', '"Skiservice abfragen", ', '"arguments": ', '{"query": "Ski"}}']
print("\n--- Streaming Simulation ---")
result = simulate_streaming_transformation(streaming_chunks)
if result:
    print(json.dumps(result, indent=2))
else:
    print("FAILED: No tool call detected in stream.")

# Test Streaming Markdown Case
streaming_chunks_md = ['```json\n', '{"name": "Wetter", ', '"arguments": {"city": "Berlin"}}', '\n```']
print("\n--- Streaming Markdown Simulation ---")
result = simulate_streaming_transformation(streaming_chunks_md)
if result:
    print(json.dumps(result, indent=2))
else:
    print("FAILED: No tool call detected in stream.")
