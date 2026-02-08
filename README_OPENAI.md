# Nele.ai OpenAI Proxy - Setup Guide

## Overview
This proxy allows n8n (and other OpenAI-compatible tools) to use Nele.ai models by translating OpenAI API endpoints to Nele.ai's format.

## Why Use This Proxy?
- **Simple**: Only does URL rewriting, no complex format conversion
- **Compatible**: Works with any OpenAI-compatible client (n8n, LangChain, etc.)
- **Transparent**: Nele.ai already speaks OpenAI protocol, proxy just fixes the endpoint paths

## Setup

### 1. Start the Proxy
```bash
python openai_proxy.py
```

The proxy will run on **http://0.0.0.0:8000**

### 2. Configure n8n

In your n8n workflow:

1. **Use the "OpenAI Chat Model" node** (NOT Ollama!)
2. **Configure the credentials:**
   - **API Key**: Your Nele.ai API key (`YOUR_NELE_API_KEY_HERE`)
   - **Base URL**: `http://192.168.188.20:8000/v1`
3. **Select Model**: `azure-o4-mini` (or any other Nele.ai model)

That's it! Your n8n workflow will now use Nele.ai models through the proxy.

## How It Works

### Request Flow
```
n8n → POST /v1/chat/completions → Proxy → POST /api:v1/chat-completion → Nele.ai
```

### Streaming vs Non-Streaming
- **Streaming (`stream: true`)**: Proxy passes through Nele.ai's stream directly
- **Non-Streaming (`stream: false`)**: Proxy aggregates the stream into a single OpenAI response

### Supported Endpoints
- `/v1/chat/completions` → `https://api.aieva.io/api:v1/chat-completion`
- `/v1/models` → `https://api.aieva.io/api:v1/models`

## Advantages Over Ollama Proxy
- ✅ No format conversion needed (Nele.ai is already OpenAI-compatible)
- ✅ Simpler code (~130 lines vs ~350 lines)
- ✅ No special handling for `usage.credits` chunks
- ✅ Works with all OpenAI-compatible tools
- ✅ Native n8n support (better integration, more features)

## Troubleshooting

### Proxy not accessible from n8n
Make sure the proxy is running and accessible from your n8n server:
```bash
curl http://192.168.188.20:8000/v1/models
```

### Wrong endpoint error
Ensure you're using the **OpenAI Chat Model** node in n8n, not the Ollama node.

### Authentication errors
Verify your Nele.ai API key is correct in `.env` or hardcoded in `openai_proxy.py`.
