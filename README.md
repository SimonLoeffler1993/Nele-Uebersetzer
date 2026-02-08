# NeleProxy

A FastAPI proxy that allows using Nele.ai with tools that expect an Ollama-compatible API (like n8n).

## Installation

1.  **Clone/Open** the repository.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  Create a `.env` file (copy from `.env.example`):
    ```bash
    cp .env.example .env
    ```
2.  Edit `.env` and set your Nele.ai API Key:
    ```ini
    NELE_API_KEY=your-actual-api-key
    ```
    *Note: Verify `NELE_API_URL` matches the correct Nele.ai endpoint.*

## Running the Proxy

Start the server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 11434 --reload
```

The server will listen on `http://localhost:11434`.

## Using with n8n

1.  Add an **Ollama Chat Model** node (or similar) in n8n.
2.  Set the **Base URL** to `http://localhost:11434` (or your machine's IP if running in Docker/Container).
3.  The proxy will intercept the request, forward it to Nele.ai, and return the response in Ollama format.
