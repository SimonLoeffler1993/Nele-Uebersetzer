# NeleProxy

Ein FastAPI-Proxy, der es ermöglicht, Nele.ai mit Tools zu verwenden, die eine OpenAI-kompatible API erwarten (wie n8n). Dieser Proxy übersetzt OpenAI-API-Endpunkte in das Format von Nele.ai.

## Installation

1.  **Repository klonen/öffnen**.
2.  **Abhängigkeiten installieren**:
    ```bash
    pip install -r requirements.txt
    ```

## Konfiguration

1.  Eine `.env`-Datei erstellen (von `.env.example` kopieren):
    ```bash
    cp .env.example .env
    ```
2.  `.env` bearbeiten und Nele.ai API-Key sowie optional die Base URL setzen:
    ```ini
    NELE_API_KEY=dein-tatsächlicher-api-key
    NELE_BASE_URL=https://api.aieva.io/api:v1
    ```
    *Hinweis: `NELE_API_KEY` ist hier optional, wenn er über den Request-Header bereitgestellt wird (z. B. aus n8n-Anmeldeinformationen). Die Umgebungsvariable dient als Fallback.*

## Ausführen des Proxys

Server starten:

```bash
python openai_proxy.py
```

Der Server läuft standardmäßig auf `http://localhost:8000`.

## Docker

Das fertige Image liegt auf Docker Hub unter [`simon8393/neleproxy`](https://hub.docker.com/r/simon8393/neleproxy) (amd64 und arm64):

```bash
docker run -d -p 8000:8000 simon8393/neleproxy:latest
```

Eine GitHub Action (`.github/workflows/docker-publish.yml`) baut das Image automatisch:

- Push auf `main` → Tag `latest` (plus `sha-<commit>`)
- Git-Tag `v1.2.3` → Tags `1.2.3` und `1.2`
- Pull Requests → nur bauen, nicht pushen

Benötigte Repository-Secrets: `DOCKERHUB_USERNAME` und `DOCKERHUB_TOKEN` (Docker Hub Access Token mit Read & Write).

## Verwendung mit n8n

1.  Ein **OpenAI Chat Model** Node in n8n hinzufügen.
2.  **Zugangsdaten konfigurieren**:
    - **API Key**: Dein Nele.ai API-Key.
    - **Base URL**: `http://localhost:8000/v1` (oder die IP deiner Maschine, wenn in Docker ausgeführt).
3.  **Modell auswählen**: Verwende ein Modell von Nele.ai (z. B. `azure-o4-mini`).

Der Proxy fängt die Anfrage ab, leitet sie an Nele.ai weiter und gibt die Antwort im OpenAI-Format zurück.
## Streaming

Der Proxy verhält sich wie die OpenAI-API: Mit `"stream": true` kommen die Tokens als Server-Sent Events (`chat.completion.chunk`), sobald nele.ai sie liefert, am Ende folgt `data: [DONE]`. Ohne `stream` gibt es wie bisher eine vollständige `chat.completion`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dein-nele-api-key")
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Übersetze 'Guten Morgen' ins Englische."}],
    stream=True,
    stream_options={"include_usage": True},  # optional: Usage-Chunk am Ende
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Oder mit curl:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $NELE_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"Hallo"}]}'
```

Hinweise:

- Fehler von nele.ai (z. B. 401, 402, 422) kommen als echter HTTP-Status im OpenAI-Fehlerformat zurück, bevor der Stream startet.
- Hinter nginx o. Ä. setzt der Proxy `X-Accel-Buffering: no`, damit nicht gepuffert wird.
- Unterstützte Endpunkte: `POST /v1/chat/completions` und `GET /v1/models` (jeweils auch ohne `/v1`).

## Dateien und Bilder

Dateien werden im OpenAI-Format als Content-Parts mitgeschickt und als `attachments` an nele.ai weitergegeben:

| Typ | Verarbeitung |
| --- | --- |
| Bilder `png`, `jpg`, `webp` (`image_url`, als data:-URL oder https-Link) | werden über `POST /image-attachment` zu nele.ai hochgeladen und angehängt (gleiche Bilder nur einmal, solange der Upload gültig ist) |
| Textdateien (txt, md, csv, json, …) | als Text-Attachment |
| PDF, Word (`.docx`) | Text wird im Proxy ausgelesen und als Text-Attachment angehängt (gescannte PDFs ohne Text gehen nicht) |
| andere Bildformate / Dateitypen | Fehler 415 |

```python
import base64
pdf = base64.b64encode(open("vertrag.pdf", "rb").read()).decode()
client.chat.completions.create(
    model="gpt-4o",
    stream=True,
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Übersetze dieses Dokument ins Englische."},
        {"type": "file", "file": {"filename": "vertrag.pdf", "file_data": f"data:application/pdf;base64,{pdf}"}},
    ]}],
)
```

Maximale Größe pro Datei: 25 MB. `file_id` aus der OpenAI-Files-API wird nicht unterstützt.

## Parameter

| OpenAI-Feld | an nele.ai |
| --- | --- |
| `model`, `messages` | 1:1 (Rollen `developer` → `system`, `tool` → `user`; Content-Listen werden zu Text) |
| `max_tokens` / `max_completion_tokens` | `max_tokens` |
| `temperature` | `temperature` (auf 0–1 begrenzt, nele.ai erlaubt nicht mehr) |
| `reasoning_effort` | `modelConfiguration.reasoning_effort` |
| `tools`, `tool_choice` | siehe Tool-Modus |

nele.ai-eigene Felder können über `extra_body` mitgeschickt werden und werden durchgereicht: `documentCollectionId`, `documentCollectionOptions`, `web_search`, `modelConfiguration`.

## Weitere Einstellungen (`.env`)

| Variable | Standard | Bedeutung |
| --- | --- | --- |
| `NELE_TOOL_MODE` | `prompt` | `prompt`: Tools werden in den System-Prompt geschrieben und der JSON-Aufruf aus der Antwort gelesen (bisheriges Verhalten). `native`: `tools`/`tool_choice` gehen direkt an nele.ai. |
| `NELE_READ_TIMEOUT` | `300` | Sekunden, die der Proxy auf Daten von nele.ai wartet. |
| `LOG_LEVEL` | `INFO` | `DEBUG` loggt auch Request-Bodies (API-Key wird geschwärzt). |
| `LOG_FILE` | `proxy_debug.log` | Leer lassen, um nicht in eine Datei zu loggen. |

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

Die Tests laufen gegen einen simulierten nele.ai-Server und nutzen das offizielle `openai`-Python-SDK.
