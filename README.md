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

## Verwendung mit n8n

1.  Ein **OpenAI Chat Model** Node in n8n hinzufügen.
2.  **Zugangsdaten konfigurieren**:
    - **API Key**: Dein Nele.ai API-Key.
    - **Base URL**: `http://localhost:8000/v1` (oder die IP deiner Maschine, wenn in Docker ausgeführt).
3.  **Modell auswählen**: Verwende ein Modell von Nele.ai (z. B. `azure-o4-mini`).

Der Proxy fängt die Anfrage ab, leitet sie an Nele.ai weiter und gibt die Antwort im OpenAI-Format zurück.