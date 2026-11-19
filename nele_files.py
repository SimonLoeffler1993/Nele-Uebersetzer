"""
Dateien und Bilder aus OpenAI-Nachrichten an nele.ai durchreichen.

OpenAI-Clients schicken Dateien als Content-Parts:
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..." | "https://..."}}
  {"type": "file", "file": {"file_data": "data:application/pdf;base64,...", "filename": "a.pdf"}}
  {"type": "input_file", "file_data": "...", "filename": "..."}   (Responses-Stil)

nele.ai kennt pro Nachricht `attachments` (Text oder vorher hochgeladenes Bild):
  - Bilder (png/jpg/webp) -> POST /image-attachment (multipart) -> `path`
  - Textdateien           -> Text-Attachment
  - PDF / Word (docx)     -> Text wird hier extrahiert -> Text-Attachment
"""

import base64
import binascii
import hashlib
import io
import logging
import mimetypes
import time
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("NeleOpenAIProxy")

NELE_IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp"}

TEXT_EXTENSIONS = {
    "txt", "md", "markdown", "csv", "tsv", "json", "jsonl", "xml", "html", "htm", "yaml", "yml",
    "log", "ini", "toml", "py", "js", "ts", "java", "c", "cpp", "cs", "go", "rs", "sql", "sh", "rtf",
}
PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# Obergrenze für eine einzelne Datei (Bytes), schützt den Proxy vor riesigen Uploads
MAX_FILE_BYTES = 25 * 1024 * 1024


class AttachmentError(Exception):
    """Datei kann nicht verarbeitet werden -> 4xx an den Client."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


# --------------------------------------------------------------------------- #
# Format der nele.ai-Attachments
# (an einer Stelle gebündelt, damit es leicht an die Doku angepasst werden kann)
# --------------------------------------------------------------------------- #

# Laut nele.ai-Doku: {type: text|image, id, name, content, detail?: high|low}
# Bei Bildern ist `content` der `path` aus POST /image-attachment.

def _attachment_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def make_text_attachment(name: str, text: str) -> dict:
    return {"type": "text", "id": _attachment_id(name.encode() + b"\0" + text.encode()), "name": name, "content": text}


def make_image_attachment(name: str, path: str, detail: Optional[str] = None) -> dict:
    att = {"type": "image", "id": _attachment_id(path.encode()), "name": name, "content": path}
    if detail in ("high", "low"):
        att["detail"] = detail
    return att


# --------------------------------------------------------------------------- #
# Dateien dekodieren
# --------------------------------------------------------------------------- #

def decode_data_url(url: str) -> Tuple[bytes, Optional[str]]:
    """data:<mime>;base64,<daten> -> (bytes, mime)"""
    if not url.startswith("data:"):
        raise AttachmentError("Nur data:-URLs können dekodiert werden.")
    header, _, data = url.partition(",")
    mime = header[5:].split(";")[0] or None
    try:
        if ";base64" in header:
            raw = base64.b64decode(data, validate=False)
        else:
            from urllib.parse import unquote_to_bytes
            raw = unquote_to_bytes(data)
    except (binascii.Error, ValueError) as e:
        raise AttachmentError(f"Datei konnte nicht dekodiert werden: {e}")
    return raw, mime


def decode_file_data(file_data: str) -> Tuple[bytes, Optional[str]]:
    """`file_data` ist entweder eine data:-URL oder reines base64."""
    if file_data.startswith("data:"):
        return decode_data_url(file_data)
    try:
        return base64.b64decode(file_data, validate=False), None
    except (binascii.Error, ValueError) as e:
        raise AttachmentError(f"Datei konnte nicht dekodiert werden: {e}")


def sniff_mime(raw: bytes, mime: Optional[str], filename: Optional[str]) -> str:
    if raw.startswith(b"\x89PNG"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"%PDF"):
        return PDF_MIME
    if mime and mime != "application/octet-stream":
        return mime
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed
        if filename.lower().endswith(".docx"):
            return DOCX_MIME
    return mime or "application/octet-stream"


def is_text_type(mime: str, filename: Optional[str]) -> bool:
    if mime.startswith("text/") or mime in ("application/json", "application/xml", "application/x-yaml"):
        return True
    ext = (filename or "").rsplit(".", 1)[-1].lower() if filename and "." in filename else ""
    return ext in TEXT_EXTENSIONS


def extract_pdf_text(raw: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"--- Seite {i} ---\n{text}")
    return "\n\n".join(pages)


def extract_docx_text(raw: bytes) -> str:
    import docx
    document = docx.Document(io.BytesIO(raw))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)


def decode_text(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Bild-Upload zu nele.ai (mit Cache, damit der Verlauf nicht bei jedem
# Request neu hochgeladen wird)
# --------------------------------------------------------------------------- #

Uploader = Callable[[bytes, str, str], Awaitable[str]]


class ImageUploader:
    def __init__(self, base_url: str, api_key: str, client: httpx.AsyncClient):
        self.base_url = base_url
        self.api_key = api_key
        self.client = client

    # Cache: (api_key-hash, bild-hash) -> (path, expires_at)
    _cache: Dict[Tuple[str, str], Tuple[str, int]] = {}

    async def __call__(self, raw: bytes, mime: str, filename: str) -> str:
        key = (hashlib.sha256((self.api_key or "").encode()).hexdigest(), hashlib.sha256(raw).hexdigest())
        cached = self._cache.get(key)
        if cached and cached[1] - 60 > time.time():
            return cached[0]

        response = await self.client.post(
            f"{self.base_url}/image-attachment",
            headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
            files={"file": (filename, raw, mime)},
        )
        if response.status_code != 200:
            raise AttachmentError(
                f"nele.ai hat das Bild '{filename}' abgelehnt ({response.status_code}): {response.text[:300]}",
                status_code=response.status_code if response.status_code in (401, 403, 415, 422) else 502,
            )
        data = response.json()
        path = data["path"]
        expires_at = int(data.get("expires_at") or 0)
        # expires_at kann Sekunden oder Millisekunden sein
        if expires_at > 10**12:
            expires_at //= 1000
        self._cache[key] = (path, expires_at or int(time.time()) + 3600)
        return path


# --------------------------------------------------------------------------- #
# Nachrichten umbauen
# --------------------------------------------------------------------------- #

async def _load_image_url(url: str, client: httpx.AsyncClient) -> Tuple[bytes, Optional[str]]:
    if url.startswith("data:"):
        return decode_data_url(url)
    if url.startswith("http://") or url.startswith("https://"):
        r = await client.get(url, follow_redirects=True, timeout=30.0)
        if r.status_code != 200:
            raise AttachmentError(f"Bild konnte nicht geladen werden ({r.status_code}): {url}")
        return r.content, r.headers.get("content-type", "").split(";")[0] or None
    raise AttachmentError(f"Nicht unterstützte Bild-URL: {url[:80]}")


async def part_to_attachment(part: dict, index: int, upload: Uploader, client: httpx.AsyncClient) -> Optional[dict]:
    ptype = part.get("type")
    filename = None
    detail = None

    if ptype == "image_url":
        img = part.get("image_url")
        url = img.get("url") if isinstance(img, dict) else img
        detail = img.get("detail") if isinstance(img, dict) else None
        raw, mime = await _load_image_url(url or "", client)
    elif ptype in ("file", "input_file"):
        f = part.get("file") if ptype == "file" else part
        f = f or {}
        filename = f.get("filename")
        if f.get("file_data"):
            raw, mime = decode_file_data(f["file_data"])
        elif f.get("file_url"):
            raw, mime = await _load_image_url(f["file_url"], client)
        elif f.get("file_id"):
            raise AttachmentError("`file_id` (OpenAI Files API) wird nicht unterstützt. Bitte die Datei als `file_data` (base64) mitschicken.")
        else:
            raise AttachmentError("Datei-Part ohne Inhalt.")
    elif ptype in ("input_image",):
        raw, mime = await _load_image_url(part.get("image_url") or "", client)
        detail = part.get("detail")
    else:
        logger.warning(f"Content-Part vom Typ '{ptype}' wird ignoriert.")
        return None

    if len(raw) > MAX_FILE_BYTES:
        raise AttachmentError(f"Datei '{filename or index}' ist zu groß (max. {MAX_FILE_BYTES // (1024 * 1024)} MB).", 413)

    mime = sniff_mime(raw, mime, filename)

    if mime in NELE_IMAGE_TYPES:
        name = filename or f"bild-{index}.{NELE_IMAGE_TYPES[mime]}"
        path = await upload(raw, mime, name)
        return make_image_attachment(name, path, detail)
    if mime.startswith("image/"):
        raise AttachmentError(f"Bildformat {mime} wird von nele.ai nicht unterstützt (nur png, jpg, webp).", 415)

    name = filename or f"datei-{index}"
    if mime == PDF_MIME:
        text = extract_pdf_text(raw)
        if not text.strip():
            raise AttachmentError(f"Aus '{name}' konnte kein Text gelesen werden (gescanntes PDF?).", 422)
    elif mime == DOCX_MIME or name.lower().endswith(".docx"):
        text = extract_docx_text(raw)
    elif is_text_type(mime, filename):
        text = decode_text(raw)
    else:
        raise AttachmentError(f"Dateityp {mime} ('{name}') wird nicht unterstützt.", 415)
    return make_text_attachment(name, text)


def has_file_parts(messages: List[dict]) -> bool:
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and p.get("type") not in ("text", "input_text"):
                    return True
    return False


async def extract_attachments(messages: List[dict], upload: Uploader, client: httpx.AsyncClient) -> List[dict]:
    """
    Wandelt Datei-/Bild-Parts in nele.ai-`attachments` um.
    Text-Parts bleiben im `content`, bereits vorhandene `attachments` bleiben erhalten.
    """
    result = []
    counter = 0
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            result.append(m)
            continue
        texts, attachments = [], list(m.get("attachments") or [])
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                texts.append(part.get("text", ""))
            elif isinstance(part, dict):
                counter += 1
                att = await part_to_attachment(part, counter, upload, client)
                if att:
                    attachments.append(att)
        new_m = {**m, "content": "\n".join(t for t in texts if t)}
        if attachments:
            new_m["attachments"] = attachments
        result.append(new_m)
    return result
