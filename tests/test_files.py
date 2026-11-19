import base64
import io

import pytest
from openai import OpenAI

from conftest import nele_chunk

# 1x1 PNG
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def minimal_pdf(text: str) -> bytes:
    """Baut ein winziges PDF mit einer Textzeile."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + o + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return out.getvalue()


def docx_bytes(text: str) -> bytes:
    import docx
    d = docx.Document()
    d.add_paragraph(text)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def ask(proxy_url, parts, stream=False):
    client = OpenAI(base_url=proxy_url, api_key="test-key")
    return client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": parts}], stream=stream
    )


@pytest.fixture
def ok(nele):
    nele.script = {"status": 200, "events": [nele_chunk("ok"), nele_chunk(finish_reason="stop"), "[DONE]"],
                   "delay": 0.01}
    return nele


def user_msg(nele):
    return nele.requests[-1]["body"]["messages"][-1]


def test_image_is_uploaded_and_attached(ok, proxy_url):
    ask(proxy_url, [{"type": "text", "text": "Was ist das?"},
                    {"type": "image_url", "image_url": {"url": data_url(PNG, "image/png")}}])
    assert ok.uploads[0]["content_type"] == "image/png"
    msg = user_msg(ok)
    assert msg["content"] == "Was ist das?"
    assert msg["attachments"][0]["type"] == "image"
    assert msg["attachments"][0]["path"].startswith("attachments/1-")


def test_same_image_is_uploaded_only_once(ok, proxy_url):
    img = data_url(PNG + b"cache", "image/png")
    ask(proxy_url, [{"type": "image_url", "image_url": {"url": img}}])
    ask(proxy_url, [{"type": "image_url", "image_url": {"url": img}}])
    assert len(ok.uploads) == 1


def test_text_file_becomes_text_attachment(ok, proxy_url):
    ask(proxy_url, [{"type": "text", "text": "Übersetze"},
                    {"type": "file", "file": {"filename": "notiz.txt",
                                              "file_data": data_url("Grüß Gott".encode(), "text/plain")}}])
    att = user_msg(ok)["attachments"][0]
    assert att == {"type": "text", "name": "notiz.txt", "content": "Grüß Gott"}


def test_pdf_text_is_extracted(ok, proxy_url):
    ask(proxy_url, [{"type": "file", "file": {"filename": "a.pdf",
                                              "file_data": data_url(minimal_pdf("Hallo PDF"), "application/pdf")}}],
        stream=True).close()
    att = user_msg(ok)["attachments"][0]
    assert att["type"] == "text" and "Hallo PDF" in att["content"]


def test_docx_text_is_extracted(ok, proxy_url):
    raw = docx_bytes("Hallo Word")
    # reines base64 ohne data:-Präfix
    ask(proxy_url, [{"type": "file", "file": {"filename": "b.docx", "file_data": base64.b64encode(raw).decode()}}])
    att = user_msg(ok)["attachments"][0]
    assert att["content"] == "Hallo Word"


def test_unsupported_image_type_is_rejected(ok, proxy_url):
    with pytest.raises(Exception) as exc:
        ask(proxy_url, [{"type": "image_url", "image_url": {"url": data_url(b"GIF89a....", "image/gif")}}])
    assert getattr(exc.value, "status_code", None) == 415
    assert not ok.requests
