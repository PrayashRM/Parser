"""
tests/test_gateway.py
=====================
Unit tests  — no real PDF, no API key, no network calls.
Integration — set GEMINI_KEY_TEST env var + drop a real PDF at tests/fixtures/real.pdf
              then run: GEMINI_KEY_TEST=your-key pytest tests/test_gateway.py -v
"""
import io
import os

import pytest
from fastapi.testclient import TestClient

from main import app

client  = TestClient(app)
API_KEY = os.getenv("GEMINI_KEY_TEST")   # only needed for integration tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pdf(body: bytes = b"") -> io.BytesIO:
    """Minimal valid-looking PDF byte stream."""
    return io.BytesIO(b"%PDF-1.4 " + body)

def _post(file_bytes: io.BytesIO, key: str = "fake-key", filename: str = "paper.pdf"):
    return client.post(
        "/parse",
        files={"file": (filename, file_bytes, "application/pdf")},
        headers={"X-Gemini-Key": key},
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Magic byte validation
# ---------------------------------------------------------------------------

def test_rejects_file_with_wrong_magic_bytes():
    fake_zip = io.BytesIO(b"PK\x03\x04 this is a zip")
    r = _post(fake_zip)
    assert r.status_code == 400
    assert "magic bytes" in r.json()["detail"].lower()

def test_rejects_plain_text_file():
    txt = io.BytesIO(b"hello world, not a pdf")
    r   = _post(txt)
    assert r.status_code == 400

def test_rejects_jpeg_renamed_as_pdf():
    # JPEG magic: FF D8 FF
    jpeg = io.BytesIO(b"\xff\xd8\xff\xe0 fake jpeg content")
    r    = _post(jpeg)
    assert r.status_code == 400

def test_rejects_empty_file():
    r = _post(io.BytesIO(b""))
    assert r.status_code == 400

def test_accepts_correct_magic_bytes_prefix():
    # Gateway will pass magic + size + MIME checks, then fail at Gemini key
    # (fake key → 401). That means the PDF itself passed validation — correct.
    r = _post(_pdf())
    # 401 means gateway let it through; anything 4xx from PDF checks = bug
    assert r.status_code in (400, 401, 422, 500)
    if r.status_code == 400:
        assert "magic" not in r.json().get("detail", "").lower()


# ---------------------------------------------------------------------------
# Size gate
# ---------------------------------------------------------------------------

def test_rejects_file_over_50mb():
    big = io.BytesIO(b"%PDF" + b"x" * (51 * 1024 * 1024))
    r   = _post(big)
    assert r.status_code == 413

def test_accepts_file_just_under_50mb():
    # Should pass size gate (will fail later on MIME or Gemini key — that's fine)
    under = io.BytesIO(b"%PDF" + b"x" * (49 * 1024 * 1024))
    r     = _post(under)
    assert r.status_code != 413

def test_rejects_file_exactly_at_limit():
    limit = io.BytesIO(b"%PDF" + b"x" * (50 * 1024 * 1024))
    r     = _post(limit)
    assert r.status_code == 413


# ---------------------------------------------------------------------------
# Missing / malformed headers
# ---------------------------------------------------------------------------

def test_rejects_missing_gemini_key_header():
    r = client.post(
        "/parse",
        files={"file": ("paper.pdf", _pdf(), "application/pdf")},
        # No X-Gemini-Key header
    )
    assert r.status_code == 422   # FastAPI: required header missing

def test_rejects_empty_gemini_key():
    r = _post(_pdf(), key="")
    # Empty string header — FastAPI may pass it; gateway probe should reject
    assert r.status_code in (401, 422)

def test_rejects_clearly_invalid_key():
    r = _post(_pdf(), key="not-a-real-key-abc123")
    assert r.status_code == 401
    assert "gemini" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# File field validation
# ---------------------------------------------------------------------------

def test_rejects_missing_file_field():
    r = client.post(
        "/parse",
        headers={"X-Gemini-Key": "fake-key"},
        # No file field at all
    )
    assert r.status_code == 422

def test_handles_unusual_filename_gracefully():
    # Filename doesn't matter — validation is on bytes, not extension
    r = _post(_pdf(), filename="thesis_final_FINAL_v3.PDF")
    assert r.status_code in (400, 401, 422, 500)
    assert r.status_code != 413


# ---------------------------------------------------------------------------
# Integration — only runs when GEMINI_KEY_TEST is set + fixture exists
# ---------------------------------------------------------------------------

FIXTURE = "tests/fixtures/real.pdf"
HAS_KEY = bool(API_KEY)
HAS_PDF = os.path.exists(FIXTURE)

@pytest.mark.skipif(not HAS_KEY, reason="Set GEMINI_KEY_TEST to run integration tests")
@pytest.mark.skipif(not HAS_PDF, reason=f"Drop a real PDF at {FIXTURE} to run integration tests")
def test_real_pdf_parses_successfully():
    with open(FIXTURE, "rb") as f:
        r = client.post(
            "/parse",
            files={"file": ("real.pdf", f, "application/pdf")},
            headers={"X-Gemini-Key": API_KEY},
        )

    assert r.status_code == 200, f"Parse failed: {r.text[:500]}"
    data = r.json()

    # Structure
    assert data["doc_id"]
    assert data["total_pages"] >= 1
    assert isinstance(data["elements"], list)
    assert len(data["elements"]) > 0
    assert data["full_markdown"]

    # Quality
    q = data["parse_quality"]
    assert q["overall_confidence"] > 0.0
    assert q["pages_total"] == data["total_pages"]
    assert q["pages_local"] + q["pages_escalated"] + q["pages_hybrid"] == q["pages_total"]

    # Every element has required fields and valid values
    for el in data["elements"]:
        assert el["element_id"],                        "element_id missing"
        assert el["content_type"] in ["text", "table", "equation", "figure"]
        assert el["page_number"] >= 1,                  "page_number invalid"
        assert 0.0 <= el["confidence"] <= 1.0,          "confidence out of range"
        assert el["source"] in ["pymupdf", "gemini", "hybrid"]

    print("\n--- Parse Quality Report ---")
    import json
    print(json.dumps(data["parse_quality"], indent=2))
    print(f"Elements: {len(data['elements'])} total")
    print(f"Tables:    {q['tables_found']}")
    print(f"Equations: {q['equations_found']}")
    print(f"Figures:   {q['figures_found']}")