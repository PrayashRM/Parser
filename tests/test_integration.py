# tests/test_integration.py
import os, pytest
from fastapi.testclient import TestClient
from main import app

client  = TestClient(app)
API_KEY = os.getenv("GEMINI_KEY_TEST")  # set in your shell, never hardcoded

@pytest.mark.skipif(not API_KEY, reason="No GEMINI_KEY_TEST set")
@pytest.mark.parametrize("fixture,min_pages", [
    ("tests/fixtures/clean.pdf",             4),
    ("tests/fixtures/math_heavy.pdf",        6),
    ("tests/fixtures/borderless_tables.pdf", 3),
    ("tests/fixtures/two_column.pdf",        8),
])
def test_parse_fixture(fixture, min_pages):
    with open(fixture, "rb") as f:
        r = client.post("/parse",
            files={"file": ("paper.pdf", f, "application/pdf")},
            headers={"X-Gemini-Key": API_KEY},
        )

    assert r.status_code == 200
    data = r.json()

    # Structure checks
    assert data["doc_id"]
    assert data["total_pages"] >= min_pages
    assert isinstance(data["elements"], list)
    assert len(data["elements"]) > 0
    assert data["full_markdown"]

    # Quality checks
    q = data["parse_quality"]
    assert q["overall_confidence"] > 0.5
    assert q["pages_total"] == data["total_pages"]

    # Every element must have required fields
    for el in data["elements"]:
        assert el["element_id"]
        assert el["content_type"] in ["text", "table", "equation", "figure"]
        assert el["page_number"] >= 1
        assert 0.0 <= el["confidence"] <= 1.0
        assert el["content"] or el["content_type"] == "figure"  # figures may be empty pre-escalation

@pytest.mark.skipif(not API_KEY, reason="No GEMINI_KEY_TEST set")
def test_math_heavy_has_equations():
    with open("tests/fixtures/math_heavy.pdf", "rb") as f:
        r = client.post("/parse",
            files={"file": ("paper.pdf", f, "application/pdf")},
            headers={"X-Gemini-Key": API_KEY},
        )
    data = r.json()
    equations = [el for el in data["elements"] if el["content_type"] == "equation"]
    assert len(equations) > 0
    for eq in equations:
        assert "$$" in eq["content"]

@pytest.mark.skipif(not API_KEY, reason="No GEMINI_KEY_TEST set")
def test_assets_saved_to_disk():
    import os
    from app.config import ASSETS_BASE_DIR
    with open("tests/fixtures/two_column.pdf", "rb") as f:
        r = client.post("/parse",
            files={"file": ("paper.pdf", f, "application/pdf")},
            headers={"X-Gemini-Key": API_KEY},
        )
    data    = r.json()
    doc_id  = data["doc_id"]
    asset_dir = os.path.join(ASSETS_BASE_DIR, doc_id, "assets")

    figures = [el for el in data["elements"] if el["content_type"] == "figure"]
    if figures:
        assert os.path.isdir(asset_dir)
        saved_files = os.listdir(asset_dir)
        assert len(saved_files) > 0
        for el in figures:
            if el.get("original") and el["original"].get("file_path"):
                full = os.path.join(ASSETS_BASE_DIR, doc_id, el["original"]["file_path"])
                assert os.path.exists(full), f"Asset missing: {full}"