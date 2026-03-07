from app.pipeline.sniffer import _score_page
import pymupdf

def make_raw_page():
    # Minimal blank pymupdf page for signal 6
    doc  = pymupdf.open()
    page = doc.new_page()
    return page

def test_clean_page_high_confidence():
    markdown = "This is a normal paragraph with regular text content about machine learning."
    page     = make_raw_page()
    conf, signals, reasons = _score_page(markdown, page, mock_page_record())
    assert conf >= 0.65
    assert reasons == []

def test_broken_math_low_confidence():
    markdown = "x 2 + y 2 = r 2 where α β γ are parameters of the equation theorem lemma"
    page     = make_raw_page()
    conf, signals, reasons = _score_page(markdown, page, mock_page_record())
    assert conf < 0.65
    assert any("math" in r.value or "greek" in r.value for r in reasons)

def test_borderless_table_detected():
    markdown = """
    Model       BLEU    Params
    Transformer  28.4   65M
    LSTM         24.1   44M
    CNN          22.0   38M
    """
    page = make_raw_page()
    conf, signals, reasons = _score_page(markdown, page, mock_page_record())
    assert signals.borderless_table is True

def mock_page_record():
    from app.models import PageRecord, ContentSource
    return PageRecord(page_number=1, confidence=0.9, source=ContentSource.PYMUPDF)