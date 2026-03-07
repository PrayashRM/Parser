from app.pipeline.assembly import run_assembly, serialize_to_markdown
from app.models import (
    DocumentModel, PageRecord, ParsedElement,
    ContentType, ContentSource
)

def make_doc_with_pages():
    doc = DocumentModel.create(b"%PDF-fake")
    doc.total_pages = 2

    el1 = ParsedElement(
        element_id="p1_text_0", content_type=ContentType.TEXT,
        content="Introduction text.", page_number=1,
        confidence=0.9, source=ContentSource.PYMUPDF,
        section_path=["Introduction"],
    )
    el2 = ParsedElement(
        element_id="p1_table_0", content_type=ContentType.TABLE,
        content="| A | B |\n|---|---|\n| 1 | 2 |", page_number=1,
        confidence=0.92, source=ContentSource.GEMINI,
        section_path=["Introduction"],
    )
    doc.pages[1] = PageRecord(
        page_number=1, confidence=0.9,
        source=ContentSource.PYMUPDF, elements=[el2, el1],  # wrong order intentionally
    )
    return doc

def test_element_ordering():
    doc = make_doc_with_pages()
    doc = run_assembly(doc)
    types = [el.content_type for el in doc.pages[1].elements]
    assert types.index(ContentType.TEXT) < types.index(ContentType.TABLE)

def test_hybrid_source_assigned():
    doc = make_doc_with_pages()
    doc = run_assembly(doc)
    assert doc.pages[1].source == ContentSource.HYBRID

def test_empty_figures_removed():
    from app.models import EscalationReason
    doc = make_doc_with_pages()
    doc.pages[1].elements.append(ParsedElement(
        element_id="p1_figure_0", content_type=ContentType.FIGURE,
        content="",  # never described — should be removed
        page_number=1, confidence=0.5, source=ContentSource.PYMUPDF,
        section_path=[],
    ))
    doc = run_assembly(doc)
    figures = [el for el in doc.pages[1].elements if el.content_type == ContentType.FIGURE]
    assert figures == []

def test_markdown_serialisation():
    doc = make_doc_with_pages()
    doc = run_assembly(doc)
    md  = serialize_to_markdown(doc)
    assert "| A | B |" in md
    assert "Introduction text." in md