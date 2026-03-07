# build_final_response(), ParseResponse Pydantic model, parse quality report

"""
output_schema.py — Phase 07: Structured Output Schema
Builds the final ParseResponse. Also runs lightweight metadata extraction
(title, authors, abstract, DOI, year) from first-page text if not already set.
"""
import re
from typing import Optional

from app.models import ContentType, DocumentModel, ParseResponse


def build_final_response(doc: DocumentModel, full_markdown: str) -> ParseResponse:
    _extract_metadata(doc)
    return ParseResponse.from_document(doc, full_markdown)


# ---------------------------------------------------------------------------
# Lightweight metadata extraction — no API cost, best-effort, first page only
# ---------------------------------------------------------------------------

def _extract_metadata(doc: DocumentModel) -> None:
    if not doc.pages:
        return
    first_page = doc.pages.get(1) or doc.pages.get(min(doc.pages))
    text_els   = first_page.by_type(ContentType.TEXT)
    if not text_els:
        return
    text = text_els[0].content

    if doc.title    is None:  doc.title    = _extract_title(text)
    if not doc.authors:       doc.authors  = _extract_authors(text)
    if doc.abstract is None:  doc.abstract = _extract_abstract(text)
    if doc.doi      is None:  doc.doi      = _extract_doi(text)
    if doc.year     is None:  doc.year     = _extract_year(text)


def _extract_title(text: str) -> Optional[str]:
    match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[0] if lines else None


def _extract_authors(text: str) -> list:
    abstract_pos   = re.search(r'\babstract\b', text, re.IGNORECASE)
    candidate_text = text[:abstract_pos.start()] if abstract_pos else text[:500]
    lines = [l.strip() for l in candidate_text.splitlines()
             if l.strip() and not l.startswith('#')]
    authors = []
    for line in lines[:5]:
        if re.search(r'\d{4}|university|institute|department|@|\{', line, re.IGNORECASE):
            continue
        if re.match(r'^[A-Z][a-z]+([ ,\-]+[A-Z][a-z]*\.?)+', line):
            authors.extend(p.strip() for p in re.split(r',\s*|\s+and\s+', line) if p.strip())
    return authors[:10]


def _extract_abstract(text: str) -> Optional[str]:
    match = re.search(
        r'\babstract\b[\s\-:]*(.+?)(?=\n{2,}|\b(?:introduction|keywords|1\.)\b)',
        text, re.IGNORECASE | re.DOTALL,
    )
    if match:
        abstract = re.sub(r'\s+', ' ', match.group(1)).strip()
        return abstract if len(abstract) > 50 else None
    return None


def _extract_doi(text: str) -> Optional[str]:
    match = re.search(r'\b(10\.\d{4,}/\S+)', text)
    return match.group(1).rstrip('.,)') if match else None


def _extract_year(text: str) -> Optional[int]:
    matches = re.findall(r'\b((?:19|20)\d{2})\b', text)
    return int(matches[0]) if matches else None