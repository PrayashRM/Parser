# assemble_document(), page-indexed merge, serialize_to_markdown()

"""
assembly.py — Phase 06: Validated Assembly
Page-indexed merge. Deduplicates overlapping figure elements.
Updates page-level source tags and emits final Markdown.
"""
from app.models import (
    BBOX_DUPLICATE_THRESHOLD, ContentSource, ContentType,
    DocumentModel, bbox_overlap_ratio,
)


def assemble_document(doc: DocumentModel) -> DocumentModel:
    """
    Escalation.py appended tables/equations directly to page_record.elements,
    so there is no merge step. We only need to:
      1. Deduplicate FIGURE elements where vector + raster both hit the same region.
      2. Update page-level source and confidence from final element set.
    """
    for page_record in doc.pages.values():
        page_record.elements   = _deduplicate_figures(page_record.elements)
        page_record.source, page_record.confidence = _page_source(page_record.elements)
    return doc


def _deduplicate_figures(elements: list) -> list:
    """
    Remove FIGURE elements whose bbox overlaps a higher-confidence FIGURE
    by > BBOX_DUPLICATE_THRESHOLD. Keeps the higher-confidence element.
    Non-FIGURE and bbox-less elements are always preserved.
    """
    figures      = [el for el in elements if el.content_type == ContentType.FIGURE and el.bbox]
    non_figs     = [el for el in elements if el.content_type != ContentType.FIGURE]
    no_bbox_figs = [el for el in elements if el.content_type == ContentType.FIGURE and not el.bbox]

    figures.sort(key=lambda el: el.confidence, reverse=True)
    kept = []
    for candidate in figures:
        if not any(bbox_overlap_ratio(candidate.bbox, k.bbox) > BBOX_DUPLICATE_THRESHOLD
                   for k in kept if k.bbox):
            kept.append(candidate)

    return non_figs + kept + no_bbox_figs


def _page_source(elements: list) -> tuple:
    if not elements:
        return ContentSource.PYMUPDF, 0.5
    sources  = {el.source for el in elements}
    avg_conf = round(sum(el.confidence for el in elements) / len(elements), 4)
    if sources == {ContentSource.PYMUPDF}:   return ContentSource.PYMUPDF, avg_conf
    if sources == {ContentSource.GEMINI}:    return ContentSource.GEMINI,  avg_conf
    return ContentSource.HYBRID, avg_conf


# ---------------------------------------------------------------------------
# Markdown serializer
# ---------------------------------------------------------------------------

def serialize_to_markdown(doc: DocumentModel) -> str:
    out = []
    if doc.title:    out.append(f"# {doc.title}\n")
    if doc.authors:  out.append(f"**Authors:** {', '.join(doc.authors)}\n")
    if doc.abstract: out.append(f"**Abstract:** {doc.abstract}\n")

    for page_num in sorted(doc.pages):
        out.append(f"\n<!-- page {page_num} -->\n")
        for el in doc.pages[page_num].elements:
            if el.content_type == ContentType.TEXT:
                out.append(el.content)
            elif el.content_type == ContentType.TABLE:
                if el.caption: out.append(f"\n**{el.caption}**\n")
                out.append(f"\n{el.content}\n")
            elif el.content_type == ContentType.EQUATION:
                if el.equation_context: out.append(f"\n<!-- {el.equation_context} -->")
                out.append(f"\n{el.content}\n")
            elif el.content_type == ContentType.FIGURE:
                caption = f": {el.caption}" if el.caption else ""
                out.append(f"\n> **[Figure{caption}]** {el.content}\n")

    return "\n".join(out)