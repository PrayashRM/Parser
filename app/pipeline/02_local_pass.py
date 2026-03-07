# runs the whole local pass, that is the whole section of PyMuPDF4LLM

"""
local_pass.py — Phase 02: Fast Local Pass
PyMuPDF4LLM page-by-page extraction with:
  - Multi-column pre-pass (fixes wrong reading order before extraction)
  - Stateful SectionTracker (fixes empty section_path across page boundaries)
  - Raster figure extraction with OriginalAsset storage
  - OriginalAsset crops for all figure regions
"""
import base64
import re
import tempfile
from pathlib import Path
from typing import List, Tuple

import pymupdf
import pymupdf4llm

from app.models import (
    ContentSource, ContentType, DocumentModel, EscalationReason,
    OriginalAsset, PageRecord, ParsedElement, SectionTracker,
)
from app.config import RENDER_DPI
from utils.id_gen import make_element_id
from utils.temp_manager import TempFileManager


async def run_local_pass(pdf_bytes: bytes, doc: DocumentModel) -> DocumentModel:
    async with TempFileManager() as tmp:
        pdf_path = tmp.write_bytes("input.pdf", pdf_bytes)
        img_dir  = tmp.make_subdir("raster_figures")

        # page_chunks=True is critical — gives per-page dicts, not one flat string.
        # Without this, Phase 06 assembly becomes an unsolvable merge problem.
        page_chunks = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks  = True,
            write_images = True,
            image_path   = str(img_dir),
            image_format = "png",
        )

        doc.total_pages = len(page_chunks)
        tracker = SectionTracker()
        raw_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

        for chunk in page_chunks:
            page_num = chunk["page"]       # 1-indexed
            markdown = chunk["text"]
            raw_page = raw_doc[page_num - 1]

            # --- Multi-column pre-pass -------------------------------------------
            # Detect column layout from block x-coordinates BEFORE extraction.
            # Prevents wrong reading order on 2-column academic papers.
            col_count, col_bounds = _detect_columns(raw_page)

            # --- Stateful section tracking (fixes cross-page section_path bug) ---
            section_path = tracker.update(markdown, page_num)

            # --- Figure elements from raster images injected by pymupdf4llm ------
            figure_elements = _extract_raster_figures(markdown, page_num, img_dir, section_path)

            # --- Clean prose text ------------------------------------------------
            clean_text = _strip_image_tags(markdown).strip()
            elements: List[ParsedElement] = []

            if clean_text:
                elements.append(ParsedElement(
                    element_id   = make_element_id(page_num, "text", 0),
                    content_type = ContentType.TEXT,
                    content      = clean_text,
                    page_number  = page_num,
                    confidence   = 0.9,
                    source       = ContentSource.PYMUPDF,
                    section_path = section_path,
                ))

            elements.extend(figure_elements)

            # --- Escalation pre-flag for multi-column pages ----------------------
            escalation_reasons = []
            if col_count > 1:
                escalation_reasons.append(EscalationReason.MULTI_COLUMN)

            doc.pages[page_num] = PageRecord(
                page_number        = page_num,
                confidence         = 0.9,
                source             = ContentSource.PYMUPDF,
                elements           = elements,
                column_count       = col_count,
                column_bounds      = col_bounds,
                escalation_reasons = escalation_reasons,
            )

        doc.section_tree = tracker.get_tree()
        raw_doc.close()

    return doc


# ---------------------------------------------------------------------------
# Column detection pre-pass
# ---------------------------------------------------------------------------

def _detect_columns(page: pymupdf.Page) -> Tuple[int, List[Tuple[float, float]]]:
    """
    Detect text column layout by clustering block x-coordinates.
    Returns (column_count, [(x_start, x_end), ...]).

    Two clusters of blocks with non-overlapping x-ranges = multi-column layout.
    Extraction per-column then merged in reading order prevents wrong text order.
    """
    blocks = page.get_text("blocks")   # (x0, y0, x1, y1, text, block_no, block_type)
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]  # type 0 = text

    if len(text_blocks) < 4:
        return 1, []

    page_width = page.rect.width
    mid = page_width / 2

    left_blocks  = [b for b in text_blocks if b[2] < mid * 1.1]   # x1 < ~mid
    right_blocks = [b for b in text_blocks if b[0] > mid * 0.9]   # x0 > ~mid

    # Two-column: both sides have substantial content and don't heavily overlap
    if len(left_blocks) >= 3 and len(right_blocks) >= 3:
        left_x_end   = max(b[2] for b in left_blocks)
        right_x_start = min(b[0] for b in right_blocks)
        if right_x_start > left_x_end * 0.85:  # <15% overlap = genuinely 2-column
            return 2, [(min(b[0] for b in left_blocks),  left_x_end),
                       (right_x_start, max(b[2] for b in right_blocks))]

    return 1, []


# ---------------------------------------------------------------------------
# Raster figure extraction
# ---------------------------------------------------------------------------

_IMG_TAG_RE = re.compile(r'!\[.*?\]\((.*?\.png)\)', re.IGNORECASE)


def _extract_raster_figures(
    markdown: str,
    page_num: int,
    img_dir: Path,
    section_path: List[str],
) -> List[ParsedElement]:
    """
    PyMuPDF4LLM injects markdown image tags for raster-embedded figures.
    Extract them, load the PNG, encode as base64 for OriginalAsset.
    """
    elements = []
    for idx, match in enumerate(_IMG_TAG_RE.finditer(markdown)):
        img_path = Path(match.group(1))
        if not img_path.is_absolute():
            img_path = img_dir / img_path.name

        image_b64 = None
        if img_path.exists():
            image_b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")

        # Try to find a caption on the line immediately after the image tag
        caption = _find_caption(markdown, match.end())

        elements.append(ParsedElement(
            element_id   = make_element_id(page_num, "figure", idx),
            content_type = ContentType.FIGURE,
            content      = "",    # Filled by Gemini Route B in Phase 05
            page_number  = page_num,
            confidence   = 0.5,   # Low until Gemini describes it
            source       = ContentSource.PYMUPDF,
            caption      = caption,
            section_path = section_path,
            original     = OriginalAsset(image_b64=image_b64, page_number=page_num),
        ))

    return elements


def _find_caption(markdown: str, pos: int) -> str | None:
    """Look for a caption-like line immediately after an image tag."""
    tail = markdown[pos:pos + 200].lstrip("\n")
    first_line = tail.split("\n")[0].strip()
    if first_line.lower().startswith(("fig", "figure", "table")):
        return first_line
    return None


def _strip_image_tags(markdown: str) -> str:
    """Remove pymupdf4llm image injection tags from prose text."""
    return _IMG_TAG_RE.sub("", markdown)