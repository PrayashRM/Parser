# Specially to detect table or mathematical equations in the form of vectors embedded in the pdf.

"""
vector_detect.py — Phase 04: Vector Figure Detection
Finds figures drawn as native PDF vector paths — invisible to PyMuPDF4LLM.
Uses union-find clustering (not naive sweep) to handle non-sequential drawing order.
"""
import base64
from pathlib import Path
from typing import Dict, List, Tuple

import pymupdf

from app.config import MIN_DRAWING_AREA, RENDER_DPI
from app.models import (
    ContentSource, ContentType, DocumentModel, EscalationReason,
    OriginalAsset, ParsedElement,
)
from utils.id_gen import make_element_id
from utils.temp_manager import TempFileManager


async def run_vector_detection(pdf_bytes: bytes, doc: DocumentModel) -> DocumentModel:
    """
    Renders vector figure regions as high-DPI PNG crops and attaches them as
    FIGURE ParsedElements on their respective PageRecords.
    Vector figures start with confidence=0.75 (description pending Gemini Route C).
    """
    async with TempFileManager() as tmp:
        img_dir = tmp.make_subdir("vector_figures")
        figures = _detect_vector_figures(pdf_bytes, img_dir)

        for page_num, page_figures in figures.items():
            if page_num not in doc.pages:
                continue
            page_record    = doc.pages[page_num]
            existing_count = len(page_record.by_type(ContentType.FIGURE))

            for i, fig in enumerate(page_figures):
                b64 = base64.b64encode(Path(fig["img_path"]).read_bytes()).decode()
                el  = ParsedElement(
                    element_id   = make_element_id(page_num, "figure", existing_count + i),
                    content_type = ContentType.FIGURE,
                    content      = f"Vector figure on page {page_num}",  # Gemini Route C fills this
                    page_number  = page_num,
                    confidence   = 0.75,
                    source       = ContentSource.GEMINI,
                    bbox         = fig["bbox"],
                    original     = OriginalAsset(
                        image_b64   = b64,
                        bbox        = fig["bbox"],
                        page_number = page_num,
                    ),
                    section_path = page_record.elements[0].section_path if page_record.elements else [],
                )
                page_record.elements.append(el)

            if page_figures and EscalationReason.VECTOR_FIGURE not in page_record.escalation_reasons:
                page_record.escalation_reasons.append(EscalationReason.VECTOR_FIGURE)

    return doc


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def _detect_vector_figures(pdf_bytes: bytes, img_dir: Path) -> Dict[int, List[dict]]:
    """Returns {page_number: [{"bbox": tuple, "img_path": str}]}"""
    pdf    = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    result = {}

    for page_idx, page in enumerate(pdf):
        page_num = page_idx + 1
        drawings = page.get_drawings()
        if not drawings:
            continue

        clusters = _union_find_cluster(drawings, gap_tolerance=20)
        figures  = []

        for i, bbox in enumerate(clusters):
            x0, y0, x1, y1 = bbox
            area = (x1 - x0) * (y1 - y0)
            if area < MIN_DRAWING_AREA:
                continue   # Skip decorative rules, borders, hairlines

            mat  = pymupdf.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
            rect = pymupdf.Rect(x0 - 4, y0 - 4, x1 + 4, y1 + 4)
            pix  = page.get_pixmap(matrix=mat, clip=rect, alpha=False)

            out_path = img_dir / f"vector_p{page_num}_f{i}.png"
            pix.save(str(out_path))
            figures.append({"bbox": (x0, y0, x1, y1), "img_path": str(out_path)})

        if figures:
            result[page_num] = figures

    pdf.close()
    return result


# ---------------------------------------------------------------------------
# Union-Find clustering — handles non-sequential drawing order
# ---------------------------------------------------------------------------

def _union_find_cluster(drawings: List[dict], gap_tolerance: int) -> List[Tuple[float, float, float, float]]:
    """
    Groups nearby drawing bboxes into figure-level bounding boxes.
    Union-find correctly handles drawings stored out of spatial order in the PDF
    stream — unlike a naive left-to-right sweep which breaks on such PDFs.
    """
    rects  = [tuple(d["rect"]) for d in drawings]
    n      = len(rects)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    def close(a: tuple, b: tuple) -> bool:
        return (max(a[0], b[0]) - min(a[2], b[2]) < gap_tolerance and
                max(a[1], b[1]) - min(a[3], b[3]) < gap_tolerance)

    for i in range(n):
        for j in range(i + 1, n):
            if close(rects[i], rects[j]):
                union(i, j)

    components: Dict[int, List[tuple]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(rects[i])

    return [
        (min(r[0] for r in g), min(r[1] for r in g),
         max(r[2] for r in g), max(r[3] for r in g))
        for g in components.values()
    ]