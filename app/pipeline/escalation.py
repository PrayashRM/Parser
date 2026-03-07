# 3-route Gemini caller, async semaphore, tenacity backoff, response validators

"""
escalation.py — Phase 05: AI Escalation Engine
Three Gemini routes, async, semaphore-gated, tenacity backoff.
Route A1: broken page tables  |  Route A2: broken page equations
Route B:  raster figures      |  Route C:  vector figures
All responses validated before acceptance. Failures logged, never crash pipeline.
"""
import asyncio
import base64
import io
import json
import re
from typing import Dict, List, Optional, Tuple

import pymupdf
import google.generativeai as genai
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import GEMINI_CONCURRENCY, RENDER_DPI
from app.models import (
    ContentSource, ContentType, DocumentModel, EscalationReason,
    FigureKind, OriginalAsset, ParsedElement, PageRecord
)
from prompts.route_a import PROMPT_TABLES, PROMPT_EQUATIONS
from prompts.route_bc import PROMPT_FIGURE_DESC
from utils.id_gen import make_element_id
from utils.asset_store import AssetStore


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_escalation(
    model: genai.GenerativeModel,
    doc: DocumentModel,
    pdf_bytes: bytes,
    store: AssetStore,
) -> DocumentModel:
    """
    Build task list from flagged pages + figure elements, run concurrently,
    merge results back into doc.
    """
    sem      = asyncio.Semaphore(GEMINI_CONCURRENCY)
    tasks: List[tuple] = []

    # Flat element index for O(1) lookup when applying figure results
    el_index: Dict[str, ParsedElement] = {
        el.element_id: el
        for page in doc.pages.values()
        for el in page.elements
    }

    # Render pages that need reconstruction — store bytes in memory for this request
    page_png_cache: Dict[int, bytes] = {}

    for page_num, page_record in doc.pages.items():
        if page_record.needs_review:
            png = _render_page_image(pdf_bytes, page_num)
            if png:
                page_png_cache[page_num] = png
                tasks.append(("a1", page_num))   # tables
                tasks.append(("a2", page_num))   # equations

        for el in page_record.by_type(ContentType.FIGURE):
            if el.has_original_image:
                route = "c" if EscalationReason.VECTOR_FIGURE in page_record.escalation_reasons else "b"
                tasks.append((route, page_num, el.element_id))

    # -------------------------------------------------------------------------
    async def run(task: tuple):
        route    = task[0]
        page_num = task[1]

        async with sem:
            try:
                if route in ("a1", "a2"):
                    png = page_png_cache.get(page_num)
                    if png is None:
                        return route, page_num, None
                    data = await _call_route_a(model, png, page_num, route)
                    return route, page_num, data

                else:   # "b" or "c"
                    el_id = task[2]
                    el    = el_index.get(el_id)
                    if el is None or not el.has_original_image:
                        return route, page_num, None
                    desc, kind = await _call_route_bc(model, el.original.image_b64, route == "c")
                    return route, page_num, (el_id, desc, kind)

            except Exception as exc:
                doc.log_error(f"Escalation {route} p{page_num}: {exc}")
                return route, page_num, None

    # -------------------------------------------------------------------------
    results = await asyncio.gather(*[run(t) for t in tasks], return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            doc.log_error(f"Escalation gather error: {result}")
            continue

        route, page_num, data = result
        if data is None or page_num not in doc.pages:
            continue

        page_record = doc.pages[page_num]

        if route == "a1":
            _apply_tables(data, page_num, page_record, store, page_png_cache[page_num])
        elif route == "a2":
            _apply_equations(data, page_num, page_record)
        elif route in ("b", "c"):
            el_id, desc, kind = data
            _apply_figure_desc(el_index, el_id, desc, kind)

    return doc

# ---------------------------------------------------------------------------
# Page image renderer (Route A)
# ---------------------------------------------------------------------------

def _render_page_image(pdf_bytes: bytes, page_num: int) -> Optional[bytes]:
    """Render a full page to PNG bytes at RENDER_DPI. Returns None on failure."""
    try:
        doc  = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        page = doc[page_num - 1]
        mat  = pymupdf.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        doc.close()
        return pix.tobytes("png")
    except Exception:
        return None
    
def _bytes_to_pil(png_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png_bytes))

# ---------------------------------------------------------------------------
# Route A1 — tables from broken page
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=3, max=30), reraise=True)
async def _call_route_a(
    model: genai.GenerativeModel,
    page_png_bytes: bytes,
    page_num: int,
    mode: str,                   # "a1" = tables, "a2" = equations
) -> List[dict]:
    prompt = PROMPT_TABLES if mode == "a1" else PROMPT_EQUATIONS
    img    = await asyncio.to_thread(_bytes_to_pil, page_png_bytes)
    resp   = await asyncio.to_thread(
        model.generate_content,
        [prompt, img],
        generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=4096),
    )
    if mode == "a1":
        return _parse_json_list(resp.text, page_num,
                                required_key="markdown",
                                validator=lambda t: "|" in t.get("markdown", ""))
    else:
        return _parse_json_list(resp.text, page_num,
                                required_key="latex",
                                validator=lambda e: "$$" in e.get("latex", "") or "\\[" in e.get("latex", ""))


# ---------------------------------------------------------------------------
# Route B/C — figure semantic description
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=3, max=30), reraise=True)
async def _call_route_bc(
    model: genai.GenerativeModel,
    image_b64: str,
    is_vector: bool,
) -> Tuple[str, FigureKind]:
    extra = (
        "\nAlso classify on the last line as exactly: "
        "Type: chart|diagram|graph|table_image|photo|equation_image|unknown"
        if is_vector else ""
    )
    img  = await asyncio.to_thread(_bytes_to_pil, base64.b64decode(image_b64))
    resp = await asyncio.to_thread(
        model.generate_content,
        [PROMPT_FIGURE_DESC + extra, img],
        generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=512),
    )
    text = resp.text.strip()

    kind = FigureKind.UNKNOWN
    m    = re.search(r'Type:\s*(chart|diagram|graph|table_image|photo|equation_image|unknown)',
                     text, re.IGNORECASE)
    if m:
        try:
            kind = FigureKind(m.group(1).lower())
        except ValueError:
            pass
        # Strip the classification line from the description
        text = text[:m.start()].strip()

    return text, kind


# ---------------------------------------------------------------------------
# Apply results to document
# ---------------------------------------------------------------------------

def _apply_tables(
    tables: List[dict],
    page_num: int,
    page_record: PageRecord,
    store: AssetStore,
    page_png_bytes: bytes,         # raw PNG bytes — read once, reused below
) -> None:
    # Remove broken/empty TABLE elements left by PyMuPDF
    page_record.elements = [el for el in page_record.elements
                            if el.content_type != ContentType.TABLE]
    base = len(page_record.by_type(ContentType.TABLE))

    for i, tbl in enumerate(tables):
        filename  = f"p{page_num}_table_{i}_original.png"
        file_path = store.save_bytes(page_png_bytes, filename)

        page_record.elements.append(ParsedElement(
            element_id   = make_element_id(page_num, "table", base + i),
            content_type = ContentType.TABLE,
            content      = tbl["markdown"],
            page_number  = page_num,
            confidence   = 0.92,
            source       = ContentSource.GEMINI,
            caption      = tbl.get("caption"),
            section_path = page_record.elements[0].section_path if page_record.elements else [],
            original     = OriginalAsset(
                raw_text  = tbl.get("raw_text"),
                image_b64 = base64.b64encode(page_png_bytes).decode("ascii"),
                file_path = file_path,
            ),
        ))


def _apply_equations(
    equations: List[dict],
    page_num: int,
    page_record: PageRecord,
) -> None:
    base = len(page_record.by_type(ContentType.EQUATION))
    for i, eq in enumerate(equations):
        page_record.elements.append(ParsedElement(
            element_id        = make_element_id(page_num, "eq", base + i),
            content_type      = ContentType.EQUATION,
            content           = eq["latex"],
            page_number       = page_num,
            confidence        = 0.90,
            source            = ContentSource.GEMINI,
            equation_context  = eq.get("context"),
            section_path      = page_record.elements[0].section_path if page_record.elements else [],
            original          = OriginalAsset(raw_text=eq.get("raw_text")),
        ))


def _apply_figure_desc(
    el_index: Dict[str, ParsedElement],
    el_id: str,
    desc: str,
    figure_kind: FigureKind,
) -> None:
    el = el_index.get(el_id)
    if el:
        el.content     = desc
        el.figure_kind = figure_kind
        el.confidence  = 0.88
        el.source      = ContentSource.GEMINI


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_json_list(
    raw: str,
    page_num: int,
    required_key: str,
    validator,
) -> List[dict]:
    clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        raise ValueError(f"Page {page_num}: non-JSON response: {raw[:200]}")
    items = data if isinstance(data, list) else data.get(required_key + "s", [])
    return [item for item in items if isinstance(item, dict) and validator(item)]  # no comma