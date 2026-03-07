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

from app.config import GEMINI_CONCURRENCY
from app.models import (
    ContentSource, ContentType, DocumentModel, EscalationReason,
    FigureKind, OriginalAsset, ParsedElement,
)
from prompts.route_a import PROMPT_TABLES, PROMPT_EQUATIONS
from prompts.route_bc import PROMPT_FIGURE_DESC
from utils.id_gen import make_element_id


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_escalation(
    model: genai.GenerativeModel,
    doc: DocumentModel,
    pdf_bytes: bytes,
) -> DocumentModel:
    """
    Build task list from flagged pages + figure elements, run concurrently,
    merge results back into doc.
    """
    sem      = asyncio.Semaphore(GEMINI_CONCURRENCY)
    tasks    = []
    el_index = {
        el.element_id: el
        for page in doc.pages.values()
        for el in page.elements
    }

    for page_num, page_record in doc.pages.items():
        if page_record.needs_review:
            page_img = _render_page_image(pdf_bytes, page_num)
            if page_img:
                tasks.append(("a1", page_num, page_img))
                tasks.append(("a2", page_num, page_img))

        for el in page_record.by_type(ContentType.FIGURE):
            if el.has_original_image:
                route = "b" if el.source == ContentSource.PYMUPDF else "c"
                tasks.append((route, page_num, el.element_id))

    async def run(task):
        route, page_num, payload = task
        async with sem:
            try:
                if route in ("a1", "a2"):
                    return route, page_num, await _call_route_a(model, payload, page_num, route)
                else:
                    el = el_index.get(payload)
                    if el and el.has_original_image:
                        desc, kind = await _call_route_bc(model, el.original.image_b64, payload)
                        return route, page_num, (payload, desc, kind)
            except Exception as exc:
                doc.log_error(f"Escalation {route} p{page_num}: {exc}")
        return route, page_num, None

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
            _apply_tables(data, page_num, page_record)
        elif route == "a2":
            _apply_equations(data, page_num, page_record)
        elif route in ("b", "c"):
            el_id, desc, kind = data
            _apply_figure_desc(el_index, el_id, desc, kind)

    return doc


# ---------------------------------------------------------------------------
# Route A1 — tables from broken page
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=3, max=30), reraise=True)
async def _call_route_a(
    model: genai.GenerativeModel,
    page_img: Image.Image,
    page_num: int,
    route: str,
) -> List[dict]:
    prompt = PROMPT_TABLES if route == "a1" else PROMPT_EQUATIONS
    response = await asyncio.to_thread(
        model.generate_content,
        [prompt, page_img],
        generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=4096),
    )
    return _validate_route_a(response.text.strip(), page_num, route)


def _validate_route_a(raw: str, page_num: int, route: str) -> List[dict]:
    clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        raise ValueError(f"p{page_num} route {route}: non-JSON: {raw[:200]}")

    if route == "a1":
        items = data if isinstance(data, list) else data.get("tables", [])
        return [t for t in items if isinstance(t, dict) and "|" in t.get("markdown", "")]
    else:
        items = data if isinstance(data, list) else data.get("equations", [])
        return [e for e in items if isinstance(e, dict) and
                ("$$" in e.get("latex", "") or "\\[" in e.get("latex", ""))]


# ---------------------------------------------------------------------------
# Route B/C — figure semantic description
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=3, max=30), reraise=True)
async def _call_route_bc(
    model: genai.GenerativeModel,
    image_b64: str,
    element_id: str,
) -> Tuple[str, FigureKind]:
    img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    response = await asyncio.to_thread(
        model.generate_content,
        [PROMPT_FIGURE_DESC, img],
        generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=512),
    )
    return _validate_route_bc(response.text.strip(), element_id)


def _validate_route_bc(raw: str, element_id: str) -> Tuple[str, FigureKind]:
    clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
    try:
        data = json.loads(clean)
        desc = data.get("description", "").strip()
        kind_str = data.get("figure_kind", "unknown").lower()
        kind = FigureKind(kind_str) if kind_str in FigureKind._value2member_map_ else FigureKind.UNKNOWN
    except (json.JSONDecodeError, KeyError):
        desc = raw.strip()
        kind = FigureKind.UNKNOWN

    if len(desc) < 10:
        raise ValueError(f"Route BC {element_id}: description too short: '{desc}'")
    return desc, kind


# ---------------------------------------------------------------------------
# Apply results to document
# ---------------------------------------------------------------------------

def _apply_tables(tables: List[dict], page_num: int, page_record) -> None:
    base = len(page_record.by_type(ContentType.TABLE))
    for i, tbl in enumerate(tables):
        page_record.elements.append(ParsedElement(
            element_id   = make_element_id(page_num, "table", base + i),
            content_type = ContentType.TABLE,
            content      = tbl["markdown"],
            page_number  = page_num,
            confidence   = 0.92,
            source       = ContentSource.GEMINI,
            caption      = tbl.get("caption"),
            section_path = page_record.elements[0].section_path if page_record.elements else [],
            original     = OriginalAsset(raw_text=tbl.get("raw_text")),
        ))


def _apply_equations(equations: List[dict], page_num: int, page_record) -> None:
    base = len(page_record.by_type(ContentType.EQUATION))
    for i, eq in enumerate(equations):
        page_record.elements.append(ParsedElement(
            element_id       = make_element_id(page_num, "eq", base + i),
            content_type     = ContentType.EQUATION,
            content          = eq["latex"],
            page_number      = page_num,
            confidence       = 0.90,
            source           = ContentSource.GEMINI,
            equation_context = eq.get("context"),
            section_path     = page_record.elements[0].section_path if page_record.elements else [],
            original         = OriginalAsset(raw_text=eq.get("raw_text")),
        ))


def _apply_figure_desc(
    el_index: Dict[str, ParsedElement],
    el_id: str,
    description: str,
    kind: FigureKind,
) -> None:
    el = el_index.get(el_id)
    if el:
        el.content     = description
        el.figure_kind = kind
        el.confidence  = 0.88
        el.source      = ContentSource.GEMINI


# ---------------------------------------------------------------------------
# Page image renderer (Route A)
# ---------------------------------------------------------------------------

def _render_page_image(pdf_bytes: bytes, page_num: int) -> Optional[Image.Image]:
    try:
        pdf  = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        page = pdf[page_num - 1]
        mat  = pymupdf.Matrix(2.0, 2.0)   # 144 DPI — sufficient for Gemini vision
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        pdf.close()
        return Image.open(io.BytesIO(pix.tobytes("png")))
    except Exception:
        return None