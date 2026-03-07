# Exports a single function - run_pipeline()
# orchestrates the whole pipeline

"""
pipeline/__init__.py — Orchestrator
The single run_pipeline() function that sequences all phases in order.
This is the only file that knows the phase execution order.
Add, remove, or reorder phases here — no other file changes needed.
"""
import google.generativeai as genai



from app.models import DocumentModel
from pipeline.gateway import validate_and_load, validate_gemini_key
from pipeline.local_pass import run_local_pass
from pipeline.sniffer import run_sniffer
from pipeline.vector_detect import run_vector_detection
from pipeline.escalation import run_escalation
from pipeline.assembly import run_assembly
from pipeline.output_schema import build_final_response
from app.models import ParseResponse
from utils.asset_store import AssetStore


async def run_pipeline(
    pdf_bytes: bytes,
    doc_id: str,
    model: genai.GenerativeModel,
) -> ParseResponse:

    # Phase 00 — initialise document model
    doc = DocumentModel.create(pdf_bytes)
    store = AssetStore(doc.doc_id) 

    # Phase 02 — fast local extraction (page-indexed, multi-column aware)
    doc = await run_local_pass(pdf_bytes, doc, store)

    # Phase 03 — per-page confidence scoring + escalation queue
    doc = run_sniffer(doc, pdf_bytes)

    # Phase 04 — vector figure detection (runs concurrently with sniffer output)
    doc = await run_vector_detection(pdf_bytes, doc, store)

    # Phase 05 — AI escalation: broken pages + figure descriptions
    doc = await run_escalation(pdf_bytes, doc, model, store)

    # Phase 06 — validated assembly: merge, sort, clean
    doc = run_assembly(doc)

    # Phase 07 — structured output
    return build_final_response(doc)