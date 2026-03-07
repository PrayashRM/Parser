"""
main.py — FastAPI entry point
Single POST /parse endpoint + GET /health.
Gateway validates before a byte of PDF touches any processing logic.
"""
from fastapi import FastAPI, Header, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.models import DocumentModel
from app.config import ASSETS_BASE_DIR
from pipeline import run_pipeline
from pipeline.gateway import validate_and_load, validate_gemini_key

app = FastAPI(title="research-rag-parser", version="1.0.0")

app.mount("/assets", StaticFiles(directory=ASSETS_BASE_DIR), name="assets")

"""
Then on the UI, any element's image is accessible at:

GET /assets/{doc_id}/assets/p3_figure_0.png
"""

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/parse")
async def parse_document(
    file: UploadFile,
    x_gemini_key: str = Header(..., alias="X-Gemini-Key"),
):
    pdf_bytes = await validate_and_load(file)
    model     = await validate_gemini_key(x_gemini_key)
    result    = await run_pipeline(pdf_bytes, model)
    return result