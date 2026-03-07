"""
main.py — FastAPI entry point
Single POST /parse endpoint + GET /health.
Gateway validates before a byte of PDF touches any processing logic.
"""
from fastapi import FastAPI, Header, UploadFile
from fastapi.responses import JSONResponse

from app.models import DocumentModel
from pipeline import run_pipeline
from pipeline.gateway import validate_and_load, validate_gemini_key

app = FastAPI(title="research-rag-parser", version="1.0.0")


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