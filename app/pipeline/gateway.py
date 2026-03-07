# validate all the pre-requisites like the users gemini-key , etc.

"""
gateway.py — Phase 01: Ingestion Gateway
                
                -> validate_and_load() - validates the file regarding
                                        :- being a pdf or not via magic bytes
                                        :- pdf being in the size limit(default size limit - 50MB)
                                        :- pdf mime check to block malicious or corrupt pdf docs

                -> validate_gemini_key() - validates the gemini key given by the user regarding being a valid and non-exhausted key. 

Hard filter before any processing. Every failure mode caught here — before
compute is spent, API quota used, or anything can corrupt downstream phases.
"""
import asyncio

import magic
import google.generativeai as genai
from fastapi import HTTPException, UploadFile

from app.config import MAX_FILE_BYTES, GEMINI_MODEL

PDF_MAGIC = b"%PDF"


async def validate_and_load(file: UploadFile) -> bytes:
    """
    Three-layer validation: magic bytes → size gate → MIME double-check.
    Returns raw PDF bytes on success; raises HTTPException on any failure.
    """
    header = await file.read(4)
    if header != PDF_MAGIC:
        raise HTTPException(400, "Invalid file: not a PDF (magic bytes check failed)")

    rest = await file.read()
    pdf_bytes = header + rest

    if len(pdf_bytes) > MAX_FILE_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB limit")

    mime = magic.from_buffer(pdf_bytes, mime=True)
    if mime != "application/pdf":
        raise HTTPException(400, f"MIME validation failed: got '{mime}', expected 'application/pdf'")

    return pdf_bytes


async def validate_gemini_key(api_key: str) -> genai.GenerativeModel:
    """
    Validates Gemini key with a minimal generate_content probe (not list_models —
    list_models succeeds even when the key lacks generateContent permission).
    Raises HTTP 401 immediately on failure. Key is request-scoped (BYOK pattern).
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        await asyncio.to_thread(
            model.generate_content,
            "hi",
            generation_config=genai.GenerationConfig(max_output_tokens=1),
        )
        return model
    except Exception as e:
        raise HTTPException(401, f"Invalid Gemini API key: {e}")