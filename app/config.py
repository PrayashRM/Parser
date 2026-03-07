"""
config.py — All tunable constants in one place.
Change a value here; the whole system picks it up.
Override via .env for deployment without touching code.
"""
import os

# Gateway
MAX_FILE_BYTES   = int(os.getenv("MAX_FILE_MB", 50)) * 1024 * 1024

# Sniffer
ESCALATION_THRESHOLD = float(os.getenv("ESCALATION_THRESHOLD", 0.65))

# Vector detection
MIN_DRAWING_AREA = 5000    # px² — ignore decorative lines/borders
RENDER_DPI       = int(os.getenv("RENDER_DPI", 216))

# Gemini
GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_CONCURRENCY = int(os.getenv("GEMINI_CONCURRENCY", 8))