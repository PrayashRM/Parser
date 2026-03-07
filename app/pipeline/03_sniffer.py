# Confidence engine to validate the final parsed output per step

"""
sniffer.py — Phase 03: Confidence Sniffer
Multi-signal per-page confidence scorer. Produces escalation queue for Phase 05.
Signals are independent; weights tuned empirically on research PDFs.
"""
import re
import statistics
from typing import List, Tuple

import pymupdf

from app.config import ESCALATION_THRESHOLD
from app.models import (
    DocumentModel, EscalationReason, PageRecord, SnifferSignals,
)


def run_sniffer(doc: DocumentModel, pdf_bytes: bytes) -> DocumentModel:
    """
    Score every page. Pages below ESCALATION_THRESHOLD get needs_review=True
    and escalation_reasons populated. Returns the mutated DocumentModel.
    """
    raw_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    for page_num, page_record in doc.pages.items():
        raw_page = raw_doc[page_num - 1]

        # Reconstruct markdown text from existing TEXT elements for signal analysis
        markdown = " ".join(
            el.content for el in page_record.elements
            if el.content_type.value == "text"
        )

        confidence, signals, reasons = _score_page(markdown, raw_page, page_record)

        page_record.confidence      = confidence
        page_record.sniffer_signals = signals
        page_record.needs_review    = confidence < ESCALATION_THRESHOLD

        # Merge with any reasons already set (e.g. MULTI_COLUMN from Phase 02)
        existing = set(page_record.escalation_reasons)
        for r in reasons:
            if r not in existing:
                page_record.escalation_reasons.append(r)

        if page_record.needs_review and not page_record.escalation_reasons:
            # Fallback: mark for review even if no specific signal fired
            page_record.escalation_reasons.append(EscalationReason.LOW_EXTRACT_YIELD)

    raw_doc.close()
    return doc


def _score_page(
    markdown: str,
    raw_page: pymupdf.Page,
    page_record: PageRecord,
) -> Tuple[float, SnifferSignals, List[EscalationReason]]:

    words       = markdown.split()
    word_count  = max(len(words), 1)
    reasons: List[EscalationReason] = []

    # --- Signal 1: Isolated character ratio -----------------------------------
    # Broken math renders as "x 2 + y 2 = r 2" — single alpha chars separated
    single_chars = [w for w in words if len(w) == 1 and w.isalpha()]
    isolated_ratio = len(single_chars) / word_count

    # --- Signal 2: Math keyword without LaTeX block ---------------------------
    has_math_kw = bool(re.search(
        r'\b(equation|theorem|lemma|formula|derivation|corollary|proof)\b',
        markdown, re.IGNORECASE,
    ))
    has_latex = bool(re.search(r'\$\$.+?\$\$|\\\[.+?\\\]|\$.+?\$', markdown, re.DOTALL))
    math_kw_no_latex = has_math_kw and not has_latex

    # --- Signal 3: Naked Greek letters ----------------------------------------
    # Unrendered: α β γ appear as isolated UTF-8 chars outside any LaTeX env
    greek   = re.findall(r'[αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ]', markdown)
    latex_e = re.findall(r'\$[^$]+\$|\$\$[^$]+\$\$', markdown)
    naked_greek = len(greek) > 3 and len(latex_e) == 0

    # --- Signal 4: Borderless table -------------------------------------------
    # Lines with 2+ large whitespace gaps and no pipe characters
    lines      = [l for l in markdown.splitlines() if l.strip()]
    pipe_lines = [l for l in lines if '|' in l]
    gappy_lines = [l for l in lines if re.search(r'\S\s{3,}\S', l) and '|' not in l]
    borderless_table = len(gappy_lines) >= 3 and len(pipe_lines) < 2

    # --- Signal 5: Median word length -----------------------------------------
    word_lengths = [len(w) for w in words if w.isalpha()]
    median_wl    = statistics.median(word_lengths) if word_lengths else 5.0
    low_median   = median_wl < 3.0

    # --- Signal 6: PyMuPDF block count vs markdown word count -----------------
    native_blocks  = len(raw_page.get_text("blocks"))
    low_text_yield = native_blocks > 10 and len(words) < 30

    signals = SnifferSignals(
        isolated_char_ratio        = round(isolated_ratio, 3),
        math_keyword_without_latex = math_kw_no_latex,
        naked_greek                = naked_greek,
        borderless_table           = borderless_table,
        low_median_word_length     = low_median,
        low_text_yield             = low_text_yield,
        multi_column_detected      = page_record.column_count > 1,
    )

    # --- Weighted penalty calculation -----------------------------------------
    penalty = 0.0
    if isolated_ratio > 0.25:   penalty += 0.30; reasons.append(EscalationReason.BROKEN_MATH)
    if math_kw_no_latex:        penalty += 0.25; reasons.append(EscalationReason.MISSING_LATEX)
    if naked_greek:             penalty += 0.20; reasons.append(EscalationReason.NAKED_GREEK)
    if borderless_table:        penalty += 0.25; reasons.append(EscalationReason.BORDERLESS_TABLE)
    if low_median:              penalty += 0.15; reasons.append(EscalationReason.FRAGMENTED_TEXT)
    if low_text_yield:          penalty += 0.20; reasons.append(EscalationReason.LOW_EXTRACT_YIELD)

    confidence = round(max(0.0, 1.0 - penalty), 3)
    return confidence, signals, reasons