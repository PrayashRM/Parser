"""
models.py — Document Data Contract
All pipeline phases import from here. Every type is validated at runtime by Pydantic.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContentSource(str, Enum):
    PYMUPDF = "pymupdf"   # Local extraction — zero API cost
    GEMINI  = "gemini"    # AI-reconstructed — complex/broken content
    HYBRID  = "hybrid"    # PyMuPDF prose + Gemini tables/equations


class ContentType(str, Enum):
    TEXT     = "text"
    TABLE    = "table"
    EQUATION = "equation"
    FIGURE   = "figure"


class FigureKind(str, Enum):
    """Sub-type for FIGURE elements — enables filtered RAG retrieval."""
    CHART        = "chart"
    DIAGRAM      = "diagram"
    GRAPH        = "graph"
    TABLE_IMG    = "table_image"    # Table rendered as image, not parseable as MD
    PHOTO        = "photo"
    EQUATION_IMG = "equation_image"
    UNKNOWN      = "unknown"


class EscalationReason(str, Enum):
    BROKEN_MATH       = "broken_math"
    MISSING_LATEX     = "missing_latex"
    NAKED_GREEK       = "naked_greek_letters"
    BORDERLESS_TABLE  = "borderless_table"
    FRAGMENTED_TEXT   = "fragmented_text"
    LOW_EXTRACT_YIELD = "low_extraction_yield"
    VECTOR_FIGURE     = "vector_figure_detected"
    RASTER_FIGURE     = "raster_figure_detected"
    MULTI_COLUMN      = "multi_column_layout"


# ---------------------------------------------------------------------------
# Original Asset  — raw source stored alongside its conversion
# ---------------------------------------------------------------------------

class OriginalAsset(BaseModel):
    """
    Stores the unmodified original of a complex element so the UI can render
    the source (image crop / raw text) alongside the parsed representation.

    TABLE    → raw_text: garbled PyMuPDF extraction; image_b64: PNG crop of region
    EQUATION → raw_text: broken char-soup from PyMuPDF;  image_b64: PNG crop
    FIGURE   → raw_text: None;                           image_b64: PNG crop

    image_b64 is base64-encoded PNG. UI: <img src="{{ element.data_uri }}" />
    """
    raw_text:    Optional[str]                               = None
    image_b64:   Optional[str]                               = None
    file_path:   Optional[str]                               = None  # relative path: "assets/p3_figure_0.png"
    mime_type:   str                                         = "image/png"
    bbox:        Optional[Tuple[float, float, float, float]] = None
    page_number: Optional[int]                               = None

    @field_validator("image_b64")
    @classmethod
    def strip_newlines(cls, v: Optional[str]) -> Optional[str]:
        # Some base64 encoders insert newlines every 76 chars — strip them
        return v.replace("\n", "").replace("\r", "") if v else v


# ---------------------------------------------------------------------------
# Parsed Element — atomic unit of content
# ---------------------------------------------------------------------------

class ParsedElement(BaseModel):
    """
    One logical content unit. element_id scheme: "p{page}_{type}_{idx}"

    content  → CONVERTED representation (GFM table / LaTeX / description / prose)
    original → RAW source asset; populated for TABLE, EQUATION, FIGURE; None for TEXT
    """
    element_id:       str
    content_type:     ContentType
    page_number:      int = Field(..., ge=1)
    content:          str   # GFM table | $$LaTeX$$ | figure description | prose
    original:         Optional[OriginalAsset] = None  # Raw original for UI rendering
    confidence:       float = Field(..., ge=0.0, le=1.0)
    source:           ContentSource
    bbox:             Optional[Tuple[float, float, float, float]] = None
    caption:          Optional[str]    = None
    section_path:     List[str]        = Field(default_factory=list)
    figure_kind:      Optional[FigureKind] = None   # FIGURE elements only
    equation_context: Optional[str]    = None       # EQUATION elements only

    @model_validator(mode="after")
    def check_figure_kind_scope(self) -> "ParsedElement":
        if self.figure_kind is not None and self.content_type != ContentType.FIGURE:
            raise ValueError("figure_kind may only be set on FIGURE elements")
        return self

    @property
    def has_original_image(self) -> bool:
        return self.original is not None and self.original.image_b64 is not None

    @property
    def data_uri(self) -> Optional[str]:
        """Ready-to-use HTML data URI. Usage: <img src="{{ element.data_uri }}" />"""
        if not self.has_original_image:
            return None
        return f"data:{self.original.mime_type};base64,{self.original.image_b64}"


# ---------------------------------------------------------------------------
# Bbox deduplication helper  (used by assembly.py)
# ---------------------------------------------------------------------------

BBOX_DUPLICATE_THRESHOLD = 0.6  # overlap ratio above this → treat as duplicate


def bbox_overlap_ratio(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Intersection area / min(area_a, area_b). Returns 0.0 if no overlap."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
    return intersection / min(area_a, area_b)


# ---------------------------------------------------------------------------
# Sniffer Signals  (preserved on PageRecord for audit + threshold tuning)
# ---------------------------------------------------------------------------

class SnifferSignals(BaseModel):
    isolated_char_ratio:        float = 0.0
    math_keyword_without_latex: bool  = False
    naked_greek:                bool  = False
    borderless_table:           bool  = False
    low_median_word_length:     bool  = False
    low_text_yield:             bool  = False
    multi_column_detected:      bool  = False


# ---------------------------------------------------------------------------
# Page Record
# ---------------------------------------------------------------------------

class PageRecord(BaseModel):
    page_number:        int
    confidence:         float = Field(..., ge=0.0, le=1.0)
    source:             ContentSource
    needs_review:       bool                     = False
    escalation_reasons: List[EscalationReason]   = Field(default_factory=list)
    sniffer_signals:    Optional[SnifferSignals]  = None
    elements:           List[ParsedElement]       = Field(default_factory=list)
    column_count:       int                       = Field(1, ge=1, le=6)
    column_bounds:      List[Tuple[float, float]] = Field(default_factory=list)

    def by_type(self, t: ContentType) -> List[ParsedElement]:
        return [el for el in self.elements if el.content_type == t]

    def elements_with_originals(self) -> List[ParsedElement]:
        return [el for el in self.elements if el.original is not None]


# ---------------------------------------------------------------------------
# Section Tree Node
# ---------------------------------------------------------------------------

class SectionNode(BaseModel):
    heading:     str
    level:       int = Field(..., ge=1, le=6)
    page_number: int
    element_id:  Optional[str] = None

SectionNode.model_rebuild()


# ---------------------------------------------------------------------------
# Section Tracker — stateful, one instance per request
# ---------------------------------------------------------------------------

@dataclass
class SectionTracker:
    """
    Carries heading context across page boundaries during Phase 02 (Local Pass).
    Fixes the bug where section_path is empty on pages 4-7 when a section
    starting on page 3 spans multiple pages.

    Usage in local_pass.py:
        tracker = SectionTracker()
        for chunk in page_chunks:
            path = tracker.update(chunk["text"], page_num)
            # assign path to all ParsedElements on this page
    """
    _stack: List[Tuple[int, str]] = field(default_factory=list)
    _nodes: List[SectionNode]     = field(default_factory=list)

    def update(self, markdown: str, page_number: int) -> List[str]:
        for hashes, text in re.findall(r'^(#{1,6})\s+(.+)$', markdown, re.MULTILINE):
            level = len(hashes)
            self._stack = [(l, h) for l, h in self._stack if l < level]
            self._stack.append((level, text.strip()))
            self._nodes.append(SectionNode(heading=text.strip(), level=level, page_number=page_number))
        return self.current_path

    @property
    def current_path(self) -> List[str]:
        return [h for _, h in self._stack]

    def get_tree(self) -> List[SectionNode]:
        return list(self._nodes)

    def reset(self) -> None:
        self._stack.clear()
        self._nodes.clear()


# ---------------------------------------------------------------------------
# Parse Quality Report
# ---------------------------------------------------------------------------

class ParseQualityReport(BaseModel):
    overall_confidence:      float
    pages_total:             int
    pages_local:             int
    pages_escalated:         int
    pages_hybrid:            int
    tables_found:            int            = 0
    equations_found:         int            = 0
    figures_found:           int            = 0
    vector_figures_found:    int            = 0
    raster_figures_found:    int            = 0
    elements_with_originals: int            = 0
    escalation_breakdown:    Dict[str, int] = Field(default_factory=dict)
    parse_errors:            List[str]      = Field(default_factory=list)

    @classmethod
    def from_document(cls, doc: "DocumentModel") -> "ParseQualityReport":
        confidences = []
        pages_local = pages_escalated = pages_hybrid = 0
        tables = equations = figures = vector_figs = raster_figs = with_originals = 0
        escalation_counts: Dict[str, int] = {}

        for page in doc.pages.values():
            confidences.append(page.confidence)
            if   page.source == ContentSource.PYMUPDF: pages_local     += 1
            elif page.source == ContentSource.GEMINI:  pages_escalated += 1
            else:                                      pages_hybrid    += 1

            for r in page.escalation_reasons:
                escalation_counts[r.value] = escalation_counts.get(r.value, 0) + 1

            for el in page.elements:
                if   el.content_type == ContentType.TABLE:    tables     += 1
                elif el.content_type == ContentType.EQUATION: equations  += 1
                elif el.content_type == ContentType.FIGURE:
                    figures += 1
                    if el.has_original_image:
                        if el.source == ContentSource.PYMUPDF: raster_figs += 1
                        else:                                   vector_figs += 1
                if el.original is not None:
                    with_originals += 1

        return cls(
            overall_confidence      = round(sum(confidences) / max(len(confidences), 1), 4),
            pages_total             = doc.total_pages,
            pages_local             = pages_local,
            pages_escalated         = pages_escalated,
            pages_hybrid            = pages_hybrid,
            tables_found            = tables,
            equations_found         = equations,
            figures_found           = figures,
            vector_figures_found    = vector_figs,
            raster_figures_found    = raster_figs,
            elements_with_originals = with_originals,
            escalation_breakdown    = escalation_counts,
            parse_errors            = list(doc.parse_errors),
        )


# ---------------------------------------------------------------------------
# Document Model — pipeline spine
# ---------------------------------------------------------------------------

class DocumentModel(BaseModel):
    """
    Passed through every pipeline phase. Each phase mutates it and passes it forward.
    Pages keyed by 1-indexed page number — makes assembly fully deterministic.
    """
    doc_id:       str                        # MD5 of PDF bytes
    source_hash:  str                        # SHA-256 of PDF bytes
    title:        Optional[str]              = None
    authors:      List[str]                  = Field(default_factory=list)
    abstract:     Optional[str]              = None
    doi:          Optional[str]              = None
    year:         Optional[int]              = None
    venue:        Optional[str]              = None
    pages:        Dict[int, PageRecord]      = Field(default_factory=dict)
    section_tree: List[SectionNode]          = Field(default_factory=list)
    total_pages:  int                        = 0
    parse_errors: List[str]                  = Field(default_factory=list)

    def all_elements(self) -> List[ParsedElement]:
        result = []
        for pn in sorted(self.pages):
            result.extend(self.pages[pn].elements)
        return result

    def by_type(self, t: ContentType) -> List[ParsedElement]:
        return [el for el in self.all_elements() if el.content_type == t]

    def elements_with_originals(self) -> List[ParsedElement]:
        return [el for el in self.all_elements() if el.original is not None]

    def log_error(self, msg: str) -> None:
        self.parse_errors.append(msg)

    def get_quality_report(self) -> ParseQualityReport:
        return ParseQualityReport.from_document(self)

    @classmethod
    def create(cls, pdf_bytes: bytes) -> "DocumentModel":
        return cls(
            doc_id      = hashlib.md5(pdf_bytes).hexdigest(),
            source_hash = hashlib.sha256(pdf_bytes).hexdigest(),
        )


# ---------------------------------------------------------------------------
# Final API Response  (Phase 07 output schema)
# ---------------------------------------------------------------------------

class ParseResponse(BaseModel):
    """
    Structured JSON returned by POST /parse.
    elements array is the primary RAG input — typed, section-aware, never
    split at wrong boundaries. full_markdown for simple downstream use.
    """
    doc_id:        str
    title:         Optional[str]
    authors:       List[str]
    abstract:      Optional[str]
    doi:           Optional[str]  = None
    year:          Optional[int]  = None
    venue:         Optional[str]  = None
    total_pages:   int
    parse_quality: ParseQualityReport
    section_tree:  List[SectionNode]
    full_markdown: str
    elements:      List[ParsedElement]

    @classmethod
    def from_document(cls, doc: DocumentModel, full_markdown: str) -> "ParseResponse":
        return cls(
            doc_id        = doc.doc_id,
            title         = doc.title,
            authors       = doc.authors,
            abstract      = doc.abstract,
            doi           = doc.doi,
            year          = doc.year,
            venue         = doc.venue,
            total_pages   = doc.total_pages,
            parse_quality = ParseQualityReport.from_document(doc),
            section_tree  = doc.section_tree,
            full_markdown = full_markdown,
            elements      = doc.all_elements(),
        )