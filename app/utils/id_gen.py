"""Deterministic element ID generation."""

def make_element_id(page: int, kind: str, idx: int) -> str:
    """e.g. make_element_id(4, "table", 0) → 'p4_table_0'"""
    return f"p{page}_{kind}_{idx}"