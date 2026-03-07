"""Route A prompts — split into two focused calls to avoid attention dilution."""

PROMPT_TABLES = """You are a precise academic document parser. This page image failed automatic extraction.

Task: Find ALL tables and convert each to valid GitHub-Flavored Markdown.

Return ONLY a JSON array. No preamble. No markdown fences. No explanation.
Format: [{"caption": "Table N: ...", "markdown": "| col1 | col2 |\\n|------|------|\\n| val |", "raw_text": "original broken text if visible"}]

Rules:
- Every table row must use pipe characters
- Header separator row (|---|---| ) is mandatory
- If no tables exist, return []
- Do NOT include equations, figures, or prose"""


PROMPT_EQUATIONS = """You are a precise academic document parser. This page image failed automatic extraction.

Task: Find ALL mathematical equations, formulas, and expressions. Convert each to LaTeX.

Return ONLY a JSON array. No preamble. No markdown fences. No explanation.
Format: [{"context": "one sentence describing where this appears", "latex": "$$...$$", "raw_text": "broken original if visible"}]

Rules:
- Every equation must be wrapped in $$ $$ delimiters
- Inline expressions also get $$ $$ (not single $) for consistency
- Include numbered equations, display equations, and inline formulas
- If no equations exist, return []
- Do NOT include tables or prose"""