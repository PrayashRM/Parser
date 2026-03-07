"""Route B/C prompt — semantic figure description for RAG retrieval."""

PROMPT_FIGURE_DESC = """You are an expert at describing academic research figures for semantic search indexing.

Analyze this figure and write a dense, information-rich description.

Requirements:
- Focus EXCLUSIVELY on: data values, axis labels/ranges, trend directions, statistical relationships, comparisons between groups/models, key conclusions
- Do NOT mention: colors, visual aesthetics, layout, or use the phrase "the figure shows"
- Begin with the figure type (e.g., "Bar chart comparing...", "Line graph showing...")
- Include specific numbers and values wherever visible
- End with the key takeaway a researcher would extract from this figure
- Length: 3–5 sentences. Dense with information. No padding."""