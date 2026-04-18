from __future__ import annotations

from typing import List


class VisualGenerator:
    """Placeholder visual answer module."""

    def answer_with_figures(self, query: str, figure_ids: List[str]) -> str:
        if not figure_ids:
            return f"No figure evidence available for query: {query}"
        joined = ", ".join(figure_ids[:5])
        return f"Figure-aware answer path selected for '{query}'. Candidate figures: {joined}"
