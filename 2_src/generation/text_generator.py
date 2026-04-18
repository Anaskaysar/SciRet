from __future__ import annotations

from typing import List, Tuple


class TextGenerator:
    """
    Simple grounded answer composer.
    Replace with LLM call for final paper runs.
    """

    def generate(self, query: str, contexts: List[Tuple[str, str]]) -> str:
        if not contexts:
            return "I could not find enough evidence in the indexed corpus."
        snippets = []
        for doc_id, text in contexts[:3]:
            preview = " ".join(text.split()[:45]).strip()
            snippets.append(f"[{doc_id}] {preview}...")
        joined = "\n".join(snippets)
        return (
            f"Question: {query}\n\n"
            "Evidence-backed summary:\n"
            f"{joined}\n\n"
            "Citations correspond to retrieved chunk identifiers."
        )
