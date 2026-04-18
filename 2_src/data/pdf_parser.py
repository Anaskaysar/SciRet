from __future__ import annotations

from pathlib import Path
from typing import List, Dict


def discover_pdf_files(pdf_root: Path) -> List[Path]:
    if not pdf_root.exists():
        return []
    return sorted(pdf_root.rglob("*.pdf"))


def extract_figure_manifest_stub(pdf_paths: List[Path]) -> List[Dict[str, str]]:
    """
    Placeholder parser that returns empty records.
    Replace with PyMuPDF/pdfplumber extraction in multimodal phase.
    """
    records: List[Dict[str, str]] = []
    for p in pdf_paths:
        records.append(
            {
                "figure_id": f"{p.stem}_fig_000",
                "cord_uid": p.stem,
                "asset_path": str(p),
                "caption": "",
            }
        )
    return records
