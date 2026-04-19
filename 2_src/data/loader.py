from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class CORDDataLoader:
    """Load and persist CORD-19 tabular artifacts for SciRet."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).resolve().parents[2]
        self.raw_dir = self.root_dir / "1_data" / "raw"
        self.processed_dir = self.root_dir / "1_data" / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_metadata(self, filename: str = "metadata.csv") -> pd.DataFrame:
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        return pd.read_csv(path, low_memory=False)

    def build_tier_subset(
        self,
        df: pd.DataFrame,
        tier_size: int = 1000,
        seed: int = 42,
        min_abstract_chars: int = 100,
    ) -> pd.DataFrame:
        data = df.copy()
        if "cord_uid" in data.columns:
            data = data.drop_duplicates(subset=["cord_uid"])
        if "abstract" in data.columns:
            mask = data["abstract"].fillna("").str.len() > min_abstract_chars
            data = data[mask]
        n = min(tier_size, len(data))
        return data.sample(n=n, random_state=seed) if n > 0 else data.head(0)

    def save_clean_subset(self, df: pd.DataFrame, filename: str = "papers_clean.parquet") -> Path:
        path = self.processed_dir / filename
        df.to_parquet(path, index=False)
        return path
