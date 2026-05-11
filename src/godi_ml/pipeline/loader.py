"""DataLoader: read CSV or Parquet into a pandas DataFrame."""

from pathlib import Path

import pandas as pd


class LoaderError(Exception):
    """Raised when data loading fails."""


class DataLoader:
    """Load tabular data from CSV or Parquet."""

    SUPPORTED_SUFFIXES = (".csv", ".parquet", ".pq")

    def __init__(self, target: str) -> None:
        self.target = target

    def load(self, path: Path) -> pd.DataFrame:
        """Load a file into a DataFrame, validating target column presence.

        Raises LoaderError on: unsupported format, missing target, empty df.
        """
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise LoaderError(
                f"unsupported file format {suffix!r} (supported: {self.SUPPORTED_SUFFIXES})"
            )

        df = pd.read_csv(path) if suffix == ".csv" else pd.read_parquet(path)

        if len(df) == 0:
            raise LoaderError(f"empty DataFrame loaded from {path}")

        if self.target not in df.columns:
            raise LoaderError(
                f"target column {self.target!r} not found in columns {list(df.columns)}"
            )

        return df
