"""Tests for godi_ml.pipeline.loader."""

from pathlib import Path

import pandas as pd
import pytest

from godi_ml.pipeline.loader import DataLoader, LoaderError


def test_load_csv(tmp_path: Path) -> None:
    """DataLoader should load a basic CSV."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b,target\n1,2,0\n3,4,1\n5,6,0\n")
    loader = DataLoader(target="target")
    df = loader.load(csv_path)
    assert len(df) == 3
    assert list(df.columns) == ["a", "b", "target"]


def test_load_parquet(tmp_path: Path) -> None:
    """DataLoader should load Parquet files."""
    pq_path = tmp_path / "data.parquet"
    pd.DataFrame({"x": [1, 2], "target": [0, 1]}).to_parquet(pq_path)
    loader = DataLoader(target="target")
    df = loader.load(pq_path)
    assert len(df) == 2


def test_target_column_missing(tmp_path: Path) -> None:
    """LoaderError raised if target column missing."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n")
    loader = DataLoader(target="missing")
    with pytest.raises(LoaderError, match="target column 'missing' not found"):
        loader.load(csv_path)


def test_unsupported_format(tmp_path: Path) -> None:
    """LoaderError raised for unsupported file format."""
    bad_path = tmp_path / "data.xlsx"
    bad_path.write_text("noop")
    loader = DataLoader(target="t")
    with pytest.raises(LoaderError, match="unsupported"):
        loader.load(bad_path)


def test_empty_dataframe(tmp_path: Path) -> None:
    """LoaderError raised for empty DataFrame."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,target\n")  # header only
    loader = DataLoader(target="target")
    with pytest.raises(LoaderError, match="empty"):
        loader.load(csv_path)
