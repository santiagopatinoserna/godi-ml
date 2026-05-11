"""Tests for godi_ml.pipeline.eda."""

import numpy as np
import pandas as pd
import pytest

from godi_ml.pipeline.eda import compute_eda


def test_compute_eda_basic() -> None:
    """compute_eda returns expected keys and shapes."""
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "cat1": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    stats = compute_eda(df, target="target", random_seed=42)
    assert stats["n_rows"] == 10
    assert stats["n_cols"] == 3
    assert stats["target_distribution"] == {0: 5, 1: 5}
    assert "missing_pct" in stats
    assert "dtypes" in stats
    assert "mutual_info" in stats
    assert stats["mutual_info"]["cat1"] > 0  # cat1 is perfectly correlated


def test_compute_eda_missing_values() -> None:
    """compute_eda computes missing percentages correctly."""
    df = pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0, np.nan],
            "target": [0, 1, 0, 1],
        }
    )
    stats = compute_eda(df, target="target", random_seed=42)
    assert stats["missing_pct"]["x"] == 50.0
    assert stats["missing_pct"]["target"] == 0.0


def test_compute_eda_high_cardinality_id_flagged() -> None:
    """Columns with all-unique values are flagged as suspicious IDs."""
    df = pd.DataFrame(
        {
            "id": [f"u{i}" for i in range(100)],
            "x": list(range(100)),
            "target": [i % 2 for i in range(100)],
        }
    )
    stats = compute_eda(df, target="target", random_seed=42)
    assert "id" in stats["suspicious_id_columns"]


def test_compute_eda_target_must_exist() -> None:
    """Raises ValueError if target column missing."""
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="target"):
        compute_eda(df, target="missing", random_seed=42)


def test_compute_eda_deterministic() -> None:
    """Same seed → same output (mutual_info estimation is stochastic)."""
    df = pd.DataFrame(
        {
            "x": np.random.RandomState(0).randn(100),
            "target": np.random.RandomState(0).randint(0, 2, 100),
        }
    )
    s1 = compute_eda(df, target="target", random_seed=42)
    s2 = compute_eda(df, target="target", random_seed=42)
    assert s1["mutual_info"]["x"] == s2["mutual_info"]["x"]
