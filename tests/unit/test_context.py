"""Tests for godi_ml.core.context."""

from pathlib import Path

import pandas as pd
import pytest

from godi_ml.core.context import RunContext


def test_run_context_creation():
    """RunContext should hold all pipeline state."""
    ctx = RunContext(
        run_id="test123",
        task="churn",
        target="churned",
        data_path=Path("data.csv"),
        output_dir=Path("godi-ml-runs/test123"),
        lang="en",
        random_seed=42,
    )
    assert ctx.run_id == "test123"
    assert ctx.task == "churn"
    assert ctx.target == "churned"
    assert ctx.df is None  # initially empty
    assert ctx.metrics == {}  # initially empty


def test_run_context_stores_dataframe():
    """RunContext should accept a pandas DataFrame."""
    ctx = RunContext(
        run_id="test",
        task="churn",
        target="y",
        data_path=Path("d.csv"),
        output_dir=Path("o"),
    )
    ctx.df = pd.DataFrame({"y": [0, 1], "x": [1.0, 2.0]})
    assert ctx.df is not None
    assert len(ctx.df) == 2


def test_run_context_invalid_task():
    """Task must be one of churn, forecast, score."""
    with pytest.raises(ValueError):
        RunContext(
            run_id="x",
            task="invalid",  # type: ignore[arg-type]
            target="y",
            data_path=Path("d"),
            output_dir=Path("o"),
        )
