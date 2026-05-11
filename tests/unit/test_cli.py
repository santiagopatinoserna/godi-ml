"""Tests for godi_ml.cli."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from godi_ml.cli import app

runner = CliRunner()


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "age": rng.randint(18, 80, 300),
            "income": rng.randn(300) * 1000 + 50_000,
            "country": rng.choice(["US", "FR"], 300),
            "churned": rng.binomial(1, 0.3, 300),
        }
    )
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    return csv


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "godi-ml" in result.stdout.lower()


def test_cli_churn_command_help() -> None:
    result = runner.invoke(app, ["churn", "--help"])
    assert result.exit_code == 0
    assert "--data" in result.stdout
    assert "--target" in result.stdout


def test_cli_churn_runs_e2e(synthetic_csv: Path, tmp_path: Path) -> None:
    runs_dir = tmp_path / "godi-ml-runs"
    result = runner.invoke(
        app,
        [
            "churn",
            "--data",
            str(synthetic_csv),
            "--target",
            "churned",
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    run_dirs = list(runs_dir.iterdir())
    assert len(run_dirs) == 1
    out = run_dirs[0]
    assert (out / "metrics.json").exists()
    assert (out / "figures").exists()


def test_cli_churn_missing_target(tmp_path: Path) -> None:
    csv = tmp_path / "d.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(csv, index=False)
    result = runner.invoke(app, ["churn", "--data", str(csv), "--target", "missing"])
    assert result.exit_code != 0
    assert "missing" in result.stdout.lower() or "not found" in result.stdout.lower()
