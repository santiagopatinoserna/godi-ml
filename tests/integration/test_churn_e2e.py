"""End-to-end integration test for godi-ml churn on IBM Telco fixture."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from godi_ml.cli import app


@pytest.mark.integration
def test_churn_e2e_on_telco(tmp_path: Path, telco_churn_path: Path) -> None:
    """Full pipeline runs on real IBM Telco data, produces report artifacts."""
    runner = CliRunner()
    runs_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "churn",
            "--data",
            str(telco_churn_path),
            "--target",
            "Churn",
            "--runs-dir",
            str(runs_dir),
            "--n-jobs",
            "1",
        ],
    )
    assert result.exit_code == 0, result.stdout

    run_dirs = list(runs_dir.iterdir())
    assert len(run_dirs) == 1
    out = run_dirs[0]

    metrics_path = out / "metrics.json"
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text())
    assert data["task"] == "churn"
    assert data["target"] == "Churn"

    leaderboard = data["leaderboard"]
    assert len(leaderboard) == 5
    for row in leaderboard:
        assert 0.5 < row["auc_roc"] < 1.0  # at minimum better than random

    figures_dir = out / "figures"
    assert figures_dir.exists()
    pngs = list(figures_dir.glob("*.png"))
    assert len(pngs) >= 4  # roc, pr, calibration, confusion
