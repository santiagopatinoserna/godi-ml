"""Tests for godi_ml.reporting.json_export."""

import json
from pathlib import Path

from godi_ml.reporting.json_export import REPORT_SCHEMA_VERSION, export_metrics_json


def test_export_basic(tmp_path: Path) -> None:
    """export_metrics_json writes valid JSON with versioned schema."""
    eda_stats = {"n_rows": 100, "n_cols": 5}
    leaderboard = [{"model": "xgb", "auc_roc": 0.85, "is_best": True}]
    out_path = tmp_path / "metrics.json"
    export_metrics_json(
        out_path=out_path,
        run_id="abc",
        task="churn",
        target="churned",
        eda_stats=eda_stats,
        leaderboard=leaderboard,
        figures=["roc.png", "pr.png"],
    )
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["schema_version"] == REPORT_SCHEMA_VERSION
    assert data["run_id"] == "abc"
    assert data["task"] == "churn"
    assert data["target"] == "churned"
    assert data["eda"] == eda_stats
    assert data["leaderboard"] == leaderboard
    assert "roc.png" in data["figures"]


def test_export_creates_parent_dir(tmp_path: Path) -> None:
    """Parent directories created if missing."""
    out = tmp_path / "nested" / "dir" / "metrics.json"
    export_metrics_json(
        out_path=out,
        run_id="x",
        task="churn",
        target="t",
        eda_stats={},
        leaderboard=[],
        figures=[],
    )
    assert out.exists()
