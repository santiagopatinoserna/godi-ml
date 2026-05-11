"""Tests for godi_ml.pipeline.visualizer."""

from pathlib import Path

import numpy as np
import pytest

from godi_ml.pipeline.visualizer import Visualizer


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 200)
    preds = {
        "model_a": rng.uniform(0, 1, 200),
        "model_b": rng.uniform(0, 1, 200),
    }
    return y_true, preds


def test_save_roc_curve(tmp_path: Path, synthetic_data) -> None:
    y_true, preds = synthetic_data
    out = Visualizer.save_roc_curve(y_true, preds, tmp_path / "roc.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_pr_curve(tmp_path: Path, synthetic_data) -> None:
    y_true, preds = synthetic_data
    out = Visualizer.save_pr_curve(y_true, preds, tmp_path / "pr.png")
    assert out.exists()


def test_save_calibration(tmp_path: Path, synthetic_data) -> None:
    y_true, preds = synthetic_data
    out = Visualizer.save_calibration_plot(y_true, preds, tmp_path / "calib.png")
    assert out.exists()


def test_save_confusion_matrix(tmp_path: Path) -> None:
    cm = [[80, 20], [15, 85]]
    out = Visualizer.save_confusion_matrix(cm, tmp_path / "cm.png", title="Test Model")
    assert out.exists()


def test_save_feature_importance(tmp_path: Path) -> None:
    importances = {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2}
    out = Visualizer.save_feature_importance(importances, tmp_path / "fi.png")
    assert out.exists()
