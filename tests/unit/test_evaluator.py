"""Tests for godi_ml.pipeline.evaluator."""

import numpy as np

from godi_ml.pipeline.evaluator import Evaluator


def test_compute_metrics_keys() -> None:
    """All expected metric keys are returned."""
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 200)
    y_proba = rng.uniform(0, 1, 200)
    metrics = Evaluator.compute_binary_metrics(y_true, y_proba)
    expected = {
        "auc_roc",
        "auc_pr",
        "brier",
        "log_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
    }
    assert expected.issubset(metrics.keys())


def test_metrics_perfect_predictions() -> None:
    """Perfect predictions → AUC=1.0, Brier=0.0."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.05, 0.95, 1.0])
    metrics = Evaluator.compute_binary_metrics(y_true, y_proba)
    assert metrics["auc_roc"] == 1.0
    assert metrics["brier"] < 0.01


def test_metrics_random_predictions() -> None:
    """Random predictions → AUC ≈ 0.5."""
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 500)
    y_proba = rng.uniform(0, 1, 500)
    metrics = Evaluator.compute_binary_metrics(y_true, y_proba)
    assert 0.4 < metrics["auc_roc"] < 0.6


def test_leaderboard_orders_by_auc() -> None:
    """build_leaderboard sorts models by AUC descending."""
    metrics_per_model = {
        "model_a": {"auc_roc": 0.75, "brier": 0.20},
        "model_b": {"auc_roc": 0.85, "brier": 0.15},
        "model_c": {"auc_roc": 0.65, "brier": 0.25},
    }
    leaderboard = Evaluator.build_leaderboard(metrics_per_model, primary="auc_roc")
    assert leaderboard[0]["model"] == "model_b"
    assert leaderboard[-1]["model"] == "model_c"


def test_leaderboard_marks_best() -> None:
    """Top model has is_best=True."""
    metrics_per_model = {
        "a": {"auc_roc": 0.75},
        "b": {"auc_roc": 0.90},
    }
    lb = Evaluator.build_leaderboard(metrics_per_model, primary="auc_roc")
    assert lb[0]["is_best"] is True
    assert lb[1]["is_best"] is False
