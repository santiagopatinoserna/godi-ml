"""Evaluator: compute binary classification metrics + build leaderboard."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Evaluator:
    """Static methods for binary classification metrics + leaderboard."""

    @staticmethod
    def compute_binary_metrics(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Compute the canonical binary classification metric bundle.

        Returns dict with: auc_roc, auc_pr, brier, log_loss, accuracy,
        precision, recall, f1, confusion_matrix (2x2 list of lists).
        """
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        return {
            "auc_roc": float(roc_auc_score(y_true, y_proba)),
            "auc_pr": float(average_precision_score(y_true, y_proba)),
            "brier": float(brier_score_loss(y_true, y_proba)),
            "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-15, 1 - 1e-15))),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": cm.tolist(),
        }

    @staticmethod
    def build_leaderboard(
        metrics_per_model: dict[str, dict[str, Any]],
        primary: str = "auc_roc",
    ) -> list[dict[str, Any]]:
        """Sort models by primary metric descending and mark the top as best."""
        rows = [
            {"model": name, **metrics, "is_best": False}
            for name, metrics in metrics_per_model.items()
        ]
        rows.sort(key=lambda r: r[primary], reverse=True)
        if rows:
            rows[0]["is_best"] = True
        return rows
