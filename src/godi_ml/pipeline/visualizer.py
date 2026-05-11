"""Visualizer: save ROC, PR, calibration, confusion matrix, feature importance PNGs."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for CI

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

_DPI = 120


class Visualizer:
    """Static methods that save matplotlib figures to disk as PNG."""

    @staticmethod
    def save_roc_curve(
        y_true: np.ndarray,
        preds: dict[str, np.ndarray],
        out: Path,
        title: str = "ROC Curve",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, proba in preds.items():
            fpr, tpr, _ = roc_curve(y_true, proba)
            ax.plot(fpr, tpr, label=name)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=_DPI)
        plt.close(fig)
        return out

    @staticmethod
    def save_pr_curve(
        y_true: np.ndarray,
        preds: dict[str, np.ndarray],
        out: Path,
        title: str = "Precision-Recall Curve",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, proba in preds.items():
            precision, recall, _ = precision_recall_curve(y_true, proba)
            ax.plot(recall, precision, label=name)
        baseline = float(np.mean(y_true))
        ax.axhline(
            baseline,
            color="k",
            linestyle="--",
            alpha=0.4,
            label=f"baseline = {baseline:.2f}",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=_DPI)
        plt.close(fig)
        return out

    @staticmethod
    def save_calibration_plot(
        y_true: np.ndarray,
        preds: dict[str, np.ndarray],
        out: Path,
        n_bins: int = 10,
        title: str = "Calibration (Reliability Diagram)",
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, proba in preds.items():
            prob_true, prob_pred = calibration_curve(
                y_true, proba, n_bins=n_bins, strategy="quantile"
            )
            ax.plot(prob_pred, prob_true, marker="o", label=name)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical fraction of positives")
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=_DPI)
        plt.close(fig)
        return out

    @staticmethod
    def save_confusion_matrix(
        cm: Sequence[Sequence[int]],
        out: Path,
        title: str = "Confusion Matrix",
        labels: Sequence[str] = ("0", "1"),
    ) -> Path:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(labels),
            yticklabels=list(labels),
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out, dpi=_DPI)
        plt.close(fig)
        return out

    @staticmethod
    def save_feature_importance(
        importances: dict[str, float],
        out: Path,
        top_k: int = 20,
        title: str = "Feature Importance",
    ) -> Path:
        items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        names, values = zip(*items, strict=True) if items else ((), ())
        fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(names))))
        ax.barh(list(reversed(names)), list(reversed(values)))
        ax.set_xlabel("Importance")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out, dpi=_DPI)
        plt.close(fig)
        return out
