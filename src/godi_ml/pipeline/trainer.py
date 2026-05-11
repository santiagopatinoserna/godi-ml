"""Trainer: fit 5 default models in parallel for binary classification."""

from typing import Any

import numpy as np
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

MODEL_NAMES: tuple[str, ...] = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
)


def _build_models(random_seed: int) -> dict[str, Any]:
    """Construct unfitted models with sensible defaults."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_seed,
            n_jobs=1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=random_seed,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            random_state=random_seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_seed,
            n_jobs=1,
            eval_metric="logloss",
            verbosity=0,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200,
            max_depth=-1,
            learning_rate=0.1,
            random_state=random_seed,
            n_jobs=1,
            verbosity=-1,
        ),
    }


class Trainer:
    """Trains 5 default binary classification models in parallel."""

    def __init__(self, random_seed: int = 42, n_jobs: int = -1) -> None:
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.models_: dict[str, Any] = {}
        self._fitted = False

    @staticmethod
    def _fit_one(name: str, model: Any, X: np.ndarray, y: np.ndarray) -> tuple[str, Any]:
        model.fit(X, y)
        return name, model

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Train all models in parallel. Returns name → fitted model."""
        unfit = _build_models(self.random_seed)
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_one)(name, model, X, y) for name, model in unfit.items()
        )
        self.models_ = dict(results)
        self._fitted = True
        return self.models_

    def predict_proba_all(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return dict name → (n_samples, n_classes) probability array."""
        if not self._fitted:
            raise RuntimeError("Trainer not fitted. Call fit() first.")
        return {name: model.predict_proba(X) for name, model in self.models_.items()}
