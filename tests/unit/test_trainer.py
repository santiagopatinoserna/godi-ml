"""Tests for godi_ml.pipeline.trainer."""

import numpy as np
import pytest

from godi_ml.pipeline.trainer import MODEL_NAMES, Trainer


def make_synthetic_xy(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_trainer_trains_all_models() -> None:
    """Trainer fits all 5 default models."""
    X, y = make_synthetic_xy(500)
    trainer = Trainer(random_seed=42, n_jobs=1)
    models = trainer.fit(X, y)
    assert set(models.keys()) == set(MODEL_NAMES)
    for m in models.values():
        assert hasattr(m, "predict_proba")


def test_trainer_predict_proba_shape() -> None:
    """predict_proba_all returns dict of (n_samples, 2) arrays."""
    X, y = make_synthetic_xy(300)
    trainer = Trainer(random_seed=42, n_jobs=1)
    trainer.fit(X, y)
    preds = trainer.predict_proba_all(X)
    assert set(preds.keys()) == set(MODEL_NAMES)
    for arr in preds.values():
        assert arr.shape == (300, 2)
        assert np.allclose(arr.sum(axis=1), 1.0, atol=1e-5)


def test_trainer_deterministic() -> None:
    """Same seed → same predictions."""
    X, y = make_synthetic_xy(200)
    t1 = Trainer(random_seed=42, n_jobs=1)
    t1.fit(X, y)
    t2 = Trainer(random_seed=42, n_jobs=1)
    t2.fit(X, y)
    p1 = t1.predict_proba_all(X)
    p2 = t2.predict_proba_all(X)
    for name in MODEL_NAMES:
        np.testing.assert_array_equal(p1[name], p2[name])


def test_predict_before_fit_raises() -> None:
    """predict_proba_all before fit() raises."""
    trainer = Trainer(random_seed=42)
    X = np.zeros((10, 3))
    with pytest.raises(RuntimeError, match="not fitted"):
        trainer.predict_proba_all(X)
