"""Tests for godi_ml.pipeline.preprocessor."""

import numpy as np
import pandas as pd
import pytest

from godi_ml.pipeline.preprocessor import Preprocessor


def make_demo_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "num1": rng.randn(n),
            "num2": rng.randn(n) * 5 + 10,
            "cat1": rng.choice(["a", "b", "c"], size=n),
            "cat_high": [f"id_{i}" for i in range(n)],
            "target": rng.binomial(1, 0.3, size=n),
        }
    )


def test_split_proportions() -> None:
    """Stratified split must respect (1 - test - val, val, test) proportions."""
    df = make_demo_df(1000)
    pp = Preprocessor(target="target", test_size=0.2, val_size=0.2, random_seed=42)
    train, val, test = pp.split(df)
    assert abs(len(train) / len(df) - 0.6) < 0.02
    assert abs(len(val) / len(df) - 0.2) < 0.02
    assert abs(len(test) / len(df) - 0.2) < 0.02


def test_split_preserves_target_distribution() -> None:
    """Stratified split → target distribution preserved."""
    df = make_demo_df(1000)
    pp = Preprocessor(target="target", test_size=0.2, val_size=0.2, random_seed=42)
    train, val, test = pp.split(df)
    p_full = df["target"].mean()
    assert abs(train["target"].mean() - p_full) < 0.05
    assert abs(val["target"].mean() - p_full) < 0.05
    assert abs(test["target"].mean() - p_full) < 0.05


def test_fit_transform_on_train_only() -> None:
    """Scaler fits on TRAIN only — transform on val/test uses train stats."""
    df = make_demo_df(500)
    pp = Preprocessor(target="target", test_size=0.2, val_size=0.2, random_seed=42)
    train, val, test = pp.split(df)
    X_train = pp.fit_transform(train)
    X_val = pp.transform(val)
    X_test = pp.transform(test)
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    train_means = X_train.mean(axis=0)
    assert np.allclose(train_means[:2], 0, atol=0.1)  # numeric features standardized
    assert not np.allclose(X_val.mean(axis=0)[:2], 0, atol=0.01)  # val NOT centered exactly


def test_transform_before_fit_raises() -> None:
    """transform() before fit_transform() raises."""
    df = make_demo_df(50)
    pp = Preprocessor(target="target", random_seed=42)
    with pytest.raises(RuntimeError, match="not fitted"):
        pp.transform(df)


def test_high_cardinality_excluded() -> None:
    """Columns where every value is unique are dropped (treated as IDs)."""
    df = make_demo_df(100)
    pp = Preprocessor(target="target", random_seed=42)
    train, _, _ = pp.split(df)
    pp.fit_transform(train)
    assert "cat_high" not in pp.feature_columns_


def test_target_excluded_from_features() -> None:
    """Target column must not appear in transformed feature matrix."""
    df = make_demo_df(50)
    pp = Preprocessor(target="target", random_seed=42)
    train, _, _ = pp.split(df)
    X = pp.fit_transform(train)
    assert "target" not in pp.feature_columns_
    assert X.shape[1] == len(pp.feature_columns_)


def test_imputes_missing_numeric() -> None:
    """Missing numeric values are filled (median by default)."""
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    pp = Preprocessor(target="target", random_seed=42)
    train, _, _ = pp.split(df)
    X = pp.fit_transform(train)
    assert not np.isnan(X).any()
