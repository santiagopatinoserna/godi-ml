"""Tests for godi_ml.config."""

import pytest

from godi_ml.config import Settings


def test_default_settings():
    """Settings should load with sensible defaults."""
    s = Settings()
    assert s.random_seed == 42
    assert s.n_jobs == -1  # use all cores
    assert s.test_size == 0.2
    assert s.val_size == 0.2
    assert s.runs_dir == "godi-ml-runs"
    assert s.lang == "en"


def test_settings_from_env(monkeypatch):
    """Settings can be overridden via env vars."""
    monkeypatch.setenv("GODI_ML_RANDOM_SEED", "123")
    monkeypatch.setenv("GODI_ML_LANG", "es")
    s = Settings()
    assert s.random_seed == 123
    assert s.lang == "es"


def test_invalid_lang_rejected():
    """Lang must be one of en, es, fr."""
    with pytest.raises(ValueError):
        Settings(lang="de")
