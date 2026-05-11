"""Shared pytest fixtures for godi-ml tests."""

from pathlib import Path

import pytest


@pytest.fixture
def telco_churn_path() -> Path:
    """Path to the IBM Telco churn fixture (1000-row deterministic slice)."""
    return Path(__file__).parent / "fixtures" / "telco_churn.csv"
