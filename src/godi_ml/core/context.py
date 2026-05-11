"""Shared run context passed between pipeline stages."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

TaskType = Literal["churn", "forecast", "score"]
LangType = Literal["en", "es", "fr"]


@dataclass
class RunContext:
    """Shared mutable state for a single pipeline run.

    Each pipeline stage reads required inputs and appends its outputs.
    Stages should never overwrite outputs of other stages (append-only).
    """

    run_id: str
    task: TaskType
    target: str
    data_path: Path
    output_dir: Path
    lang: LangType = "en"
    random_seed: int = 42
    n_jobs: int = -1

    df: pd.DataFrame | None = None
    df_train: pd.DataFrame | None = None
    df_val: pd.DataFrame | None = None
    df_test: pd.DataFrame | None = None
    feature_columns: list[str] = field(default_factory=list)

    eda_stats: dict[str, Any] = field(default_factory=dict)
    preprocessor: Any = None
    models: dict[str, Any] = field(default_factory=dict)
    predictions: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    figures: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate task is recognized."""
        valid_tasks = ("churn", "forecast", "score")
        if self.task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}, got {self.task!r}")
