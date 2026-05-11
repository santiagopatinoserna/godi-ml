"""Tests for godi_ml.tasks.churn (ChurnRecipe)."""

from pathlib import Path

import numpy as np
import pandas as pd

from godi_ml.core.context import RunContext
from godi_ml.tasks.churn import ChurnRecipe


def make_synthetic_csv(path: Path, n: int = 500) -> None:
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "age": rng.randint(18, 80, n),
            "income": rng.randn(n) * 1000 + 50_000,
            "country": rng.choice(["US", "FR", "ES"], n),
            "tenure": rng.randint(0, 60, n),
            "churned": rng.binomial(1, 0.25, n),
        }
    )
    df.to_csv(path, index=False)


def test_churn_recipe_runs_e2e(tmp_path: Path) -> None:
    """ChurnRecipe runs full pipeline and writes JSON + figures."""
    csv = tmp_path / "data.csv"
    out = tmp_path / "run_out"
    make_synthetic_csv(csv)

    ctx = RunContext(
        run_id="test_run",
        task="churn",
        target="churned",
        data_path=csv,
        output_dir=out,
        random_seed=42,
        n_jobs=1,
    )

    recipe = ChurnRecipe()
    recipe.run(ctx)

    assert (out / "metrics.json").exists()
    figures_dir = out / "figures"
    assert figures_dir.exists()
    assert any(figures_dir.glob("*.png"))
    assert "leaderboard" in ctx.metrics
    assert len(ctx.metrics["leaderboard"]) == 5  # 5 models
