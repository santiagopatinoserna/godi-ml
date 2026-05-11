"""TaskRecipe ABC: every task (churn, forecast, score) implements run()."""

from abc import ABC, abstractmethod

from godi_ml.core.context import RunContext


class TaskRecipe(ABC):
    """Abstract base for task-specific pipelines (churn, forecast, score)."""

    name: str = "abstract"

    @abstractmethod
    def run(self, ctx: RunContext) -> None:
        """Execute the full pipeline, mutating ctx with intermediate state.

        After run() returns successfully:
        - ctx.df is populated
        - ctx.eda_stats has summary stats
        - ctx.models has fitted models
        - ctx.predictions has held-out predictions
        - ctx.metrics has 'leaderboard' key
        - ctx.figures lists generated PNG paths
        - metrics.json + figures/ are written to ctx.output_dir
        """
