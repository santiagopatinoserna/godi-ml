"""godi-ml CLI entry point (typer)."""

import secrets
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from godi_ml import __version__
from godi_ml.config import Settings
from godi_ml.core.context import RunContext
from godi_ml.tasks.churn import ChurnRecipe

app = typer.Typer(
    name="godi-ml",
    help="ML quality assurance via LLM sub-agents — catch what AutoML doesn't catch.",
    no_args_is_help=True,
)
console = Console()


def _make_run_id() -> str:
    return secrets.token_hex(4)


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", help="Show version and exit")] = False,
) -> None:
    if version:
        console.print(f"godi-ml {__version__}")
        raise typer.Exit()


@app.command()
def churn(
    data: Annotated[Path, typer.Option("--data", "-d", exists=True, help="Path to CSV/Parquet")],
    target: Annotated[str, typer.Option("--target", "-t", help="Name of target column")],
    runs_dir: Annotated[Path, typer.Option("--runs-dir", help="Output base directory")] = Path(
        "godi-ml-runs"
    ),
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    n_jobs: Annotated[int, typer.Option("--n-jobs", help="Parallel jobs (-1 = all cores)")] = -1,
) -> None:
    """Train binary classification models and evaluate them — churn use case."""
    settings = Settings()
    run_id = _make_run_id()
    output_dir = runs_dir / run_id

    console.print(f"[bold cyan][godi-ml v{__version__}] Starting run {run_id}[/bold cyan]")
    console.print(f"  Data:    {data}")
    console.print(f"  Target:  {target}")
    console.print(f"  Output:  {output_dir}")

    ctx = RunContext(
        run_id=run_id,
        task="churn",
        target=target,
        data_path=data,
        output_dir=output_dir,
        lang=settings.lang,
        random_seed=seed,
        n_jobs=n_jobs,
    )

    try:
        ChurnRecipe().run(ctx)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    leaderboard = ctx.metrics.get("leaderboard", [])
    console.print("\n[bold]Leaderboard:[/bold]")
    for row in leaderboard:
        marker = "← best" if row["is_best"] else ""
        console.print(
            f"  {row['model']:25s}  AUC={row['auc_roc']:.3f}  "
            f"Brier={row['brier']:.3f}  {marker}"
        )

    console.print(f"\n[green]✓ Report:[/green] {output_dir}/metrics.json")
    console.print(f"[green]✓ Figures:[/green] {output_dir}/figures/")
