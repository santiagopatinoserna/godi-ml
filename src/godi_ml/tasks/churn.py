"""ChurnRecipe: end-to-end binary classification pipeline for churn."""

from godi_ml.core.context import RunContext
from godi_ml.pipeline.eda import compute_eda
from godi_ml.pipeline.evaluator import Evaluator
from godi_ml.pipeline.loader import DataLoader
from godi_ml.pipeline.preprocessor import Preprocessor
from godi_ml.pipeline.trainer import Trainer
from godi_ml.pipeline.visualizer import Visualizer
from godi_ml.reporting.json_export import export_metrics_json
from godi_ml.tasks.base import TaskRecipe


class ChurnRecipe(TaskRecipe):
    """Binary classification pipeline for customer churn use cases."""

    name = "churn"

    def run(self, ctx: RunContext) -> None:
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = ctx.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        loader = DataLoader(target=ctx.target)
        ctx.df = loader.load(ctx.data_path)

        ctx.eda_stats = compute_eda(ctx.df, target=ctx.target, random_seed=ctx.random_seed)

        pp = Preprocessor(target=ctx.target, random_seed=ctx.random_seed)
        ctx.df_train, ctx.df_val, ctx.df_test = pp.split(ctx.df)
        X_train = pp.fit_transform(ctx.df_train)
        X_test = pp.transform(ctx.df_test)
        y_train = pp.y(ctx.df_train)
        y_test = pp.y(ctx.df_test)
        ctx.preprocessor = pp
        ctx.feature_columns = pp.feature_columns_

        trainer = Trainer(random_seed=ctx.random_seed, n_jobs=ctx.n_jobs)
        ctx.models = trainer.fit(X_train, y_train)
        proba_all = trainer.predict_proba_all(X_test)
        ctx.predictions = {name: proba[:, 1] for name, proba in proba_all.items()}

        per_model_metrics = {
            name: Evaluator.compute_binary_metrics(y_test, proba)
            for name, proba in ctx.predictions.items()
        }
        leaderboard = Evaluator.build_leaderboard(per_model_metrics, primary="auc_roc")
        ctx.metrics = {
            "per_model": per_model_metrics,
            "leaderboard": leaderboard,
        }

        roc_path = Visualizer.save_roc_curve(y_test, ctx.predictions, figures_dir / "roc.png")
        pr_path = Visualizer.save_pr_curve(y_test, ctx.predictions, figures_dir / "pr.png")
        calib_path = Visualizer.save_calibration_plot(
            y_test, ctx.predictions, figures_dir / "calibration.png"
        )
        ctx.figures = [roc_path, pr_path, calib_path]

        best_name = leaderboard[0]["model"]
        best_cm = leaderboard[0]["confusion_matrix"]
        cm_path = Visualizer.save_confusion_matrix(
            best_cm, figures_dir / f"confusion_{best_name}.png", title=f"Confusion — {best_name}"
        )
        ctx.figures.append(cm_path)

        best_model = ctx.models[best_name]
        if hasattr(best_model, "feature_importances_"):
            fi = dict(zip(ctx.feature_columns, best_model.feature_importances_, strict=False))
            fi_path = Visualizer.save_feature_importance(
                fi, figures_dir / f"importance_{best_name}.png"
            )
            ctx.figures.append(fi_path)

        export_metrics_json(
            out_path=ctx.output_dir / "metrics.json",
            run_id=ctx.run_id,
            task=ctx.task,
            target=ctx.target,
            eda_stats=ctx.eda_stats,
            leaderboard=leaderboard,
            figures=[p.name for p in ctx.figures],
        )
