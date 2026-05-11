"""JSON exporter for godi-ml metrics output (versioned schema)."""

import json
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = "1.0.0"


def export_metrics_json(
    out_path: Path,
    run_id: str,
    task: str,
    target: str,
    eda_stats: dict[str, Any],
    leaderboard: list[dict[str, Any]],
    figures: list[str],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the canonical metrics.json file.

    Schema is versioned via REPORT_SCHEMA_VERSION. Downstream consumers
    (CI, MLflow bridge, Narrator agent in M3) check this version and
    raise on incompatible reads.
    """
    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "task": task,
        "target": target,
        "eda": eda_stats,
        "leaderboard": leaderboard,
        "figures": figures,
    }
    if extra:
        payload["extra"] = extra

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path
