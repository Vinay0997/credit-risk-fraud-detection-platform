"""Optional MLflow tracking helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def log_model_run(
    model_name: str,
    model: Any,
    metrics: dict[str, Any],
    artifact_dir: str | Path,
) -> bool:
    """Log a model run to MLflow when MLflow is installed and configured."""
    try:
        import mlflow
        import mlflow.sklearn
    except Exception:
        return False

    with mlflow.start_run(run_name=model_name):
        flat_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if flat_metrics:
            mlflow.log_metrics(flat_metrics)
        mlflow.log_artifacts(str(artifact_dir))
        mlflow.sklearn.log_model(model, artifact_path=model_name)
    return True
