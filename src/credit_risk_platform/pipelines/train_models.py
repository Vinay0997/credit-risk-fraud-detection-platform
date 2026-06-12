"""Train credit risk and fraud models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from credit_risk_platform.lifecycle.mlflow_registry import log_model_run
from credit_risk_platform.models.credit_model import train_credit_model
from credit_risk_platform.models.fraud_model import train_fraud_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data)

    model_dir = args.artifacts / "models"
    metrics_dir = args.artifacts / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    credit = train_credit_model(frame, random_seed=args.seed)
    fraud = train_fraud_model(frame, random_seed=args.seed)

    joblib.dump(credit.pipeline, model_dir / "credit_model.joblib")
    joblib.dump(fraud.pipeline, model_dir / "fraud_model.joblib")
    write_json(metrics_dir / "credit_metrics.json", credit.metrics)
    write_json(metrics_dir / "fraud_metrics.json", fraud.metrics)

    credit_logged = log_model_run("credit_risk_model", credit.pipeline, credit.metrics, metrics_dir)
    fraud_logged = log_model_run("fraud_detection_model", fraud.pipeline, fraud.metrics, metrics_dir)

    print(f"Saved credit model to {model_dir / 'credit_model.joblib'}")
    print(f"Saved fraud model to {model_dir / 'fraud_model.joblib'}")
    print(f"MLflow logging: credit={credit_logged}, fraud={fraud_logged}")


if __name__ == "__main__":
    main()
