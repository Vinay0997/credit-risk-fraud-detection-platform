"""Fraud model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from credit_risk_platform.features import FRAUD_FEATURES, require_columns


@dataclass
class TrainedFraudModel:
    pipeline: Pipeline
    metrics: dict[str, Any]
    feature_names: list[str]


def _build_fraud_estimator(random_seed: int) -> Any:
    try:
        import torch  # noqa: F401

        return MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            alpha=0.001,
            max_iter=250,
            random_state=random_seed,
        )
    except Exception:
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.06,
            max_leaf_nodes=20,
            random_state=random_seed,
        )


def train_fraud_model(
    frame: pd.DataFrame,
    target: str = "fraud_label",
    random_seed: int = 42,
) -> TrainedFraudModel:
    """Train a fraud classifier."""
    require_columns(frame, FRAUD_FEATURES + [target])

    x_train, x_test, y_train, y_test = train_test_split(
        frame[FRAUD_FEATURES],
        frame[target],
        test_size=0.25,
        random_state=random_seed,
        stratify=frame[target],
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", _build_fraud_estimator(random_seed)),
        ]
    )
    pipeline.fit(x_train, y_train)

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.65).astype(int)
    metrics = {
        "model_type": pipeline.named_steps["model"].__class__.__name__,
        "target": target,
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "positive_rate_test": float(np.mean(y_test)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }
    return TrainedFraudModel(pipeline=pipeline, metrics=metrics, feature_names=FRAUD_FEATURES)
