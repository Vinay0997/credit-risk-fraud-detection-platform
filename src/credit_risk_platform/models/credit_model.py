"""Credit risk model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from credit_risk_platform.features import CREDIT_FEATURES, require_columns


@dataclass
class TrainedModel:
    pipeline: Pipeline
    metrics: dict[str, Any]
    feature_names: list[str]


def _build_credit_estimator(random_seed: int) -> Any:
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_seed,
        )
    except Exception:
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=24,
            random_state=random_seed,
        )


def train_credit_model(
    frame: pd.DataFrame,
    target: str = "default_label",
    random_seed: int = 42,
) -> TrainedModel:
    """Train a default risk classifier."""
    require_columns(frame, CREDIT_FEATURES + [target])

    x_train, x_test, y_train, y_test = train_test_split(
        frame[CREDIT_FEATURES],
        frame[target],
        test_size=0.25,
        random_state=random_seed,
        stratify=frame[target],
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", _build_credit_estimator(random_seed)),
        ]
    )
    pipeline.fit(x_train, y_train)

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.35).astype(int)
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
    return TrainedModel(pipeline=pipeline, metrics=metrics, feature_names=CREDIT_FEATURES)
