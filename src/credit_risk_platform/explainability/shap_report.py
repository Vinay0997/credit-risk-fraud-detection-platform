"""SHAP-ready model explanations with a deterministic fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _feature_importance_fallback(model: Any, feature_names: list[str]) -> np.ndarray:
    estimator = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        importances = np.ones(len(feature_names)) / len(feature_names)
    importances = np.asarray(importances, dtype=float)
    if importances.sum() == 0:
        return np.ones(len(feature_names)) / len(feature_names)
    return importances / importances.sum()


def explain_records(
    model: Any,
    frame: pd.DataFrame,
    feature_names: list[str],
    max_rows: int = 100,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Return local explanations for scored records.

    If SHAP is installed and supports the estimator, SHAP values are used. Otherwise
    the function returns a stable approximation based on normalized feature values
    and global feature importance.
    """
    sample = frame[feature_names].head(max_rows).copy()
    values: np.ndarray | None = None
    method = "importance_fallback"

    try:
        import shap

        transformed = model.named_steps["imputer"].transform(sample)
        estimator = model.named_steps["model"]
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed)
        values = shap_values[1] if isinstance(shap_values, list) else shap_values
        method = "shap"
    except Exception:
        importances = _feature_importance_fallback(model, feature_names)
        numeric_sample = sample.astype(float)
        centered = numeric_sample - numeric_sample.mean(axis=0)
        scaled = centered / (numeric_sample.std(axis=0).replace(0, 1))
        values = scaled.to_numpy() * importances

    explanations: list[dict[str, Any]] = []
    for row_index, row_values in enumerate(np.asarray(values)):
        ranked = sorted(
            zip(feature_names, row_values, strict=True),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:top_n]
        explanations.append(
            {
                "row_number": int(row_index),
                "method": method,
                "top_features": [
                    {
                        "feature": feature,
                        "contribution": round(float(contribution), 6),
                    }
                    for feature, contribution in ranked
                ],
            }
        )
    return explanations
