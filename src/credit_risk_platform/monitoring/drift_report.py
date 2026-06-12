"""Generate drift reports for model input features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk_platform.features import CREDIT_FEATURES, FRAUD_FEATURES


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    buckets: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate population stability index for a numeric feature."""
    reference_values = pd.to_numeric(reference, errors="coerce").dropna()
    current_values = pd.to_numeric(current, errors="coerce").dropna()
    if reference_values.empty or current_values.empty:
        return 0.0

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(reference_values.quantile(quantiles).to_numpy())
    if len(breakpoints) <= 2:
        breakpoints = np.linspace(reference_values.min(), reference_values.max(), buckets + 1)
    if breakpoints[0] == breakpoints[-1]:
        return 0.0

    reference_counts = np.histogram(reference_values, bins=breakpoints)[0]
    current_counts = np.histogram(current_values, bins=breakpoints)[0]

    reference_pct = np.clip(reference_counts / max(reference_counts.sum(), 1), epsilon, None)
    current_pct = np.clip(current_counts / max(current_counts.sum(), 1), epsilon, None)
    return float(np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct)))


def classify_psi(psi_value: float) -> str:
    if psi_value >= 0.25:
        return "critical"
    if psi_value >= 0.10:
        return "warning"
    return "stable"


def build_drift_report(
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    features: list[str] | None = None,
) -> dict[str, object]:
    selected_features = features or sorted(set(CREDIT_FEATURES + FRAUD_FEATURES))
    report_features = []
    for feature in selected_features:
        if feature not in reference_frame.columns or feature not in current_frame.columns:
            continue
        psi_value = calculate_psi(reference_frame[feature], current_frame[feature])
        report_features.append(
            {
                "feature": feature,
                "psi": round(psi_value, 6),
                "status": classify_psi(psi_value),
            }
        )

    overall_status = "stable"
    if any(item["status"] == "critical" for item in report_features):
        overall_status = "critical"
    elif any(item["status"] == "warning" for item in report_features):
        overall_status = "warning"

    return {
        "overall_status": overall_status,
        "reference_rows": int(len(reference_frame)),
        "current_rows": int(len(current_frame)),
        "features": report_features,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/drift/drift_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)
    report = build_drift_report(reference, current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote drift report to {args.output}")


if __name__ == "__main__":
    main()
