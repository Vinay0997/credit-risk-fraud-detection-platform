"""Score a batch of events with trained models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from credit_risk_platform.explainability.shap_report import explain_records
from credit_risk_platform.features import (
    CREDIT_FEATURES,
    FRAUD_FEATURES,
    IDENTIFIER_COLUMNS,
    assign_risk_band,
    require_columns,
    score_to_priority,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/scoring/scored_events.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    require_columns(frame, CREDIT_FEATURES + FRAUD_FEATURES)

    model_dir = args.artifacts / "models"
    credit_model = joblib.load(model_dir / "credit_model.joblib")
    fraud_model = joblib.load(model_dir / "fraud_model.joblib")

    scored = frame[[column for column in IDENTIFIER_COLUMNS if column in frame.columns]].copy()
    scored["credit_default_score"] = credit_model.predict_proba(frame[CREDIT_FEATURES])[:, 1]
    scored["credit_risk_band"] = scored["credit_default_score"].map(assign_risk_band)
    scored["fraud_score"] = fraud_model.predict_proba(frame[FRAUD_FEATURES])[:, 1]
    scored["fraud_priority"] = scored["fraud_score"].map(score_to_priority)

    credit_explanations = explain_records(credit_model, frame, CREDIT_FEATURES, max_rows=len(frame))
    fraud_explanations = explain_records(fraud_model, frame, FRAUD_FEATURES, max_rows=len(frame))
    scored["credit_explanation"] = [json.dumps(item["top_features"]) for item in credit_explanations]
    scored["fraud_explanation"] = [json.dumps(item["top_features"]) for item in fraud_explanations]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(f"Wrote scored records to {args.output}")


if __name__ == "__main__":
    main()
