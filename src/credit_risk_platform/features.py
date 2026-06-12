"""Shared feature definitions and scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

CREDIT_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "bureau_score",
    "credit_utilization",
    "debt_to_income",
    "delinquencies_30d",
    "avg_monthly_spend",
    "cash_advance_ratio",
    "device_risk_score",
]

FRAUD_FEATURES = [
    "transaction_amount",
    "merchant_risk_score",
    "device_risk_score",
    "login_velocity_1h",
    "geo_velocity_km",
    "cash_advance_ratio",
    "account_age_days",
    "hour_of_day",
]

IDENTIFIER_COLUMNS = ["customer_id", "transaction_id", "event_timestamp"]


@dataclass(frozen=True)
class RiskBand:
    name: str
    lower_bound: float
    upper_bound: float


RISK_BANDS = [
    RiskBand("low", 0.0, 0.20),
    RiskBand("medium", 0.20, 0.35),
    RiskBand("high", 0.35, 0.60),
    RiskBand("critical", 0.60, 1.01),
]


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def assign_risk_band(probability: float) -> str:
    """Map a probability to a business-friendly risk band."""
    for band in RISK_BANDS:
        if band.lower_bound <= probability < band.upper_bound:
            return band.name
    return "unknown"


def score_to_priority(probability: float) -> str:
    """Map a fraud probability to an operations priority."""
    if probability >= 0.80:
        return "critical"
    if probability >= 0.65:
        return "high"
    if probability >= 0.40:
        return "medium"
    return "low"
