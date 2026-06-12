"""Generate a synthetic credit risk and fraud dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_synthetic_risk_events(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create synthetic records with credit and fraud labels."""
    rng = np.random.default_rng(seed)

    customer_id = rng.integers(100000, 999999, size=rows)
    transaction_id = [f"txn_{seed}_{index:07d}" for index in range(rows)]
    event_timestamp = pd.Timestamp("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 180 * 24 * 60, size=rows),
        unit="m",
    )

    age = np.clip(rng.normal(42, 12, size=rows), 18, 78).round()
    annual_income = np.clip(rng.lognormal(mean=11.0, sigma=0.45, size=rows), 18000, 250000)
    employment_years = np.clip(rng.gamma(shape=3.0, scale=2.0, size=rows), 0, 35)
    bureau_score = np.clip(rng.normal(690, 75, size=rows), 420, 850)
    credit_utilization = np.clip(rng.beta(2.2, 3.0, size=rows), 0, 1)
    debt_to_income = np.clip(rng.normal(0.35, 0.16, size=rows), 0.03, 0.95)
    delinquencies_30d = rng.poisson(lam=np.clip((700 - bureau_score) / 180, 0.05, 2.5))
    avg_monthly_spend = np.clip(annual_income / 18 * rng.normal(1.0, 0.25, size=rows), 100, 20000)
    cash_advance_ratio = np.clip(rng.beta(1.2, 8.0, size=rows), 0, 1)
    device_risk_score = np.clip(rng.beta(1.5, 5.0, size=rows), 0, 1)
    account_age_days = rng.integers(10, 3650, size=rows)

    transaction_amount = np.clip(rng.lognormal(mean=4.8, sigma=1.1, size=rows), 5, 25000)
    merchant_risk_score = np.clip(rng.beta(1.4, 4.0, size=rows), 0, 1)
    login_velocity_1h = rng.poisson(lam=1.5 + 6.0 * device_risk_score)
    geo_velocity_km = np.clip(rng.exponential(scale=120, size=rows), 0, 5000)
    hour_of_day = rng.integers(0, 24, size=rows)

    default_logit = (
        -3.0
        + 2.2 * credit_utilization
        + 1.9 * debt_to_income
        + 0.45 * delinquencies_30d
        - 0.006 * (bureau_score - 650)
        - 0.000006 * (annual_income - 60000)
        - 0.04 * employment_years
        + 0.70 * cash_advance_ratio
        + 0.45 * device_risk_score
    )
    default_probability = _sigmoid(default_logit)
    default_label = rng.binomial(1, default_probability)

    night_indicator = ((hour_of_day <= 5) | (hour_of_day >= 23)).astype(int)
    fraud_logit = (
        -4.2
        + 2.8 * merchant_risk_score
        + 2.4 * device_risk_score
        + 0.12 * login_velocity_1h
        + 0.00035 * transaction_amount
        + 0.00045 * geo_velocity_km
        + 0.50 * night_indicator
        - 0.00018 * account_age_days
    )
    fraud_probability = _sigmoid(fraud_logit)
    fraud_label = rng.binomial(1, fraud_probability)

    frame = pd.DataFrame(
        {
            "customer_id": customer_id,
            "transaction_id": transaction_id,
            "event_timestamp": event_timestamp,
            "age": age.astype(int),
            "annual_income": annual_income.round(2),
            "employment_years": employment_years.round(2),
            "bureau_score": bureau_score.round(0).astype(int),
            "credit_utilization": credit_utilization.round(4),
            "debt_to_income": debt_to_income.round(4),
            "delinquencies_30d": delinquencies_30d.astype(int),
            "avg_monthly_spend": avg_monthly_spend.round(2),
            "cash_advance_ratio": cash_advance_ratio.round(4),
            "device_risk_score": device_risk_score.round(4),
            "account_age_days": account_age_days.astype(int),
            "transaction_amount": transaction_amount.round(2),
            "merchant_risk_score": merchant_risk_score.round(4),
            "login_velocity_1h": login_velocity_1h.astype(int),
            "geo_velocity_km": geo_velocity_km.round(2),
            "hour_of_day": hour_of_day.astype(int),
            "default_probability_true": default_probability.round(4),
            "fraud_probability_true": fraud_probability.round(4),
            "default_label": default_label.astype(int),
            "fraud_label": fraud_label.astype(int),
        }
    )
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000, help="Number of synthetic rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out", type=Path, default=Path("data/processed/risk_events.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = generate_synthetic_risk_events(rows=args.rows, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame):,} synthetic events to {args.out}")


if __name__ == "__main__":
    main()
