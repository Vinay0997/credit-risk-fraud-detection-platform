from credit_risk_platform.data.make_dataset import generate_synthetic_risk_events
from credit_risk_platform.features import CREDIT_FEATURES, FRAUD_FEATURES


def test_generate_synthetic_dataset_has_required_columns():
    frame = generate_synthetic_risk_events(rows=100, seed=7)
    required = set(CREDIT_FEATURES + FRAUD_FEATURES + ["default_label", "fraud_label"])
    assert required.issubset(frame.columns)
    assert len(frame) == 100
    assert frame["default_label"].isin([0, 1]).all()
    assert frame["fraud_label"].isin([0, 1]).all()
