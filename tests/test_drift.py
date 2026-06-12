import pandas as pd

from credit_risk_platform.monitoring.drift_report import build_drift_report


def test_build_drift_report_returns_feature_status():
    reference = pd.DataFrame({"bureau_score": [650, 660, 670, 680, 690], "age": [30, 35, 40, 45, 50]})
    current = pd.DataFrame({"bureau_score": [640, 655, 665, 675, 720], "age": [31, 36, 41, 46, 51]})
    report = build_drift_report(reference, current, features=["bureau_score", "age"])
    assert report["overall_status"] in {"stable", "warning", "critical"}
    assert len(report["features"]) == 2
