# Sample Fraud Operations Control Standard

Fraud alerts should be prioritized by risk score, transaction value, customer impact, and observed behavioral anomalies.

Alert queues must include enough evidence for analyst review, including transaction amount, merchant risk, device risk, login velocity, geolocation velocity, and model explanation.

False positives must be measured and reviewed. When false-positive rates increase materially, fraud analytics should review thresholds, labels, and feature drift.

High-priority alerts should receive human review before customer-impacting actions when required by policy.

Investigation outcomes should be captured as feedback labels for model monitoring and future retraining.
