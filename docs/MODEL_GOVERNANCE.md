# Model Governance and Validation Approach

## Governance Objective

The platform is designed to support model risk management expectations similar to SR 11-7: conceptual soundness, ongoing monitoring, outcome analysis, documentation, and independent validation.

## Model Inventory

| Model | Purpose | Inputs | Output | Owner |
| --- | --- | --- | --- | --- |
| Credit risk model | Estimate probability of default. | Bureau, behavioral, transaction, and profile features. | Default probability and risk band. | Credit Risk Analytics |
| Fraud detection model | Identify suspicious transaction behavior. | Device, transaction, merchant, and velocity features. | Fraud probability and alert priority. | Fraud Analytics |
| Policy assistant | Retrieve policy evidence for analysts. | Internal policy and regulatory documents. | Citation-backed answer. | Compliance |

## Validation Evidence

Each model release should include:

- Training dataset snapshot or query reference.
- Feature list and data dictionary.
- Model algorithm and hyperparameters.
- Performance metrics on train, validation, and holdout sets.
- Bias and segmentation analysis where applicable.
- Explainability report with top global and local drivers.
- Back-testing result against recent outcomes.
- Drift report against training reference.
- Approval record and model version identifier.

## Explainability Standard

Every model score should include:

- Score probability.
- Risk band or alert priority.
- Top positive and negative feature drivers.
- Model version and scoring timestamp.
- Input feature snapshot or traceable feature reference.

The repository includes a SHAP-ready explanation module. If SHAP is not installed, the module uses a deterministic feature-importance fallback so the demo remains runnable.

## Monitoring Standard

| Metric | Frequency | Action Threshold |
| --- | --- | --- |
| Population Stability Index | Daily or weekly | Warning above 0.10, critical above 0.25 |
| AUC / PR AUC | Monthly with labels | Investigate material decline |
| Approval and decline rates | Daily | Investigate unusual shifts |
| Fraud alert precision | Weekly | Tune thresholds or labels |
| Missing feature rate | Daily | Stop scoring when critical |

## RAG Control Standard

The policy assistant should:

- Return citations for every compliance-facing answer.
- Refuse or narrow answers when retrieval confidence is low.
- Use approved source documents only.
- Log user question, retrieved source IDs, and response timestamp.
- Keep generated summaries separate from source citations.

## Human Review

The platform is intended to assist analysts, not replace accountable business decisions. High-risk credit cases, high-priority fraud alerts, and ambiguous policy answers should be routed to human review.
