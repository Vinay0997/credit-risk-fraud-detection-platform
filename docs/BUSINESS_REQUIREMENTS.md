# Business Requirements

## 1. Purpose

The Credit Risk & Fraud Detection Platform helps risk, fraud, and compliance teams make faster, more explainable, and audit-ready decisions. The platform combines supervised credit scoring, fraud pattern detection, model explainability, production drift monitoring, and a regulatory policy assistant that returns citation-backed answers.

## 2. Business Objectives

- Reduce time required for credit risk review and fraud investigation.
- Improve early-warning default detection using behavioral, transactional, and bureau signals.
- Reduce fraud false positives by ranking alerts with model-based risk scores.
- Provide clear feature rationale for every model decision.
- Preserve evidence for audit, model validation, and compliance review.
- Let analysts query internal policy and regulatory guidance in natural language with traceable citations.

## 3. Stakeholders

| Stakeholder | Need |
| --- | --- |
| Credit risk analysts | Understand borrower default risk and key drivers. |
| Fraud operations | Prioritize suspicious activity and reduce unnecessary investigations. |
| Compliance teams | Search policy and regulation evidence quickly. |
| Model risk management | Validate stability, explainability, and lifecycle controls. |
| Data engineering | Maintain reliable data pipelines and feature quality. |
| Audit teams | Review decision evidence, model documentation, and citations. |

## 4. Functional Requirements

### Credit Risk Scoring

- Ingest borrower, bureau, transaction, and behavioral features.
- Train a supervised model to estimate probability of default.
- Output credit risk probability, risk band, and top contributing features.
- Support back-testing and model performance reporting.
- Store training metrics and model artifacts for reproducibility.

### Fraud Detection

- Ingest transaction amount, device risk, login velocity, merchant risk, and geolocation velocity.
- Score events for suspicious activity.
- Generate alert priority based on probability threshold and feature rationale.
- Support batch scoring and real-time pipeline extension.
- Track false-positive reduction through operational KPIs.

### Explainability

- Produce feature-level rationale for every scored record.
- Use SHAP when installed and configured.
- Provide deterministic fallback explanations when SHAP is unavailable.
- Save explanations with model score outputs for audit review.

### Regulatory & Policy Assistant

- Index policy, control, and regulatory documents.
- Answer natural language questions with source citations.
- Return document name, chunk identifier, and relevance score.
- Avoid unsupported answers when no relevant source is found.
- Preserve traceability for compliance and audit use cases.

### Model Monitoring

- Compare current production data against reference training data.
- Calculate population stability index for important features.
- Flag warning and critical drift conditions.
- Generate JSON reports suitable for dashboards and evidence packs.

### Model Lifecycle

- Track model parameters, metrics, versions, and artifacts.
- Support MLflow integration when configured.
- Keep reproducible training and scoring scripts.
- Maintain model governance documentation aligned to SR 11-7 expectations.

## 5. Non-Functional Requirements

| Category | Requirement |
| --- | --- |
| Explainability | Every material model output must include score rationale. |
| Auditability | Scores, features, model versions, and citations must be traceable. |
| Reliability | Batch jobs should fail loudly on missing required columns. |
| Security | Secrets must stay outside source control and use environment variables. |
| Privacy | Public repository must use synthetic data only. |
| Scalability | Pipelines should support migration from local pandas to Spark/SageMaker. |
| Maintainability | Code should be modular across data, models, monitoring, RAG, and lifecycle. |

## 6. Business Rules

- Credit score probability above `approval_cutoff` is routed to manual review.
- Fraud score probability above `fraud_alert_cutoff` creates a high-priority alert.
- Drift PSI above `0.10` requires monitoring review.
- Drift PSI above `0.25` requires model owner escalation.
- Policy assistant responses must include citations when used for compliance decisions.
- Model releases require evidence of training data, performance metrics, explainability, and drift checks.

## 7. KPIs

- Default prediction AUC and precision-recall AUC.
- Fraud alert precision and false-positive rate.
- Analyst review time per case.
- Percentage of model outputs with explanations.
- Number of policy assistant answers with citations.
- Drift warning count by feature and model.
- Time to produce audit evidence pack.

## 8. Assumptions

- This public project uses synthetic data and sample policy documents.
- Production deployment would connect to Snowflake, streaming transaction sources, MLflow, SageMaker, and Azure OpenAI.
- Final credit and fraud decisions remain subject to human review and organizational policy.

## 9. Acceptance Criteria

- Synthetic data can be generated locally.
- Credit and fraud models can be trained from the generated dataset.
- Batch scoring produces model scores and feature explanations.
- Drift report can be generated from reference and current datasets.
- Policy assistant can answer a sample governance question with citations.
- Business and governance documentation is present for GitHub review.
