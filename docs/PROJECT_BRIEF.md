# Project Brief

## Project Title

Credit Risk & Fraud Detection Platform

## Project Description

This project demonstrates a risk intelligence platform that combines machine learning, model explainability, monitoring, and a regulatory policy assistant. It is designed for credit risk, fraud operations, compliance, and model risk teams that need faster decisions with clear evidence and audit traceability.

The public repository uses synthetic data and sample policy documents, while the architecture is written to align with a production stack using Snowflake, PySpark, MLflow, AWS SageMaker, and Azure OpenAI.

## Responsibilities Represented

- Built credit risk scoring workflows using supervised machine learning over bureau, behavioral, and transaction features.
- Developed fraud scoring workflows over transaction amount, merchant risk, device risk, login velocity, and geolocation signals.
- Added SHAP-ready model explainability so analysts can understand top feature drivers for each score.
- Created a regulatory and policy assistant that retrieves citation-backed answers from controlled documents.
- Added model drift monitoring using population stability index, with thresholds for warning and critical review.
- Included MLflow-ready lifecycle hooks for experiment tracking and model artifact governance.
- Documented business requirements, architecture, operating controls, and validation expectations.

## Technology Stack

| Area | Tools |
| --- | --- |
| Modeling | scikit-learn, optional XGBoost, optional PyTorch |
| Feature engineering | pandas, optional PySpark |
| Explainability | SHAP-ready module with fallback feature attribution |
| Monitoring | PSI drift reports, optional Evidently extension |
| Lifecycle | MLflow-ready tracking hooks |
| Data platform | Snowflake feature view examples |
| LLM and RAG | Local TF-IDF retrieval demo, Azure OpenAI integration placeholders |
| App demo | Optional Streamlit app |
| Testing | pytest |

## Business Value

- Helps analysts prioritize credit and fraud cases.
- Reduces manual review time by attaching explanations and policy evidence.
- Improves model governance through reproducible artifacts and monitoring reports.
- Gives compliance teams traceable answers instead of unsupported summaries.
- Provides a public, safe, synthetic version of an enterprise risk platform.
