# Business Architecture

## 1. Executive View

The platform supports the credit risk and fraud operating model by connecting data, decision models, analyst workflows, governance controls, and policy knowledge retrieval. It is designed for explainable decisions, controlled model operations, and faster compliance review.

## 2. Capability Map

| Capability | Description | Business Owner |
| --- | --- | --- |
| Credit risk assessment | Estimate default risk and identify early-warning signals. | Credit Risk |
| Fraud detection | Detect suspicious behavioral and transaction patterns. | Fraud Operations |
| Explainability | Show feature rationale for every model decision. | Model Risk |
| Regulatory search | Retrieve citation-backed policy answers. | Compliance |
| Drift monitoring | Detect data and model stability issues. | Model Operations |
| Lifecycle management | Track experiments, artifacts, versions, and approvals. | ML Platform |
| Audit evidence | Preserve decision records and governance documentation. | Audit |

## 3. Business Process Flow

```mermaid
flowchart LR
    A["Customer or transaction event"] --> B["Data quality and feature checks"]
    B --> C["Credit risk score"]
    B --> D["Fraud score"]
    C --> E["Explainability package"]
    D --> E
    E --> F["Risk or fraud analyst review"]
    F --> G["Decision and case notes"]
    H["Policies and regulations"] --> I["Policy assistant"]
    I --> F
    G --> J["Audit-ready evidence record"]
    C --> K["Monitoring and back-testing"]
    D --> K
    K --> L["Model owner action"]
```

## 4. Logical Architecture

```mermaid
flowchart TB
    subgraph "Data Sources"
        DS1["Transactions"]
        DS2["Behavioral events"]
        DS3["Bureau attributes"]
        DS4["Internal memos and policies"]
    end

    subgraph "Data Platform"
        DP1["Snowflake curated tables"]
        DP2["Feature engineering jobs"]
        DP3["Reference and current datasets"]
    end

    subgraph "Decision Intelligence"
        M1["Credit risk model"]
        M2["Fraud detection model"]
        M3["SHAP explainability"]
        M4["Risk summaries"]
    end

    subgraph "Knowledge Intelligence"
        R1["Document chunking"]
        R2["Embedding and retrieval index"]
        R3["Azure OpenAI answer generation"]
        R4["Citation controls"]
    end

    subgraph "Governance"
        G1["MLflow experiment tracking"]
        G2["Model registry"]
        G3["Evidently drift reports"]
        G4["Validation documentation"]
    end

    subgraph "Users"
        U1["Credit analyst"]
        U2["Fraud analyst"]
        U3["Compliance reviewer"]
        U4["Model validator"]
    end

    DS1 --> DP1
    DS2 --> DP1
    DS3 --> DP1
    DS4 --> R1
    DP1 --> DP2
    DP2 --> M1
    DP2 --> M2
    M1 --> M3
    M2 --> M3
    M3 --> M4
    R1 --> R2
    R2 --> R3
    R3 --> R4
    M1 --> G1
    M2 --> G1
    G1 --> G2
    DP3 --> G3
    G2 --> G4
    M4 --> U1
    M4 --> U2
    R4 --> U3
    G4 --> U4
```

## 5. Data Domains

| Domain | Examples | Governance Notes |
| --- | --- | --- |
| Customer profile | Age band, income, employment years | Sensitive data, access controlled. |
| Bureau risk | Credit score, utilization, delinquencies | Regulated third-party data. |
| Transaction activity | Amount, merchant risk, channel | Used for fraud and credit signals. |
| Behavioral signals | Login velocity, device risk, geo velocity | Requires privacy review and minimization. |
| Model evidence | Scores, features, explanations, metrics | Retained for audit and validation. |
| Policy knowledge | Regulations, standards, internal controls | Citation-backed retrieval required. |

## 6. User Journey

1. A new application or transaction event enters the platform.
2. Feature checks validate that required attributes are present.
3. Credit risk and fraud models generate probability scores.
4. Explainability logic attaches feature rationale to each score.
5. Analysts review high-risk or suspicious cases in priority order.
6. Compliance teams use the policy assistant to confirm required evidence.
7. Decisions, citations, model version, and explanations are stored for audit.
8. Model owners review drift and back-testing results on a scheduled cadence.

## 7. Business Controls

| Control | Purpose |
| --- | --- |
| Feature completeness checks | Prevent scoring on incomplete records. |
| Score threshold routing | Ensure high-risk cases receive human review. |
| Explanation capture | Support fair, transparent, and auditable decisions. |
| Citation requirement | Prevent unsupported regulatory responses. |
| Drift thresholds | Trigger investigation when input populations change. |
| Model registry approval | Prevent unapproved model versions from production use. |
| Evidence retention | Preserve decision support material for audits. |

## 8. Target Operating Model

| Function | Responsibility |
| --- | --- |
| Risk Analytics | Own model development, thresholds, and score interpretation. |
| Fraud Operations | Own alert triage, feedback labels, and false-positive review. |
| Compliance | Own policy corpus, citation standards, and regulatory interpretation. |
| Model Risk Management | Validate methodology, explainability, and performance stability. |
| Data Engineering | Own data quality, feature pipelines, and source integrations. |
| ML Platform | Own deployment, MLflow tracking, SageMaker jobs, and monitoring automation. |

## 9. Production Deployment View

```mermaid
flowchart LR
    A["Snowflake"] --> B["Feature pipeline"]
    B --> C["SageMaker training"]
    C --> D["MLflow registry"]
    D --> E["Batch and real-time inference"]
    E --> F["Analyst workbench"]
    E --> G["Decision audit store"]
    H["Document repository"] --> I["Embedding pipeline"]
    I --> J["Vector index"]
    J --> K["Azure OpenAI RAG assistant"]
    K --> F
    E --> L["Evidently monitoring"]
    L --> M["Model risk dashboard"]
```

## 10. Current Repository Scope

This repository implements a local, shareable version of the architecture. It uses synthetic data, local model artifacts, local retrieval, and optional integration hooks for Azure OpenAI, MLflow, PySpark, Evidently, and SageMaker.
