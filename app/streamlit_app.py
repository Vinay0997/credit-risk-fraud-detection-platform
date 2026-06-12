"""Streamlit web interface for the credit risk and fraud platform."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from credit_risk_platform.data.make_dataset import generate_synthetic_risk_events
from credit_risk_platform.explainability.shap_report import explain_records
from credit_risk_platform.features import (
    CREDIT_FEATURES,
    FRAUD_FEATURES,
    assign_risk_band,
    score_to_priority,
)
from credit_risk_platform.models.credit_model import train_credit_model
from credit_risk_platform.models.fraud_model import train_fraud_model
from credit_risk_platform.monitoring.drift_report import build_drift_report
from credit_risk_platform.rag.policy_assistant import PolicyAssistant


st.set_page_config(
    page_title="Credit Risk & Fraud Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.55rem;}
    [data-testid="stMetricLabel"] {font-size: .85rem;}
    div[data-testid="stDataFrame"] {border: 1px solid #d7dde8; border-radius: 6px;}
    .status-note {
        border: 1px solid #d7dde8;
        border-radius: 6px;
        padding: .75rem .9rem;
        background: #f8fafc;
        color: #243044;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    if "events" not in st.session_state:
        st.session_state.events = generate_synthetic_risk_events(rows=1200, seed=42)
    if "credit_model" not in st.session_state:
        st.session_state.credit_model = None
    if "fraud_model" not in st.session_state:
        st.session_state.fraud_model = None
    if "credit_metrics" not in st.session_state:
        st.session_state.credit_metrics = None
    if "fraud_metrics" not in st.session_state:
        st.session_state.fraud_metrics = None
    if "scored" not in st.session_state:
        st.session_state.scored = None


def train_models(frame: pd.DataFrame, seed: int) -> None:
    credit_result = train_credit_model(frame, random_seed=seed)
    fraud_result = train_fraud_model(frame, random_seed=seed)
    st.session_state.credit_model = credit_result.pipeline
    st.session_state.fraud_model = fraud_result.pipeline
    st.session_state.credit_metrics = credit_result.metrics
    st.session_state.fraud_metrics = fraud_result.metrics


def score_events(frame: pd.DataFrame) -> pd.DataFrame:
    credit_model = st.session_state.credit_model
    fraud_model = st.session_state.fraud_model
    if credit_model is None or fraud_model is None:
        raise RuntimeError("Train both models before scoring.")

    scored = frame[
        ["customer_id", "transaction_id", "event_timestamp", "default_label", "fraud_label"]
    ].copy()
    scored["credit_default_score"] = credit_model.predict_proba(frame[CREDIT_FEATURES])[:, 1]
    scored["credit_risk_band"] = scored["credit_default_score"].map(assign_risk_band)
    scored["fraud_score"] = fraud_model.predict_proba(frame[FRAUD_FEATURES])[:, 1]
    scored["fraud_priority"] = scored["fraud_score"].map(score_to_priority)
    return scored


def explain_selected_record(
    model: object,
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, object]:
    context = pd.concat([frame[feature_names].head(250), selected[feature_names]], ignore_index=True)
    explanations = explain_records(
        model,
        context,
        feature_names,
        max_rows=len(context),
        top_n=5,
    )
    return explanations[-1]


def build_shifted_current_frame(frame: pd.DataFrame, strength: float) -> pd.DataFrame:
    current = frame.copy()
    current["credit_utilization"] = (current["credit_utilization"] * (1 + strength)).clip(0, 1)
    current["debt_to_income"] = (current["debt_to_income"] * (1 + strength / 2)).clip(0, 1)
    current["device_risk_score"] = (current["device_risk_score"] * (1 + strength)).clip(0, 1)
    current["merchant_risk_score"] = (current["merchant_risk_score"] * (1 + strength)).clip(0, 1)
    current["login_velocity_1h"] = (current["login_velocity_1h"] * (1 + strength)).round()
    return current


def metric_value(metrics: dict[str, object] | None, name: str) -> str:
    if not metrics or name not in metrics:
        return "Not run"
    return f"{float(metrics[name]):.3f}"


def dataframe_download(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def metrics_download() -> bytes:
    payload = {
        "credit_metrics": st.session_state.credit_metrics,
        "fraud_metrics": st.session_state.fraud_metrics,
    }
    return json.dumps(payload, indent=2).encode("utf-8")


initialize_state()

st.title("Credit Risk & Fraud Detection Platform")

with st.sidebar:
    st.header("Pipeline Controls")
    rows = st.slider("Synthetic rows", min_value=500, max_value=10000, value=1200, step=100)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)

    if st.button("Generate Dataset", use_container_width=True):
        st.session_state.events = generate_synthetic_risk_events(rows=rows, seed=int(seed))
        st.session_state.credit_model = None
        st.session_state.fraud_model = None
        st.session_state.credit_metrics = None
        st.session_state.fraud_metrics = None
        st.session_state.scored = None
        st.success("Synthetic dataset generated.")

    if st.button("Train Models", type="primary", use_container_width=True):
        with st.spinner("Training credit risk and fraud models..."):
            train_models(st.session_state.events, int(seed))
        st.success("Models trained.")

    if st.button("Score Portfolio", use_container_width=True):
        if st.session_state.credit_model is None or st.session_state.fraud_model is None:
            st.warning("Train the models before scoring.")
        else:
            with st.spinner("Scoring records..."):
                st.session_state.scored = score_events(st.session_state.events)
            st.success("Portfolio scored.")

    st.divider()


events = st.session_state.events
scored = st.session_state.scored

tab_overview, tab_models, tab_monitoring, tab_policy, tab_downloads = st.tabs(
    ["Overview", "Model Scoring", "Monitoring", "Policy Assistant", "Downloads"]
)

with tab_overview:
    left, middle, right, far_right = st.columns(4)
    left.metric("Records", f"{len(events):,}")
    middle.metric("Observed default rate", f"{events['default_label'].mean():.1%}")
    right.metric("Observed fraud rate", f"{events['fraud_label'].mean():.1%}")
    far_right.metric("Avg bureau score", f"{events['bureau_score'].mean():.0f}")

    st.subheader("Synthetic Risk Events")
    st.dataframe(events.head(200), use_container_width=True, hide_index=True)

    st.subheader("Portfolio Signals")
    signal_frame = events[
        [
            "bureau_score",
            "credit_utilization",
            "debt_to_income",
            "device_risk_score",
            "merchant_risk_score",
            "transaction_amount",
        ]
    ]
    st.bar_chart(signal_frame.mean(numeric_only=True))

with tab_models:
    credit_col, fraud_col = st.columns(2)
    credit_col.subheader("Credit Risk Model")
    credit_col.metric("ROC AUC", metric_value(st.session_state.credit_metrics, "roc_auc"))
    credit_col.metric(
        "Average precision",
        metric_value(st.session_state.credit_metrics, "average_precision"),
    )
    if st.session_state.credit_metrics:
        credit_col.caption(f"Estimator: {st.session_state.credit_metrics['model_type']}")

    fraud_col.subheader("Fraud Detection Model")
    fraud_col.metric("ROC AUC", metric_value(st.session_state.fraud_metrics, "roc_auc"))
    fraud_col.metric(
        "Average precision",
        metric_value(st.session_state.fraud_metrics, "average_precision"),
    )
    if st.session_state.fraud_metrics:
        fraud_col.caption(f"Estimator: {st.session_state.fraud_metrics['model_type']}")

    if scored is None:
        st.markdown(
            '<div class="status-note">Train the models and score the portfolio from the sidebar.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.subheader("Scored Portfolio")
        st.dataframe(scored.head(250), use_container_width=True, hide_index=True)

        band_col, priority_col = st.columns(2)
        band_col.write("Credit risk band distribution")
        band_col.bar_chart(scored["credit_risk_band"].value_counts())
        priority_col.write("Fraud priority distribution")
        priority_col.bar_chart(scored["fraud_priority"].value_counts())

        st.subheader("Record Explanation")
        row_number = st.number_input(
            "Row number",
            min_value=0,
            max_value=max(len(events) - 1, 0),
            value=0,
            step=1,
        )
        selected = events.iloc[[int(row_number)]]
        selected_score = scored.iloc[[int(row_number)]]
        st.dataframe(selected_score, use_container_width=True, hide_index=True)

        exp_credit_col, exp_fraud_col = st.columns(2)
        credit_explanation = explain_selected_record(
            st.session_state.credit_model,
            events,
            selected,
            CREDIT_FEATURES,
        )
        fraud_explanation = explain_selected_record(
            st.session_state.fraud_model,
            events,
            selected,
            FRAUD_FEATURES,
        )
        exp_credit_col.write("Credit explanation")
        exp_credit_col.json(credit_explanation)
        exp_fraud_col.write("Fraud explanation")
        exp_fraud_col.json(fraud_explanation)

with tab_monitoring:
    st.subheader("Data Drift Monitoring")
    drift_strength = st.slider("Current population shift", 0.0, 0.8, 0.2, 0.05)
    current_frame = build_shifted_current_frame(events, drift_strength)
    drift_report = build_drift_report(events, current_frame)
    st.metric("Overall drift status", str(drift_report["overall_status"]).upper())
    st.dataframe(pd.DataFrame(drift_report["features"]), use_container_width=True, hide_index=True)
    st.download_button(
        "Download Drift Report",
        data=json.dumps(drift_report, indent=2).encode("utf-8"),
        file_name="drift_report.json",
        mime="application/json",
    )

with tab_policy:
    st.subheader("Regulatory & Policy Assistant")
    question = st.text_input(
        "Question",
        "What evidence is required for model validation?",
    )
    top_k = st.slider("Citations", min_value=1, max_value=5, value=3)
    if st.button("Ask Policy Assistant", type="primary"):
        assistant = PolicyAssistant(Path("data/regulations"))
        response = assistant.answer(question, top_k=top_k)
        st.write(response["answer"])
        st.dataframe(pd.DataFrame(response["citations"]), use_container_width=True, hide_index=True)

with tab_downloads:
    st.subheader("Export Demo Outputs")
    st.download_button(
        "Download Synthetic Dataset",
        data=dataframe_download(events),
        file_name="risk_events.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if scored is not None:
        st.download_button(
            "Download Scored Portfolio",
            data=dataframe_download(scored),
            file_name="scored_events.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if st.session_state.credit_metrics and st.session_state.fraud_metrics:
        st.download_button(
            "Download Model Metrics",
            data=metrics_download(),
            file_name="model_metrics.json",
            mime="application/json",
            use_container_width=True,
        )

    st.write("Model state")
    st.json(
        {
            "credit_model_trained": st.session_state.credit_model is not None,
            "fraud_model_trained": st.session_state.fraud_model is not None,
            "portfolio_scored": scored is not None,
        }
    )
