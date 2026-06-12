"""Gradio web interface for the credit risk and fraud platform."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

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


APP_STATE: dict[str, Any] = {
    "events": generate_synthetic_risk_events(rows=1200, seed=42),
    "credit_model": None,
    "fraud_model": None,
    "credit_metrics": None,
    "fraud_metrics": None,
    "scored": None,
}


def _events() -> pd.DataFrame:
    return APP_STATE["events"]


def _summary_markdown(frame: pd.DataFrame) -> str:
    return "\n".join(
        [
            f"**Records:** {len(frame):,}",
            f"**Observed default rate:** {frame['default_label'].mean():.1%}",
            f"**Observed fraud rate:** {frame['fraud_label'].mean():.1%}",
            f"**Average bureau score:** {frame['bureau_score'].mean():.0f}",
        ]
    )


def _model_state_markdown() -> str:
    credit_ready = APP_STATE["credit_model"] is not None
    fraud_ready = APP_STATE["fraud_model"] is not None
    scored_ready = APP_STATE["scored"] is not None
    return "\n".join(
        [
            f"**Credit model trained:** {'Yes' if credit_ready else 'No'}",
            f"**Fraud model trained:** {'Yes' if fraud_ready else 'No'}",
            f"**Portfolio scored:** {'Yes' if scored_ready else 'No'}",
        ]
    )


def _metrics_payload() -> dict[str, Any]:
    return {
        "credit_metrics": APP_STATE["credit_metrics"],
        "fraud_metrics": APP_STATE["fraud_metrics"],
    }


def generate_dataset(rows: int, seed: int) -> tuple[str, str, pd.DataFrame, str]:
    rows = int(rows)
    seed = int(seed)
    APP_STATE["events"] = generate_synthetic_risk_events(rows=rows, seed=seed)
    APP_STATE["credit_model"] = None
    APP_STATE["fraud_model"] = None
    APP_STATE["credit_metrics"] = None
    APP_STATE["fraud_metrics"] = None
    APP_STATE["scored"] = None
    frame = _events()
    return (
        f"Generated {len(frame):,} synthetic records.",
        _summary_markdown(frame),
        frame.head(200),
        _model_state_markdown(),
    )


def train_models(seed: int) -> tuple[str, dict[str, Any], str]:
    seed = int(seed)
    frame = _events()
    credit_result = train_credit_model(frame, random_seed=seed)
    fraud_result = train_fraud_model(frame, random_seed=seed)
    APP_STATE["credit_model"] = credit_result.pipeline
    APP_STATE["fraud_model"] = fraud_result.pipeline
    APP_STATE["credit_metrics"] = credit_result.metrics
    APP_STATE["fraud_metrics"] = fraud_result.metrics
    APP_STATE["scored"] = None
    return (
        "Models trained successfully.",
        _metrics_payload(),
        _model_state_markdown(),
    )


def score_portfolio() -> tuple[str, pd.DataFrame, str]:
    if APP_STATE["credit_model"] is None or APP_STATE["fraud_model"] is None:
        return "Train the models before scoring.", pd.DataFrame(), _model_state_markdown()

    frame = _events()
    credit_model = APP_STATE["credit_model"]
    fraud_model = APP_STATE["fraud_model"]
    scored = frame[
        ["customer_id", "transaction_id", "event_timestamp", "default_label", "fraud_label"]
    ].copy()
    scored["credit_default_score"] = credit_model.predict_proba(frame[CREDIT_FEATURES])[:, 1]
    scored["credit_risk_band"] = scored["credit_default_score"].map(assign_risk_band)
    scored["fraud_score"] = fraud_model.predict_proba(frame[FRAUD_FEATURES])[:, 1]
    scored["fraud_priority"] = scored["fraud_score"].map(score_to_priority)
    APP_STATE["scored"] = scored
    return "Portfolio scored successfully.", scored.head(250), _model_state_markdown()


def explain_record(row_number: int) -> tuple[str, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if APP_STATE["scored"] is None:
        return (
            "Score the portfolio before generating explanations.",
            pd.DataFrame(),
            {},
            {},
        )

    frame = _events()
    row_number = int(max(0, min(row_number, len(frame) - 1)))
    selected = frame.iloc[[row_number]]
    scored = APP_STATE["scored"].iloc[[row_number]]
    context_credit = pd.concat(
        [frame[CREDIT_FEATURES].head(250), selected[CREDIT_FEATURES]],
        ignore_index=True,
    )
    context_fraud = pd.concat(
        [frame[FRAUD_FEATURES].head(250), selected[FRAUD_FEATURES]],
        ignore_index=True,
    )
    credit_explanation = explain_records(
        APP_STATE["credit_model"],
        context_credit,
        CREDIT_FEATURES,
        max_rows=len(context_credit),
        top_n=5,
    )[-1]
    fraud_explanation = explain_records(
        APP_STATE["fraud_model"],
        context_fraud,
        FRAUD_FEATURES,
        max_rows=len(context_fraud),
        top_n=5,
    )[-1]
    return (
        f"Explanation generated for row {row_number}.",
        scored,
        credit_explanation,
        fraud_explanation,
    )


def _build_shifted_current_frame(frame: pd.DataFrame, strength: float) -> pd.DataFrame:
    current = frame.copy()
    current["credit_utilization"] = (current["credit_utilization"] * (1 + strength)).clip(0, 1)
    current["debt_to_income"] = (current["debt_to_income"] * (1 + strength / 2)).clip(0, 1)
    current["device_risk_score"] = (current["device_risk_score"] * (1 + strength)).clip(0, 1)
    current["merchant_risk_score"] = (current["merchant_risk_score"] * (1 + strength)).clip(0, 1)
    current["login_velocity_1h"] = (current["login_velocity_1h"] * (1 + strength)).round()
    return current


def monitor_drift(shift_strength: float) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    frame = _events()
    current = _build_shifted_current_frame(frame, float(shift_strength))
    report = build_drift_report(frame, current)
    features = pd.DataFrame(report["features"])
    return (
        f"Overall drift status: {str(report['overall_status']).upper()}",
        features,
        report,
    )


def ask_policy_assistant(question: str, top_k: int) -> tuple[str, pd.DataFrame]:
    assistant = PolicyAssistant(Path("data/regulations"))
    response = assistant.answer(question, top_k=int(top_k))
    citations = pd.DataFrame(response["citations"])
    return str(response["answer"]), citations


def export_outputs() -> tuple[str, str | None, str | None]:
    output_dir = Path("artifacts/gradio")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "risk_events.csv"
    _events().to_csv(dataset_path, index=False)

    scored_path: Path | None = None
    if APP_STATE["scored"] is not None:
        scored_path = output_dir / "scored_events.csv"
        APP_STATE["scored"].to_csv(scored_path, index=False)

    metrics_path: Path | None = None
    if APP_STATE["credit_metrics"] and APP_STATE["fraud_metrics"]:
        metrics_path = output_dir / "model_metrics.json"
        metrics_path.write_text(json.dumps(_metrics_payload(), indent=2), encoding="utf-8")

    return (
        str(dataset_path),
        str(scored_path) if scored_path else None,
        str(metrics_path) if metrics_path else None,
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Credit Risk & Fraud Detection Platform") as demo:
        gr.Markdown("# Credit Risk & Fraud Detection Platform")
        gr.Markdown(
            "Generate synthetic risk events, train credit and fraud models, score a portfolio, "
            "review explanations, monitor drift, and query policy controls."
        )

        with gr.Row():
            rows = gr.Slider(500, 10000, value=1200, step=100, label="Synthetic rows")
            seed = gr.Number(value=42, precision=0, label="Random seed")

        with gr.Tab("Overview"):
            generate_button = gr.Button("Generate Dataset", variant="primary")
            generate_status = gr.Markdown("Dataset is ready.")
            summary = gr.Markdown(_summary_markdown(_events()))
            events_table = gr.Dataframe(
                value=_events().head(200),
                label="Synthetic risk events",
                interactive=False,
                wrap=True,
            )

        with gr.Tab("Model Scoring"):
            with gr.Row():
                train_button = gr.Button("Train Models", variant="primary")
                score_button = gr.Button("Score Portfolio")
            model_message = gr.Markdown()
            model_status = gr.Markdown(_model_state_markdown())
            metrics_json = gr.JSON(label="Model metrics")
            scored_table = gr.Dataframe(label="Scored portfolio", interactive=False, wrap=True)

        with gr.Tab("Explainability"):
            row_number = gr.Number(value=0, precision=0, label="Row number")
            explain_button = gr.Button("Explain Record", variant="primary")
            explanation_status = gr.Markdown()
            selected_score = gr.Dataframe(label="Selected scored record", interactive=False)
            with gr.Row():
                credit_explanation = gr.JSON(label="Credit explanation")
                fraud_explanation = gr.JSON(label="Fraud explanation")

        with gr.Tab("Monitoring"):
            shift_strength = gr.Slider(
                0.0,
                0.8,
                value=0.2,
                step=0.05,
                label="Current population shift",
            )
            drift_button = gr.Button("Run Drift Report", variant="primary")
            drift_status = gr.Markdown()
            drift_table = gr.Dataframe(label="Feature drift", interactive=False)
            drift_json = gr.JSON(label="Drift report")

        with gr.Tab("Policy Assistant"):
            question = gr.Textbox(
                value="What evidence is required for model validation?",
                label="Policy question",
                lines=2,
            )
            top_k = gr.Slider(1, 5, value=3, step=1, label="Number of citations")
            ask_button = gr.Button("Ask Policy Assistant", variant="primary")
            answer = gr.Textbox(label="Answer", lines=8)
            citations = gr.Dataframe(label="Citations", interactive=False)

        with gr.Tab("Downloads"):
            export_button = gr.Button("Prepare Downloads", variant="primary")
            dataset_file = gr.File(label="Synthetic dataset CSV")
            scored_file = gr.File(label="Scored portfolio CSV")
            metrics_file = gr.File(label="Model metrics JSON")

        generate_button.click(
            generate_dataset,
            inputs=[rows, seed],
            outputs=[generate_status, summary, events_table, model_status],
        )
        train_button.click(
            train_models,
            inputs=[seed],
            outputs=[model_message, metrics_json, model_status],
        )
        score_button.click(
            score_portfolio,
            inputs=[],
            outputs=[model_message, scored_table, model_status],
        )
        explain_button.click(
            explain_record,
            inputs=[row_number],
            outputs=[
                explanation_status,
                selected_score,
                credit_explanation,
                fraud_explanation,
            ],
        )
        drift_button.click(
            monitor_drift,
            inputs=[shift_strength],
            outputs=[drift_status, drift_table, drift_json],
        )
        ask_button.click(
            ask_policy_assistant,
            inputs=[question, top_k],
            outputs=[answer, citations],
        )
        export_button.click(
            export_outputs,
            inputs=[],
            outputs=[dataset_file, scored_file, metrics_file],
        )

    return demo


if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    build_demo().launch(server_name="0.0.0.0", server_port=7860, share=share)
