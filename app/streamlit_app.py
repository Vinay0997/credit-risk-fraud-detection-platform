"""Optional Streamlit demo app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from credit_risk_platform.data.make_dataset import generate_synthetic_risk_events
from credit_risk_platform.rag.policy_assistant import PolicyAssistant


st.set_page_config(page_title="Credit Risk & Fraud Platform", layout="wide")
st.title("Credit Risk & Fraud Detection Platform")

tab_data, tab_policy = st.tabs(["Synthetic Portfolio", "Policy Assistant"])

with tab_data:
    rows = st.slider("Rows", min_value=100, max_value=5000, value=1000, step=100)
    frame = generate_synthetic_risk_events(rows=rows)
    st.metric("Default rate", f"{frame['default_label'].mean():.1%}")
    st.metric("Fraud rate", f"{frame['fraud_label'].mean():.1%}")
    st.dataframe(frame.head(100), use_container_width=True)

with tab_policy:
    question = st.text_input("Ask a policy question", "What evidence is required for model validation?")
    assistant = PolicyAssistant("data/regulations")
    answer = assistant.answer(question)
    st.write(answer["answer"])
    st.caption(f"Citations: {answer['citations']}")
