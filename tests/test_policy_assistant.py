from credit_risk_platform.rag.policy_assistant import PolicyAssistant


def test_policy_assistant_returns_citations():
    assistant = PolicyAssistant("data/regulations")
    response = assistant.answer("What evidence is needed for model validation?")
    assert response["citations"]
    assert "answer" in response
