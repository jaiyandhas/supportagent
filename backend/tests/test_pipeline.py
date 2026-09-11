"""Unit and integration tests for outcome retriever, classifier, and escalation gate."""

import pytest
from backend.pipeline.engine import SupportAgentEngine


@pytest.fixture(scope="module")
def engine():
    return SupportAgentEngine()


def test_hierarchical_classifier_coarse_and_sub(engine):
    pred = engine.classifier.predict("Where is my package? The tracking number is not updating.")
    assert pred.coarse_intent == "order_delivery"
    assert pred.sub_intent in ["tracking_status_eta", "damaged_or_missing_items"]
    assert 0.0 <= pred.coarse_confidence <= 1.0
    assert 0.0 <= pred.sub_confidence <= 1.0


def test_hard_safety_override_account_access(engine):
    res = engine.process_query("My account was hacked and someone changed my email address!")
    assert res.intent.is_hard_override is True
    assert res.intent.coarse_intent == "account_access"
    assert res.escalation.should_escalate is True
    assert res.escalation.hard_override_triggered is True
    assert "Hard safety override triggered" in res.escalation.stated_reason


def test_hard_safety_override_abuse_safety(engine):
    res = engine.process_query("A scammer called me demanding gift cards and claiming to be Amazon support.")
    assert res.intent.is_hard_override is True
    assert res.intent.coarse_intent == "abuse_safety"
    assert res.escalation.should_escalate is True
    assert res.escalation.hard_override_triggered is True
    assert "Hard safety override triggered" in res.escalation.stated_reason


def test_cold_case_override(engine):
    # Out of domain query with zero precedent
    res = engine.process_query("How do I solder an ESP32 microcontroller to my AWS Greengrass MQTT cluster?")
    assert res.escalation.should_escalate is True
    assert res.escalation.is_cold_case is True
    assert "Cold case" in res.escalation.stated_reason


def test_outcome_retriever_ranking_ordering(engine):
    # Retrieve candidates with alpha=0.6 (outcome reranked)
    candidates = engine.retriever.retrieve_candidates("Where is my order?", top_k=5, custom_alpha=0.6)
    assert len(candidates) > 0
    # Scores must be sorted in descending order of reranked_score
    scores = [c.reranked_score for c in candidates]
    assert scores == sorted(scores, reverse=True)
