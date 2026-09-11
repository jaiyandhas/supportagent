"""Unit tests for AmazonHelp ingestion, parsing, and resolution scoring."""

import pytest
from backend.ingest.pull_amazonhelp_threads import clean_tweet_text, compute_resolution_score, is_english, map_to_taxonomy


def test_clean_tweet_text():
    raw = "Hello @AmazonHelp @112849 my order is delayed! http://t.co/XYZ123 &amp; &gt;"
    cleaned = clean_tweet_text(raw)
    assert "@AmazonHelp" not in cleaned
    assert "@112849" not in cleaned
    assert "http://t.co" not in cleaned
    assert "&" in cleaned
    assert ">" in cleaned


def test_is_english_detection():
    assert is_english("Where is my package? The tracking says delivered yesterday.") is True
    assert is_english("¿Dónde está mi paquete? No he recibido nada todavía.") is False
    assert is_english("Hola amigos gracias por la ayuda") is False


def test_compute_resolution_score_positive():
    t1 = "Where is my package?"
    t2 = "Please check your tracking link."
    t3 = "Thank you so much, I got it and it is sorted now!"
    score, rationale = compute_resolution_score(t1, t2, t3)
    assert score >= 0.85
    assert "positive_acknowledgment" in rationale


def test_compute_resolution_score_negative():
    t1 = "My order is delayed"
    t2 = "We are looking into this."
    t3 = "Still waiting, this is terrible service and totally unacceptable!"
    score, rationale = compute_resolution_score(t1, t2, t3)
    assert score <= 0.30
    assert "escalation_negative" in rationale


def test_map_to_taxonomy_hard_overrides():
    c_hack, s_hack = map_to_taxonomy("Someone hacked into my account and placed unauthorized orders")
    assert c_hack == "account_access"
    assert s_hack == "compromised_account_hijack"

    c_scam, s_scam = map_to_taxonomy("Received a fake scam phone call from someone pretending to be Amazon")
    assert c_scam == "abuse_safety"
    assert s_scam == "phishing_scam_report"
