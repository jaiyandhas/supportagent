"""Baseline agents for empirical comparison (Phase 4).

1. Trivial Baseline: Majority class intent, static canned reply, never escalates.
2. Simple Baseline: TF-IDF intent classifier, top-1 nearest neighbor by plain similarity (no outcome reranking), fixed raw similarity threshold for escalation.
"""

import json
from typing import Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics.pairwise import cosine_similarity

from backend.models.schemas import EscalationDecision, IntentPrediction, PrecedentCandidate, SupportPredictionResponse


class TrivialBaselineAgent:
    """Trivial baseline: predicts majority class, returns fixed canned reply, never escalates."""

    def __init__(self, majority_coarse: str = "order_delivery", majority_sub: str = "tracking_status_eta"):
        self.majority_coarse = majority_coarse
        self.majority_sub = majority_sub
        self.canned_reply = (
            "Thank you for contacting AmazonHelp. Please track your delivery in the Amazon app or visit amazon.com/contact-us."
        )

    def process_query(self, query: str) -> SupportPredictionResponse:
        intent = IntentPrediction(
            coarse_intent=self.majority_coarse,
            coarse_confidence=0.50,
            sub_intent=self.majority_sub,
            sub_confidence=0.50,
            is_hard_override=False
        )
        escalation = EscalationDecision(
            should_escalate=False,
            status="auto_handled",
            calibrated_confidence=1.0,
            raw_agreement_score=1.0,
            adaptive_k=1,
            stated_reason="Trivial baseline: static rule never escalates.",
            hard_override_triggered=False,
            is_cold_case=False
        )
        return SupportPredictionResponse(
            query=query,
            intent=intent,
            escalation=escalation,
            precedents=[],
            drafted_reply=self.canned_reply,
            generation_mode="canned_baseline",
            debug_trace={"baseline": "trivial"}
        )


class SimpleBaselineAgent:
    """Simple baseline: TF-IDF intent classifier, plain top-1 retrieval, fixed similarity threshold."""

    def __init__(
        self,
        precedents_path: str = "data/historical_precedents.json",
        fixed_similarity_threshold: float = 0.65
    ):
        self.fixed_similarity_threshold = fixed_similarity_threshold
        with open(precedents_path, "r", encoding="utf-8") as f:
            self.precedents = json.load(f)

        self.queries = [p["customer_query"] for p in self.precedents]
        self.coarse_labels = [p["coarse_intent"] for p in self.precedents]
        self.sub_labels = [p["sub_intent"] for p in self.precedents]

        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.precedent_matrix = self.vectorizer.fit_transform(self.queries)

        # TF-IDF Intent Classifiers
        self.coarse_clf = RidgeClassifier()
        self.coarse_clf.fit(self.precedent_matrix, self.coarse_labels)

        self.sub_clf = RidgeClassifier()
        self.sub_clf.fit(self.precedent_matrix, self.sub_labels)

    def process_query(self, query: str) -> SupportPredictionResponse:
        q_vec = self.vectorizer.transform([query])

        # Predict intent
        pred_coarse = self.coarse_clf.predict(q_vec)[0]
        pred_sub = self.sub_clf.predict(q_vec)[0]

        intent = IntentPrediction(
            coarse_intent=pred_coarse,
            coarse_confidence=0.70,
            sub_intent=pred_sub,
            sub_confidence=0.70,
            is_hard_override=False
        )

        # Plain Top-1 Cosine Similarity Retrieval (NO outcome reranking)
        sims = cosine_similarity(q_vec, self.precedent_matrix)[0]
        top_idx = int(np.argmax(sims))
        top_sim = float(sims[top_idx])
        top_prec = self.precedents[top_idx]

        # Fixed Threshold Escalation
        should_escalate = top_sim < self.fixed_similarity_threshold
        status = "escalated" if should_escalate else "auto_handled"
        reason = (
            f"Simple baseline: raw cosine similarity {top_sim:.2f} "
            f"{'<' if should_escalate else '>='} fixed threshold {self.fixed_similarity_threshold:.2f}."
        )

        candidate = PrecedentCandidate(
            precedent_id=top_prec["precedent_id"],
            customer_query=top_prec["customer_query"],
            support_reply=top_prec["support_reply"],
            customer_followup=top_prec.get("customer_followup"),
            resolution_score=top_prec.get("resolution_score", 0.5),
            resolution_rationale=top_prec.get("resolution_rationale", ""),
            coarse_intent=top_prec.get("coarse_intent", "order_delivery"),
            sub_intent=top_prec.get("sub_intent", "tracking_status_eta"),
            semantic_similarity=round(top_sim, 3),
            reranked_score=round(top_sim, 3)
        )

        escalation = EscalationDecision(
            should_escalate=should_escalate,
            status=status,
            calibrated_confidence=round(top_sim, 3),  # Uncalibrated raw similarity
            raw_agreement_score=round(top_sim, 3),
            adaptive_k=1,
            stated_reason=reason,
            hard_override_triggered=False,
            is_cold_case=False
        )

        reply = top_prec["support_reply"]
        if not reply.startswith(("Hi", "Hello", "Thanks")):
            reply = f"Hello, {reply}"

        return SupportPredictionResponse(
            query=query,
            intent=intent,
            escalation=escalation,
            precedents=[candidate],
            drafted_reply=reply,
            generation_mode="simple_baseline_nn",
            debug_trace={"baseline": "simple", "raw_similarity": top_sim}
        )
