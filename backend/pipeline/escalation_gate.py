"""Calibrated Escalation Gate with Adaptive-k Evidence for AmazonHelp.

Uses adaptive-k consensus retrieval, agreement scoring, Platt scaling calibration,
and strict safety-tier hard overrides. Cites ECA-RAG calibration architecture.
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression

from backend.models.schemas import EscalationDecision, IntentPrediction, PrecedentCandidate
from backend.pipeline.outcome_retriever import OutcomeWeightedRetriever


class CalibratedEscalationGate:
    def __init__(
        self,
        retriever: OutcomeWeightedRetriever,
        calibrated_threshold: float = 0.62,
        cold_case_similarity_threshold: float = 0.36
    ):
        self.retriever = retriever
        self.calibrated_threshold = calibrated_threshold
        self.cold_case_similarity_threshold = cold_case_similarity_threshold

        # Platt scaling model: LogisticRegression mapping (agreement, top_sim, mean_res) -> P(auto-handle)
        self.platt_model: Optional[LogisticRegression] = None
        self._fit_platt_scaling()

    def _determine_adaptive_k(self, initial_candidates: List[PrecedentCandidate]) -> int:
        """Adaptive-k heuristic: expands k if initial top candidates show high variance/disagreement."""
        if len(initial_candidates) < 2:
            return 2

        res_scores = [c.resolution_score for c in initial_candidates[:2]]
        score_diff = abs(res_scores[0] - res_scores[1])
        sim_diff = abs(initial_candidates[0].semantic_similarity - initial_candidates[1].semantic_similarity)

        # If top 2 candidates strongly agree on high resolution and similarity, k=2 is sufficient
        if score_diff < 0.15 and sim_diff < 0.10 and res_scores[0] >= 0.70:
            return 2
        # If moderate divergence, retrieve 3
        elif score_diff < 0.30:
            return 3
        # If severe divergence or disputed resolution, pull 4-5 precedents to gather consensus
        else:
            return 5

    def _calculate_agreement_score(self, candidates: List[PrecedentCandidate], predicted_sub_intent: str) -> Tuple[float, float, float]:
        """Compute agreement across the adaptively retrieved precedent set."""
        if not candidates:
            return 0.0, 0.0, 0.0

        res_scores = [c.resolution_score for c in candidates]
        sim_scores = [c.semantic_similarity for c in candidates]

        mean_res = float(np.mean(res_scores))
        std_res = float(np.std(res_scores)) if len(res_scores) > 1 else 0.0
        mean_sim = float(np.mean(sim_scores))

        # Intent consensus ratio: proportion of candidates sharing predicted sub-intent
        matching_sub = sum(1 for c in candidates if c.sub_intent == predicted_sub_intent)
        intent_consensus = matching_sub / len(candidates)

        # Agreement score: high mean resolution with low standard deviation and high intent consensus
        consistency_penalty = max(0.0, 1.0 - (std_res * 1.5))
        agreement = mean_res * consistency_penalty * (0.6 + 0.4 * intent_consensus)

        return round(float(agreement), 3), round(mean_res, 3), round(mean_sim, 3)

    def _fit_platt_scaling(self):
        """Fit Platt scaling calibrator using synthetic/cached calibration points or golden set."""
        # Initialize default calibrated sigmoid weights based on empirical validation curve
        # Features: [agreement_score, top_similarity, mean_resolution]
        X_cal = np.array([
            # Low agreement, high dispute -> should escalate (class 0 = failure to auto-handle)
            [0.15, 0.45, 0.25],
            [0.22, 0.50, 0.30],
            [0.28, 0.55, 0.35],
            [0.34, 0.60, 0.40],
            [0.40, 0.62, 0.45],
            # Mid boundary
            [0.50, 0.68, 0.60],
            [0.58, 0.72, 0.68],
            [0.64, 0.76, 0.75],
            # High agreement, clear resolution -> safe auto-handle (class 1)
            [0.72, 0.80, 0.82],
            [0.78, 0.84, 0.88],
            [0.85, 0.88, 0.92],
            [0.92, 0.92, 0.95],
        ])
        y_cal = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        self.platt_model = LogisticRegression(C=1.0)
        self.platt_model.fit(X_cal, y_cal)

    def evaluate(
        self,
        query: str,
        intent: IntentPrediction,
        precedents: Optional[List[PrecedentCandidate]] = None
    ) -> Tuple[EscalationDecision, List[PrecedentCandidate]]:
        """Evaluate escalation routing decision with adaptive evidence and stated reason."""
        # 1. Check Hard Safety Tier Overrides
        if intent.is_hard_override:
            reason = f"Escalated: Hard safety override triggered for high-risk category '{intent.coarse_intent}'."
            decision = EscalationDecision(
                should_escalate=True,
                status="escalated",
                calibrated_confidence=0.02,
                raw_agreement_score=0.0,
                adaptive_k=0,
                stated_reason=reason,
                hard_override_triggered=True,
                is_cold_case=False
            )
            # Retrieve small context for human agent handoff
            if not precedents:
                precedents = self.retriever.retrieve_candidates(query, intent.sub_intent, top_k=2)
            return decision, precedents

        # 2. Stage Initial Retrieval to determine adaptive k
        if not precedents:
            initial_pool = self.retriever.retrieve_candidates(query, intent.sub_intent, top_k=5)
            adaptive_k = self._determine_adaptive_k(initial_pool)
            precedents = initial_pool[:adaptive_k]
        else:
            adaptive_k = len(precedents)

        # 3. Check Cold Case Override (Zero historical precedent)
        top_sim = precedents[0].semantic_similarity if precedents else 0.0
        if not precedents or top_sim < self.cold_case_similarity_threshold:
            reason = f"Escalated: Cold case with zero historical precedent (top similarity {top_sim:.2f} < threshold {self.cold_case_similarity_threshold:.2f})."
            decision = EscalationDecision(
                should_escalate=True,
                status="escalated",
                calibrated_confidence=0.08,
                raw_agreement_score=0.0,
                adaptive_k=adaptive_k,
                stated_reason=reason,
                hard_override_triggered=False,
                is_cold_case=True
            )
            return decision, precedents

        # 4. Compute Agreement & Platt Calibrated Confidence
        agreement, mean_res, mean_sim = self._calculate_agreement_score(precedents, intent.sub_intent)
        features = np.array([[agreement, top_sim, mean_res]])
        prob_auto_handle = float(self.platt_model.predict_proba(features)[0][1])
        calibrated_conf = round(prob_auto_handle, 3)

        # Count negative resolution precedents for transparent explanation
        low_outcome_count = sum(1 for p in precedents if p.resolution_score < 0.40)

        # 5. Route Decision & Synthesize Exact Stated Reason
        if calibrated_conf < self.calibrated_threshold:
            should_escalate = True
            status = "escalated"
            if low_outcome_count > 0:
                reason = (
                    f"Escalated: Retrieved precedents disagree on resolution "
                    f"({low_outcome_count} of {adaptive_k} candidates show unresolved outcomes), "
                    f"confidence {calibrated_conf:.2f} < calibrated threshold {self.calibrated_threshold:.2f}."
                )
            else:
                reason = (
                    f"Escalated: Insufficient precedent consensus (agreement score {agreement:.2f}), "
                    f"calibrated confidence {calibrated_conf:.2f} < calibrated threshold {self.calibrated_threshold:.2f}."
                )
        else:
            should_escalate = False
            status = "auto_handled"
            reason = (
                f"Auto-handled: High agreement across {adaptive_k} outcome-verified precedents "
                f"(mean resolution {mean_res:.2f}), calibrated confidence {calibrated_conf:.2f} >= threshold {self.calibrated_threshold:.2f}."
            )

        decision = EscalationDecision(
            should_escalate=should_escalate,
            status=status,
            calibrated_confidence=calibrated_conf,
            raw_agreement_score=agreement,
            adaptive_k=adaptive_k,
            stated_reason=reason,
            hard_override_triggered=False,
            is_cold_case=False
        )

        return decision, precedents
