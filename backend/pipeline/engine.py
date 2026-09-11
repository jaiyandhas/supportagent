"""Unified Pipeline Engine orchestrating classification, retrieval, escalation gating, and reply generation."""

import time
from typing import Optional
from backend.models.schemas import PredictionRequest, SupportPredictionResponse
from backend.pipeline.classifier import HierarchicalIntentClassifier
from backend.pipeline.outcome_retriever import OutcomeWeightedRetriever
from backend.pipeline.escalation_gate import CalibratedEscalationGate
from backend.pipeline.generator import ReplyGenerator


class SupportAgentEngine:
    def __init__(
        self,
        taxonomy_path: str = "data/taxonomy.json",
        precedents_path: str = "data/historical_precedents.json"
    ):
        self.classifier = HierarchicalIntentClassifier(taxonomy_path, precedents_path)
        self.retriever = OutcomeWeightedRetriever(precedents_path)
        self.gate = CalibratedEscalationGate(self.retriever)
        self.generator = ReplyGenerator()

    def process_query(self, query: str, force_live_llm: bool = False) -> SupportPredictionResponse:
        """Process customer query through full calibrated trust-first pipeline."""
        start_time = time.time()

        # Step 1: Hierarchical Intent Classification
        intent = self.classifier.predict(query)

        # Step 2 & 3: Outcome-Weighted Retrieval and Calibrated Escalation Gate
        decision, precedents = self.gate.evaluate(query, intent)

        # Step 4: Reply Generation
        reply, mode, debug_trace = self.generator.generate(
            query, intent, decision, precedents, force_live_llm=force_live_llm
        )

        elapsed_ms = round((time.time() - start_time) * 1000, 1)
        debug_trace["latency_ms"] = elapsed_ms

        return SupportPredictionResponse(
            query=query,
            intent=intent,
            escalation=decision,
            precedents=precedents,
            drafted_reply=reply,
            generation_mode=mode,
            debug_trace=debug_trace
        )
