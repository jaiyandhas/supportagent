"""Pydantic schemas for AmazonHelp Trust-First Support Agent."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PrecedentCandidate(BaseModel):
    precedent_id: str
    customer_query: str
    support_reply: str
    customer_followup: Optional[str] = None
    resolution_score: float
    resolution_rationale: str
    coarse_intent: str
    sub_intent: str
    semantic_similarity: float
    reranked_score: float


class IntentPrediction(BaseModel):
    coarse_intent: str
    coarse_confidence: float
    sub_intent: str
    sub_confidence: float
    is_hard_override: bool


class EscalationDecision(BaseModel):
    should_escalate: bool
    status: str  # "auto_handled" or "escalated"
    calibrated_confidence: float
    raw_agreement_score: float
    adaptive_k: int
    stated_reason: str
    hard_override_triggered: bool
    is_cold_case: bool


class SupportPredictionResponse(BaseModel):
    query: str
    intent: IntentPrediction
    escalation: EscalationDecision
    precedents: List[PrecedentCandidate]
    drafted_reply: str
    generation_mode: str  # "live_llm" or "deterministic_repro"
    debug_trace: Dict[str, Any]


class PredictionRequest(BaseModel):
    query: str
    force_live_llm: Optional[bool] = False
