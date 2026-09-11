export interface PrecedentCandidate {
  precedent_id: string;
  customer_query: string;
  support_reply: string;
  customer_followup?: string | null;
  resolution_score: number;
  resolution_rationale: string;
  coarse_intent: string;
  sub_intent: string;
  semantic_similarity: number;
  reranked_score: number;
}

export interface IntentPrediction {
  coarse_intent: string;
  coarse_confidence: number;
  sub_intent: string;
  sub_confidence: number;
  is_hard_override: boolean;
}

export interface EscalationDecision {
  should_escalate: boolean;
  status: "auto_handled" | "escalated";
  calibrated_confidence: number;
  raw_agreement_score: number;
  adaptive_k: number;
  stated_reason: string;
  hard_override_triggered: boolean;
  is_cold_case: boolean;
}

export interface SupportPredictionResponse {
  query: string;
  intent: IntentPrediction;
  escalation: EscalationDecision;
  precedents: PrecedentCandidate[];
  drafted_reply: string;
  generation_mode: string;
  debug_trace: Record<string, any>;
}

export interface QueueMessage {
  id: string;
  customer_query: string;
  prediction: SupportPredictionResponse;
  gold_should_escalate: boolean;
  is_cold_case: boolean;
  is_borderline: boolean;
}

export interface ReliabilityPoint {
  bin_index: number;
  bin_center: number;
  mean_confidence: number;
  empirical_accuracy: number;
  sample_count: number;
}

export interface CoverageRiskPoint {
  threshold: number;
  coverage: number;
  risk: number;
  auto_handled_count: number;
}

export interface CalibrationPayload {
  calibrated_threshold: number;
  trust_first: {
    ece: number;
    brier_score: number;
    reliability_curve: ReliabilityPoint[];
  };
  simple_baseline: {
    ece: number;
    brier_score: number;
    reliability_curve: ReliabilityPoint[];
  };
  coverage_risk_curve: CoverageRiskPoint[];
}

export interface HealthInfo {
  status: string;
  brand: string;
  generation_mode: string;
  active_provider: string;
  embedding_model: string;
  retrieval_index: string;
  calibration_technique: string;
}
