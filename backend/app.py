"""FastAPI backend application for AmazonHelp Trust-First Support Agent."""

import json
import os
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.models.schemas import PredictionRequest, SupportPredictionResponse
from backend.pipeline.engine import SupportAgentEngine

app = FastAPI(
    title="AmazonHelp Trust-First Support Agent API",
    description="Backend API powering outcome-verified precedent retrieval, calibrated escalation gating, and trust dashboard.",
    version="0.1.0"
)

# Enable CORS for React frontend (Vite port 5173 / localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine singleton
ENGINE: Optional[SupportAgentEngine] = None


def get_engine() -> SupportAgentEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = SupportAgentEngine()
    return ENGINE


@app.get("/api/health")
def health_check():
    gemini_active = bool(os.getenv("GEMINI_API_KEY"))
    openai_active = bool(os.getenv("OPENAI_API_KEY"))
    active_mode = "live_llm" if (gemini_active or openai_active) else "deterministic_repro"
    active_provider = "Gemini" if gemini_active else ("OpenAI" if openai_active else "Local Engine")

    return {
        "status": "healthy",
        "brand": "AmazonHelp",
        "generation_mode": active_mode,
        "active_provider": active_provider,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "retrieval_index": "FAISS IndexFlatIP (cosine)",
        "calibration_technique": "Platt scaling + Adaptive-k consensus (ECA-RAG)"
    }


@app.get("/api/taxonomy")
def get_taxonomy():
    tax_path = "data/taxonomy.json"
    if not os.path.exists(tax_path):
        raise HTTPException(status_code=404, detail="Taxonomy file not found")
    with open(tax_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/messages")
def get_message_queue(limit: int = 50, filter_status: Optional[str] = None):
    """Retrieve queue of messages for the UI left pane."""
    eval_path = "data/golden_eval_set.json"
    if not os.path.exists(eval_path):
        raise HTTPException(status_code=404, detail="Evaluation dataset not found")

    with open(eval_path, "r", encoding="utf-8") as f:
        gold_items = json.load(f)

    engine = get_engine()
    queue = []

    # Take representative slice of test items
    slice_items = gold_items[:limit]
    for item in slice_items:
        pred = engine.process_query(item["customer_query"])
        if filter_status and pred.escalation.status != filter_status:
            continue
        queue.append({
            "id": item["id"],
            "customer_query": item["customer_query"],
            "prediction": pred.model_dump(),
            "gold_should_escalate": item["gold_should_escalate"],
            "is_cold_case": item.get("is_cold_case", False),
            "is_borderline": item.get("is_borderline", False)
        })

    return {"count": len(queue), "messages": queue}


@app.post("/api/predict", response_model=SupportPredictionResponse)
def predict_query(req: PredictionRequest):
    """Process real-time customer query through calibrated pipeline."""
    engine = get_engine()
    return engine.process_query(req.query, force_live_llm=req.force_live_llm or False)


@app.get("/api/evaluation")
def get_evaluation_metrics():
    """Return comprehensive evaluation results and baseline comparison."""
    eval_res_path = "data/eval_results.json"
    if not os.path.exists(eval_res_path):
        raise HTTPException(status_code=404, detail="Evaluation results not found. Run backend/eval/run_eval.py first.")
    with open(eval_res_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/calibration_curve")
def get_calibration_curve():
    """Return reliability diagram and coverage-risk curve data points for interactive UI rendering."""
    eval_res_path = "data/eval_results.json"
    if not os.path.exists(eval_res_path):
        raise HTTPException(status_code=404, detail="Evaluation results not found")
    with open(eval_res_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cal_data = data["systems"]["calibrated_trust_first"]["calibration"]
    cov_risk = data["systems"]["calibrated_trust_first"]["coverage_risk_curve"]
    baseline_cal = data["systems"]["simple_baseline"]["calibration"]

    return {
        "calibrated_threshold": 0.62,
        "trust_first": cal_data,
        "simple_baseline": baseline_cal,
        "coverage_risk_curve": cov_risk
    }
