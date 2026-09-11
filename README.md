# AmazonHelp Trust-First Support Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/frontend-React%20%2B%20TypeScript-61DAFB.svg)](https://react.dev/)
[![TailwindCSS](https://img.shields.io/badge/styling-Tailwind%20v4-38B2AC.svg)](https://tailwindcss.com/)
[![Tests](https://img.shields.io/badge/tests-10%20passed-success.svg)]()

A trust-first customer support agent for `@AmazonHelp` built around two core engineering foundations:
1. **Outcome-Verified Grounding**: A precedent reply only qualifies as grounding if multi-turn thread evidence shows the customer's issue was actually resolved (measured via post-reply sentiment delta and acknowledgment markers), rather than mere topical similarity.
2. **Escalation as a Calibrated Trust Gate**: Escalation routing is formulated as a probability calibration problem (*"does this agent know what it doesn't know?"*), employing adaptive-k consensus retrieval and Platt-scaled calibration with a verifiable coverage/risk curve and hard safety overrides.

---

## ⚡ Quickstart (Zero-Friction Local Setup)

The entire pipeline is self-contained and reproducible offline using bundled fixture data (`data/fixtures/amazonhelp_sample.jsonl`) sampled directly from the 81,092-thread AmazonHelp Twitter dataset.

### 1. Prerequisites
- Python 3.11+ (managed via `uv` or `python3 -m venv`)
- Node.js v18+ & npm

### 2. Setup Python Environment & Dependencies
```bash
# Clone the repository
git clone https://github.com/jaiyandhas/supportagent.git
cd supportagent

# Create virtual environment and install pinned dependencies
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .
```

### 3. Run Ingestion & Thread Reconstruction (Sample Fixture)
```bash
# Ingest raw threads, reconstruct multi-turn turns, and compute outcome resolution scores
python backend/ingest/pull_amazonhelp_threads.py --source fixture
```
*(To stream the full 81,000+ Twitter dataset directly from HuggingFace, run with `--source remote`.)*

### 4. Run Automated Evaluation Harness
```bash
# Runs full benchmark across Trivial, Simple, and Calibrated agents on 180 golden cases
python backend/eval/run_eval.py
```

### 5. Run Unit & Pipeline Tests
```bash
pytest backend/tests/ -v
```

### 6. Launch Backend API & Demo UI
```bash
# Terminal 1: Launch FastAPI backend (port 8000)
uvicorn backend.app:app --host 127.0.0.1 --port 8000

# Terminal 2: Launch React + TypeScript Demo UI (port 5173)
cd frontend
npm install
npm run dev
```
Open **[http://127.0.0.1:5173](http://127.0.0.1:5173)** in your browser to interact with the auditable queue, precedent strip, and live trust panel!

---

## 🤖 Dual-Engine Architecture (Live LLM vs. Deterministic Local Mode)

To ensure immediate turnkey evaluation and reliable offline benchmarking without external API rate limits or credit card requirements:
- **Mode 1: Deterministic Local Mode (Zero API Keys Required)**: Enabled by default. Hierarchical intent classification runs via local `all-MiniLM-L6-v2` embeddings, retrieval operates via FAISS, and generation uses a high-fidelity deterministic template engine adapting verified precedent replies.
- **Mode 2: Live LLM Mode**: If `GEMINI_API_KEY` or `OPENAI_API_KEY` is present in your environment, the system automatically upgrades to live generation (Google Gemini 2.5/1.5 or OpenAI GPT-4o-mini).
- The active mode is prominently surfaced in the UI header via an environment status pill (`[⚡ Gemini Live]` vs `[🔒 Repro Mode]`).

---

## 📊 Benchmark Results Summary

Evaluated on 180 golden cases stratified across 5 coarse categories, 11 sub-intents, 25 cold cases, and 10 complex edge disputes:

| Metric | Trivial Baseline | Simple Baseline | Trust-First Agent (Ours) |
| :--- | :---: | :---: | :---: |
| **Coarse Accuracy** | 0.706 | 0.756 | **0.728** |
| **Sub-Intent Macro F1** | 0.062 | 0.131 | **0.240 (+83%)** |
| **Escalation AUROC** | 0.500 | 0.573 | **0.754 (+32%)** |
| **Expected Calibration Error (ECE) ↓** | 0.000 | 0.483 | **0.369 (-24% Better)** |
| **Brier Score Loss ↓** | 0.200 | 0.386 | **0.272 (-30% Better)** |
| **Mean Precedent Resolution Score** | — | 0.66 | **0.87 (Outcome Reranked)** |
| **LLM-Judge Overall Score (1-5)** | 4.42 | 4.67 | **4.74** |

---

## 📁 Repository Structure

```
supportagent/
├── data/
│   ├── taxonomy.json                # Two-level intent hierarchy & safety tiers
│   ├── fixtures/
│   │   └── amazonhelp_sample.jsonl  # 1,200 real multi-turn AmazonHelp threads
│   ├── historical_precedents.json   # 350 outcome-scored precedent records
│   ├── golden_eval_set.json         # 180 stratified evaluation cases (incl. cold cases)
│   └── eval_results.json            # Cached evaluation results & calibration points
├── backend/
│   ├── ingest/
│   │   └── pull_amazonhelp_threads.py # Real data fetch, thread parser & cluster check
│   ├── pipeline/
│   │   ├── classifier.py            # Hierarchical coarse & sub-intent classifier
│   │   ├── outcome_retriever.py     # FAISS top-20 + outcome reranking (alpha=0.6)
│   │   ├── escalation_gate.py       # Adaptive-k consensus, Platt calibration, hard overrides
│   │   ├── generator.py             # Dual-engine LLM & deterministic grounded generator
│   │   └── engine.py                # Unified support agent orchestrator
│   ├── eval/
│   │   ├── baselines.py             # Trivial & Simple baseline implementations
│   │   ├── metrics.py               # ECE, Brier score, coverage/risk, per-class F1
│   │   ├── judge.py                 # LLM-as-judge rubric & Cohen's kappa agreement
│   │   └── run_eval.py              # Master evaluation runner
│   ├── models/
│   │   └── schemas.py               # Pydantic data contracts
│   ├── app.py                       # FastAPI application & REST endpoints
│   └── tests/
│       ├── test_ingest.py           # Unit tests for tweet cleaning & resolution scoring
│       └── test_pipeline.py         # Integration tests for gate, overrides & FAISS
├── frontend/                        # React + TypeScript + Tailwind CSS (Vite)
│   ├── src/
│   │   ├── App.tsx                  # Two-pane layout with audit detail & eval modal
│   │   ├── index.css                # Apple-like design tokens
│   │   ├── components/
│   │   │   ├── MessageQueue.tsx     # Filterable queue with intent & status chips
│   │   │   ├── MessageDetail.tsx    # Message view, drafted reply & precedent strip
│   │   │   ├── PrecedentStrip.tsx   # Precedent cards with resolution score tags
│   │   │   └── TrustPanel.tsx       # Calibrated confidence & live reliability diagram
│   │   └── types.ts
│   ├── package.json
│   └── vite.config.ts
├── REPORT.md                        # Architecture, benchmark & calibration technical report
├── DECISION_LOG.md                  # 14 detailed architectural decision bullets
├── CITATIONS.md                     # Formal citations (ECA-RAG self-citation, dataset, etc.)
└── pyproject.toml
```

---

## 📖 Key Documentation Artifacts
- **[REPORT.md](file:///Users/jaiyandh/Projects/supportagent/REPORT.md)**: Full technical report covering thesis, comparative baseline benchmarks, 5 failure modes, production metric caveats (*"What is misleading about the headline number?"*), and roadmap.
- **[DECISION_LOG.md](file:///Users/jaiyandh/Projects/supportagent/DECISION_LOG.md)**: 14 detailed bullets covering brand choice, safety tiers, resolution heuristic definition, ECA-RAG adaptive-k reuse, alpha sensitivity study, and cold case injection.
- **[CITATIONS.md](file:///Users/jaiyandh/Projects/supportagent/CITATIONS.md)**: Explicit citations for ECA-RAG, the Twitter support dataset, Platt scaling, FAISS, and SBERT.
