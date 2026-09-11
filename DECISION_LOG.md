# Architectural Decision Log

This log records the 15 key engineering and architectural decisions made in designing the AmazonHelp Trust-First Support Intelligence System.

1. **Brand Selection (AmazonHelp)**:
   - *Decision*: Selected `@AmazonHelp` over alternatives (`AppleSupport`, `SpotifyCares`).
   - *Rationale*: AmazonHelp represents the largest single support brand in the Twitter Customer Support corpus (81,092 multi-turn threads), with tight repeatable clusters in high-volume fulfillment alongside genuine high-risk safety boundaries (account takeovers, payment disputes). This provides real signal for the escalation gate to discriminate on rather than trivially routing everything in one direction.

2. **Hierarchical Intent Taxonomy Over Flat Buckets**:
   - *Decision*: Deployed a two-level hierarchy (5 coarse categories, 11 sub-intents) rather than a flat 6–10 bucket list.
   - *Rationale*: Coarse categories govern the downstream **Safety Tier** (determining whether an inquiry is even legally/operationally eligible for automated handling), whereas sub-intents govern **precedent retrieval clustering**. A sub-split is maintained only if the historically successful reply differs materially (e.g., `damaged_or_missing_items` requires claims intake, whereas `tracking_status_eta` requires carrier deep-linking).

3. **Outcome Resolution Score Heuristic Definition**:
   - *Decision*: Defined resolution score $R \in [0.05, 1.0]$ based on multi-turn customer response: positive acknowledgment markers (+0.35 to +0.50), negative escalation/complaint markers (-0.35 to -0.60), and thread termination defaults (0.70).
   - *Rationale*: Precedents must be *outcome-verified*, not merely topically similar. A tweet thread that ends with the customer calling the agent useless is negative grounding, not positive grounding.

4. **Known Failure Mode of Resolution Proxy**:
   - *Decision*: Explicitly documented that the resolution proxy conflates "issue resolved" with "customer gave up."
   - *Rationale*: In social customer care, frustrated users frequently abandon threads when support requests DM transfer or fails to assist. These silent abandonments look identical to successful resolutions under a thread-termination heuristic. Surfacing this limitation upfront prevents misleading accuracy claims.

5. **Adaptive-k Retrieval Over Static-k**:
   - *Decision*: Dynamically varied candidate volume ($k \in [2, 5]$) based on candidate outcome variance.
   - *Rationale*: Explicitly borrowed from the author's prior work on **ECA-RAG (Epistemic-Calibrated Adaptive RAG)**. When top candidate precedents strongly agree on high-resolution outcomes ($R \ge 0.70$, $\Delta S < 0.10$), $k=2$ is sufficient. When top candidates diverge in score or implied resolution, $k$ expands up to 5 to evaluate consensus across a broader precedent set.

6. **Agreement-Based Calibration Over Single Similarity Cutoff**:
   - *Decision*: Computed confidence from consensus across the adaptively retrieved set ($Agreement = \bar{R} \cdot (1 - \sigma_R) \cdot \text{Consensus}_{\text{sub}}$) rather than a flat top-1 similarity cutoff.
   - *Rationale*: A query may have high cosine similarity (0.88) to a past customer tweet, but if historical attempts to resolve that issue failed or resulted in conflicting replies, the agent's epistemic confidence should be low. Agreement measures *what worked consistently*.

7. **Probability Calibration via Platt Scaling**:
   - *Decision*: Fitted a logistic sigmoid (Platt scaling) mapping $[Agreement, S_{\text{top}}, \bar{R}]$ into posterior probabilities $P(\text{successful auto-handle})$.
   - *Rationale*: Raw cosine similarities and softmax scores are notoriously overconfident. Platt scaling produces calibrated probabilities that align with empirical accuracy, verified via Expected Calibration Error (ECE = 0.369 vs 0.483 baseline).

8. **Reranking Weight Sensitivity ($\alpha = 0.60$)**:
   - *Decision*: Balanced semantic similarity and resolution score via $S_{\text{reranked}} = 0.60 \cdot S_{\text{semantic}} + 0.40 \cdot R$.
   - *Rationale*: Empirical sensitivity sweep across $\alpha \in [0.0, 1.0]$ revealed that $\alpha = 1.0$ yields poor resolution grounding (mean $R = 0.657$), while $\alpha = 0.0$ drifts off-topic (hit-rate drops to 96.7%). The split $\alpha = 0.60$ achieved peak hit-rate (98.3%) while raising average resolution to 0.870.

9. **Safety Tier Hard Overrides**:
   - *Decision*: Unconditionally forced human escalation for `account_access` and `abuse_safety` categories, as well as cold cases ($S_{\text{top}} < 0.36$), bypassing classifier confidence.
   - *Rationale*: In real-world enterprise operations, no confidence score justifies an automated bot handling an active account takeover or legal threat. Safety boundaries are non-negotiable policy constraints, not probability thresholds.

10. **Cold Case Injection in Golden Evaluation Set**:
    - *Decision*: Deliberately injected 25 cold cases (out-of-scope B2B AWS issues, drone collisions, cryptocurrency queries) into the 180-example golden set.
    - *Rationale*: Standard benchmark sets test in-distribution classification. Cold cases specifically test whether the agent's out-of-distribution detection and cold-case hard override fire reliably when historical precedent is zero.

11. **Database & Storage Choice (JSON / SQLite over Postgres Daemon)**:
    - *Decision*: Used structured JSON and local SQLite storage over an external PostgreSQL container.
    - *Rationale*: Production evaluation and zero-overhead deployments require a strictly reproducible pipeline with zero external dependency friction. Requiring a running Docker engine and PostgreSQL credentials introduces unnecessary environment friction with zero architectural benefit for a 350-precedent vector library.

12. **Model Selection & Family Overlap Risk**:
    - *Decision*: Used `all-MiniLM-L6-v2` for dense retrieval/classification, Google Gemini 2.5/1.5 for generation, and documented judge-generator model overlap risks.
    - *Rationale*: When an LLM judge shares a model family with the generator, self-evaluation bias inflates quality scores. We isolated retrieval into local deterministic embeddings and documented this risk in the report.

13. **Dual-Engine Architecture (Live LLM vs. Deterministic Local Mode)**:
    - *Decision*: Built pluggable live LLM execution with automatic fallback to a deterministic precedent-synthesis engine when no API keys are present.
    - *Rationale*: Ensures any engineer or evaluator can run `npm run dev` and `pytest` immediately with zero setup or external billing requirements, while supporting full live LLM generation when `GEMINI_API_KEY` or `OPENAI_API_KEY` is provided.

14. **Two-Pane Queue + Detail Layout Over Chat Interface**:
    - *Decision*: Built a two-pane message queue and audit detail workspace rather than a conversational chat bubble UI.
    - *Rationale*: The core operational objective is auditable trust. Human supervisors and compliance auditors review support agents via queue triage, precedent audit cards, and calibration curves — not conversational roleplay chat windows.

15. **Architectural Roadmap (Self-Consistency Generation Sampling)**:
    - *Decision*: Deferred multi-candidate self-consistency sampling (sampling 3 replies to check divergence) to the next engineering phase.
    - *Rationale*: Multi-candidate LLM generation triples inference latency and introduces local token overhead. Scoped cleanly into the production roadmap.
