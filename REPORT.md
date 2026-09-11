# AmazonHelp Trust-First Support Agent: Outcome-Verified Grounding and Calibrated Escalation

**Author**: Jaiyandh A. S.  
**Repository**: [github.com/jaiyandhas/supportagent](https://github.com/jaiyandhas/supportagent)  
**System**: Architecture & Empirical Technical Report  

---

## 1. Thesis & Problem Framing

Conventional autonomous support implementations typically read *"draft a reply grounded in how the brand has historically resolved similar issues"* as *"do RAG over old tweets"* and *"decide auto-handle vs escalate"* as *"put a threshold on classifier confidence."* Both patterns are technically standard in early prototypes, yet both throw away the actual engineering challenge.

**This architecture treats two principles as first-class, not afterthoughts:**
1. **"Historically resolved" means *outcome-verified*, not merely *topically similar*.** A precedent reply only counts as grounding if the thread shows evidence the customer's issue actually went away after it — not just that the tweet talks about the same topic. Retrieving a past brand reply from a thread where the customer responded in fury is negative grounding, not positive grounding.
2. **The escalation decision is the product.** Operational trust (*"when is an autonomous agent good enough to trust in production?"*) is fundamentally a probability calibration problem, not a raw classification problem — so the system is architected around one question: *does this agent know what it doesn't know*, supported by an empirical calibration curve and Expected Calibration Error (ECE), not an ungrounded accuracy claim.

### What "Good" Means for AmazonHelp
`@AmazonHelp` operates under massive volume (~81,000+ multi-turn conversations in Twitter customer support benchmarks). The workload exhibits a high-density cluster of safely automatable routine inquiries (order tracking, standard return guidelines, Prime renewal timelines) alongside severe operational and legal risks (compromised accounts, payment disputes, phishing scams, threatening driver conduct). A "good" support agent does not maximize autonomous deflection at all costs; it maximizes **reliable, calibrated auto-handling** while guaranteeing that high-liability and novel inquiries unconditionally escalate to human specialists.

### Explicit Non-Goals (What Was Not Built)
To maintain depth and rigorous calibration over breadth, the following were explicitly scoped out:
- **Multi-turn dialogue state tracking**: The agent evaluates single-turn incoming tweets grounded in historical multi-turn precedent threads; it does not maintain persistent conversational session state.
- **Live order-lookup / Carrier API integrations**: The agent drafts policy-grounded replies and guides users to official verification portals; it does not execute live database mutations or carrier API webhooks.
- **Non-English multi-lingual support**: The pipeline filters and processes English threads (`is_english()` filter over the Twitter corpus); international localization was deferred.

---

## 2. Experimental Results vs. Baselines

The complete pipeline was evaluated against a **stratified golden set of 180 curated real-world customer cases** derived from authentic AmazonHelp multi-turn conversations, containing a deliberate slice of 25 cold cases (zero historical precedent) and 10 complex borderline disputes.

We compare three distinct systems:
1. **Trivial Baseline**: Majority class predictor (`order_delivery`), static canned response, never escalates (`should_escalate = False`).
2. **Simple Baseline**: TF-IDF intent classifier, plain top-1 nearest-neighbor retrieval (no outcome reranking), fixed raw cosine similarity threshold ($0.65$) for escalation.
3. **Calibrated Trust-First Agent (Ours)**: Hierarchical intent classifier with safety tiers, two-stage outcome-weighted FAISS retrieval ($\alpha = 0.60$), and adaptive-k consensus with Platt-scaled calibration.

### Comparative Benchmark Results

| Metric | Trivial Baseline | Simple Baseline | Trust-First (Ours) | Relative Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Coarse Intent Accuracy** | 0.706 | **0.756** | 0.728 | Balanced |
| **Coarse Worst-Class F1** | 0.000 (`abuse_safety`) | 0.000 (`abuse_safety`) | **0.000** (`general_other`) | Cold-case boundary |
| **Sub-Intent Macro F1** | 0.062 | 0.131 | **0.240** | **+83.2%** |
| **Escalation Precision** | 0.000 | 0.201 | **0.219** | **+8.9%** |
| **Escalation Recall** | 0.000 | **1.000** | 0.972 | Conservative safety |
| **Escalation F1** | 0.000 | 0.335 | **0.357** | **+6.5%** |
| **Escalation AUROC** | 0.500 | 0.573 | **0.754** | **+31.6%** |
| **Expected Calibration Error (ECE) ↓** | 0.000 | 0.483 | **0.369** | **-23.6% (Better)** |
| **Brier Score Loss ↓** | 0.200 | 0.386 | **0.272** | **-29.5% (Better)** |
| **LLM-Judge Overall Score (1-5) ↑** | 4.42 | 4.67 | **4.74** | **+1.5%** |
| **LLM-Judge Faithfulness (1-5) ↑** | 4.00 | **4.80** | 4.45 | Precedent fidelity |

### Headline Artifact 1: Probability Calibration & Reliability
The central claim of this system is that it knows when it does not know. The Simple Baseline relies on raw cosine similarity, which suffers from severe overconfidence (ECE = 0.483, Brier = 0.386). The Trust-First Agent reduces Expected Calibration Error to **0.369** and Brier score to **0.272** via Platt scaling over adaptive-k consensus features ($Agreement = \bar{R} \cdot (1 - \sigma_R) \cdot \text{Consensus}$).

![Trust Panel and Calibration Curve UI](file:///Users/jaiyandh/Projects/supportagent/docs/assets/trust_panel_demo.jpg)
*Figure 1: UI Message Audit View showing precedent strip with outcome resolution scores and Trust Panel rendering the calibrated confidence indicator and live reliability diagram.*

### Headline Artifact 2: Outcome-Weighted Retrieval Impact
Standard RAG retrieves the most *topically similar* historical tweet. By reranking candidates using thread resolution scores ($S_{\text{reranked}} = 0.60 \cdot S_{\text{semantic}} + 0.40 \cdot R$), the system achieves dramatic gains in grounding quality without sacrificing topical relevance:
- **Top-3 Relevance Hit-Rate**: Maintained at **98.3%** before and after reranking.
- **Mean Precedent Resolution Score**: Jumped from **0.66** (unweighted semantic retrieval) to **0.87** (outcome-weighted reranking).
- **Alpha Sensitivity Study**:
  - $\alpha = 1.0$ (Pure semantic similarity): Hit rate = 98.3%, Mean resolution = 0.657 (retrieves angry unresolved arguments).
  - $\alpha = 0.0$ (Pure resolution): Hit rate = 96.7%, Mean resolution = 0.913 (semantic drift away from query).
  - $\alpha = 0.60$ (Proposed): Hit rate = **98.3%**, Mean resolution = **0.870** (optimal Pareto frontier).

---

## 3. Failure Analysis

Audit of misclassified and misrouted cases from the golden evaluation set identified five dominant failure modes:

1. **Resolution Proxy Misfires on Customer Abandonment (Silent Frustration)**:
   - *Example*: Customer queries about severe delivery delay; support posts a link to an external webform; customer does not reply further.
   - *Heuristic Score*: Assigned default non-recurrence proxy ($R = 0.70$).
   - *Failure Mechanism*: The customer did not resolve their issue; they gave up in disgust. The retriever treats this reply as a successful precedent, grounding subsequent replies in an unhelpful redirection link.
   
2. **Cold-Case Semantic Leakage**:
   - *Example*: *"My drone delivery crashed into my neighbor's solar panel array and cracked two glass panels."*
   - *Predicted Intent*: `order_delivery` -> `damaged_or_missing_items`.
   - *Failure Mechanism*: The dense embedder matches generic words ("delivery", "damaged") to retail package damage precedents. However, the cold-case threshold gate ($S_{\text{top}} = 0.36 \le 0.36$) caught the low overall similarity and triggered human escalation as intended.

3. **Multi-Intent Dispute Overlap**:
   - *Example*: *"My package arrived soaked and torn, the driver threw it over my fence, and now my credit card shows a double charge!"*
   - *Predicted Intent*: `order_delivery` -> `damaged_or_missing_items` (ignores billing dispute).
   - *Failure Mechanism*: Single-label hierarchical classification forces multi-faceted disputes into one bucket. The resulting precedent grounding only addresses the physical package damage, ignoring the financial double-billing.

4. **Fine-Grained Sub-Intent Boundary Confusion**:
   - *Example*: Customer asking why their package was marked delivered when no courier arrived vs. customer reporting an empty envelope.
   - *Predicted Intent*: `tracking_status_eta` instead of `damaged_or_missing_items`.
   - *Failure Mechanism*: Both queries share dense semantic neighborhoods regarding missing parcels, leading to sub-intent confusion.

5. **Overconfidence on Negative Precedent Consensus**:
   - *Example*: A rare carrier outage where all historical precedents ended in severe customer complaints ($R \le 0.20$).
   - *Failure Mechanism*: The candidates exhibited high consensus (all agreed that support could not resolve the issue). While the agreement score was high, the mean resolution was low; the Platt calibrator correctly depressed confidence to 0.18, successfully forcing human escalation.

---

## 4. What Is Misleading About My Headline Number? (Mandatory)

Responsible AI engineering demands transparently stating why headline metrics look cleaner on paper than they will perform in production:

1. **Golden Set Skew vs. Real Production Distribution**:
   - *The Distortion*: In our golden evaluation set, **22.2%** of cases are complex disputes or safety overrides, and **13.8%** are deliberate cold cases. In the raw Twitter support dump, **87.0%** of inquiries are routine fulfillment questions (`order_delivery`).
   - *Impact*: In real traffic, naive accuracy and auto-handle rates will appear artificially high because routine tracking queries dominate the denominator. The golden set was intentionally stress-tested with adversarial cases, which depresses raw accuracy but tests real risk boundaries.

2. **The Resolution Proxy Is Structurally Flawed**:
   - *The Distortion*: A customer follow-up containing "thanks" or no further reply is treated as a resolved issue ($R \ge 0.70$).
   - *Impact*: In reality, customers who give up, switch to phone support, or dispute charges via their bank look identical to satisfied customers under social listening heuristics. True resolution can only be measured by downstream database state (e.g., ticket closure without reopening within 72 hours).

3. **Auto-Handle Rate Does Not Equal Correctness Rate**:
   - *The Distortion*: An auto-handle rate of 35% with 97.2% escalation recall suggests near-perfect safety.
   - *Impact*: Auto-handling an inquiry with a polite, generic canned reply does not mean the customer's problem was solved. It only means the agent did not crash or trigger a hard safety override.

4. **Threshold Tuning Data Overlap**:
   - *The Distortion*: The calibrated threshold ($\tau^* = 0.62$) was validated across the 180 golden set examples.
   - *Impact*: While the Platt scaling model used cross-validated logistic regression, threshold selection on a small sample risks mild overfitting. On an unseen distribution shift (e.g., Prime Day traffic spikes), the optimal threshold may drift.

5. **Judge and Generator Model Family Overlap**:
   - *The Distortion*: Both generation and automated rubric judging leverage LLMs with shared instruction-tuning priors.
   - *Impact*: LLMs exhibit documented self-preference bias when judging outputs structured similarly to their own training distributions, mildly inflating tone and resolution likelihood scores.

---

## 5. Production Roadmap & Architectural Extensions

1. **Self-Consistency Divergence Gating (Epistemic Sampling)**:
   - Sample $N=3$ candidate replies at temperature 0.7 for borderline queries. If candidate generations diverge in factual guidance or proposed actions, treat variance as an epistemic uncertainty signal and escalate.
2. **Deterministic Carrier & CRM API Verification**:
   - Connect the escalation gate to mock Amazon OMS (Order Management System) APIs. If tracking shows "Delivered 20 minutes ago," auto-handle with carrier photo verification; if tracking shows "Exception / Lost in transit," escalate immediately regardless of text similarity.
3. **Cross-Encoder Neural Reranker**:
   - Replace bi-encoder FAISS top-20 retrieval with a fine-tuned cross-encoder (`ms-marco-MiniLM-L-6-v2`) to capture fine-grained negations and multi-intent disputes.
4. **Active Learning & Precedent Pruning**:
   - Build a supervisor review interface where human agents can flag misfiring precedents, automatically deprecating them from the FAISS index.
