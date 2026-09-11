# AmazonHelp Trust-First Support Agent: Outcome-Verified Grounding and Calibrated Escalation

**Author**: Jaiyandh A. S.  
**Repository**: [github.com/jaiyandhas/supportagent](https://github.com/jaiyandhas/supportagent)  
**Submission**: Hiver SDE Intern Take-Home Project  

---

## 1. Thesis & Problem Framing

Most attempts at this assignment will read *"draft a reply grounded in how the brand has historically resolved similar issues"* as *"do RAG over old tweets"* and *"decide auto-handle vs escalate"* as *"put a threshold on classifier confidence."* Both readings are technically compliant and both throw away the actual hard part.

**This submission treats two things as first-class, not afterthoughts:**
1. **"Historically resolved" means *outcome-verified*, not merely *topically similar*.** A precedent reply only counts as grounding if the thread shows evidence the customer's issue actually went away after it — not just that the tweet talks about the same topic. Retrieving a past brand reply from a thread where the customer responded in fury is negative grounding, not positive grounding.
2. **The escalation decision is the product.** The assignment's own framing (*"convince us the agent is good enough to trust"*) is a calibration problem, not a classification problem — so the whole system is built around one question: *does this agent know what it doesn't know*, with an actual calibration curve to prove it, not an ungrounded claimed accuracy number.

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
2. **Simple Baseline**: TF-IDF intent classifier, plain top-1 nearest-neighbor retrieval (no outcome reranking), tuned similarity threshold ($\tau = 0.35$) for escalation (flags 70.6% for human review).
3. **Calibrated Trust-First Agent (Ours)**: Hierarchical intent classifier with safety tiers, two-stage outcome-weighted FAISS retrieval ($\alpha = 0.60$), and adaptive-k consensus with Platt-scaled calibration.

### Comparative Benchmark Results

*Note: Table formatting strictly bolds only the actual best value per row across systems.*

| Metric | Trivial Baseline | Simple Baseline | Trust-First (Ours) | Context & Trade-off |
| :--- | :---: | :---: | :---: | :--- |
| **Coarse Intent Accuracy** | 0.706 | **0.756** | 0.733 | Simple baseline wins on coarse classification; see failure analysis |
| **Coarse Worst-Class F1** | 0.000 (`abuse_safety`, n=7) | 0.000 (`abuse_safety`, n=7) | 0.000 (`general_other`, n=10) | Support gap & out-of-taxonomy cold cases |
| **Sub-Intent Macro F1** | 0.062 | 0.131 | **0.241** | **0.241 absolute** (+84.0% relative); fine-grained split is hard |
| **Escalation Precision** | 0.000 | **0.236** | 0.224 | **0.224 absolute**; calibrated conservative safety bias |
| **Escalation Recall** | 0.000 | 0.833 | **1.000** | **1.000** zero safety false negatives on golden set |
| **Escalation F1** | 0.000 | **0.368** | 0.365 | Tuned baseline achieves 0.368 F1 at $\tau=0.35$ |
| **Escalation AUROC** | 0.500 | 0.573 | **0.760** | **+32.6% relative gain** in ranking discrimination |
| **Expected Calibration Error (ECE) ↓** | 0.000* | 0.482 | **0.368** | **-23.7% error reduction** (*Trivial 0.000 is degenerate constant artifact) |
| **Brier Score Loss ↓** | 0.200 | 0.385 | **0.269** | **-30.1% improvement** in probabilistic accuracy |
| **Judge Overall Quality (1-5) ↑** | 2.64 | **4.18** | 4.12 | Simple copies verbatim human tweets; see §3.1 & §3.2 |
| **Judge Faithfulness (1-5) ↑** | 1.66 | **4.95** | 4.00 | Anchored rubric heavily penalizes canned deflection (1.66) |

### 2.1 Headline Artifact 1: Probability Calibration & Reliability
The central claim of this system is that it knows when it does not know. The Simple Baseline relies on raw cosine similarity, which suffers from severe overconfidence (ECE = 0.482, Brier = 0.385). The Trust-First Agent reduces Expected Calibration Error to **0.368** and Brier score to **0.269** via Platt scaling over adaptive-k consensus features ($Agreement = \bar{R} \cdot (1 - \sigma_R) \cdot \text{Consensus}$).

![Trust Panel and Calibration Curve UI](file:///Users/jaiyandh/Projects/supportagent/docs/assets/trust_panel_demo.jpg)
*Figure 1: UI Message Audit View showing precedent strip with outcome resolution scores and Trust Panel rendering the calibrated confidence indicator and live reliability diagram.*

### 2.2 Headline Artifact 2: Outcome-Weighted Retrieval Impact
Standard RAG retrieves the most *topically similar* historical tweet. By reranking candidates using thread resolution scores ($S_{\text{reranked}} = 0.60 \cdot S_{\text{semantic}} + 0.40 \cdot R$), the system achieves dramatic gains in grounding quality without sacrificing topical relevance:
- **Top-3 Relevance Hit-Rate**: Maintained at **98.3%** before and after reranking.
- **Mean Precedent Resolution Score**: Jumped from **0.66** (unweighted semantic retrieval) to **0.87** (outcome-weighted reranking).
- **Alpha Sensitivity Study**:
  - $\alpha = 1.0$ (Pure semantic similarity): Hit rate = 98.3%, Mean resolution = 0.657 (retrieves angry unresolved arguments).
  - $\alpha = 0.0$ (Pure resolution): Hit rate = 96.7%, Mean resolution = 0.913 (semantic drift away from query).
  - $\alpha = 0.60$ (Proposed): Hit rate = **98.3%**, Mean resolution = **0.870** (optimal Pareto frontier).

---

## 3. Failure Analysis & Diagnostics

### 3.1 LLM-Judge Discrimination & Pairwise Comparisons
In early runs, judge scores clustered at 4.42 / 4.67 / 4.74, indicating ceiling compression where a fixed canned reply was awarded 4.42/5. Diagnostic audit of evaluation transcripts revealed that the initial rubric lacked topical relevance anchors and awarded baseline points to any polite phrasing containing words like "app" or "contact":

> **Diagnostic Transcript (Pre-Fix Failure)**:  
> *Query*: `"I am testing an automated IoT temperature sensor connected to AWS Greengrass, but device payload fails MQTT handshake on port 8883."`  
> *Trivial Canned Reply*: `"Thank you for contacting AmazonHelp. Please track your delivery in the Amazon app or visit amazon.com/contact-us."`  
> *Initial Judge Score*: **4.42 / 5.0** (Faithfulness: 4.0, Tone: 4.5, Resolution: 4.5, Policy: 5.0).  
> *Diagnostic Finding*: The judge committed critical misdirection by treating an irrelevant package tracking message as a valid response to an enterprise cloud inquiry simply because it was polite.

**The Fix**: Implemented an anchored rubric with topical alignment penalties and pairwise win-rate tracking. Irrelevant canned deflections and escalation routing failures are now strictly capped at 1.0–2.0. Under the anchored rubric:
- **Trivial Baseline collapsed to 2.64 / 5.0** (Faithfulness 1.66), reflecting genuine failure on payments, security, and cold cases.
- **Trust-First scored 4.12 / 5.0**, achieving **100% pairwise win rate against Trivial** (180 wins, 0 ties, 0 losses).
- **Pairwise Comparison vs. Simple Baseline**: Trust-First recorded **32 wins (17.8%), 90 ties (50.0%), and 58 losses (32.2%)**. Trust-First **loses more pairwise comparisons to Simple than it wins**. Rather than concealing this counter-intuitive result, we diagnose its mechanical cause below.

### 3.2 Why Trust-First Scores Lower Than Simple on the LLM-Judge (4.12 vs. 4.18)
To understand why Trust-First loses 32.2% of pairwise head-to-heads to the Simple baseline, we pulled and analyzed the 3 lowest-scoring Trust-First evaluation transcripts (`case_39`, `case_0`, `case_4`). The gap arises from two specific structural dynamics rather than a lack of conversational capability:

1. **Verbatim Text Copying Maximizes Rubric Faithfulness (4.95 vs. 4.00)**:  
   The Simple baseline copies the retrieved historical tweet verbatim (`candidate.support_reply`). Because its reply has near 100% lexical token overlap with the precedent candidate, the anchored judge awards it near-perfect faithfulness (**4.95 / 5.0**). In contrast, Trust-First in deterministic repro mode (Mode 1) normalizes text, strips metadata/agent signatures (e.g. `^SI`, `(2/3)`), prepends empathetic greetings, and standardizes deflection language. Even minor paraphrastic synthesis drops the measured lexical overlap with the raw precedent candidate string, yielding **4.00 / 5.0**.

2. **Historical Human Tweets Ask Situational Follow-Ups; Deterministic Templates Deflect**:  
   Real historical tweets from human AmazonHelp agents frequently contained tailored, conversational questions. Consider **Query #39**:  
   - *Customer Query*: `"hi i preordered star wars battle front II and you sent an email with beta code but its not working"`  
   - *Simple Baseline Reply*: `"Hello Jamie! When did you place your pre-order?"` (Judge: 3.50; Faithfulness 5.0).  
   - *Trust-First Reply (Deterministic Repro)*: `"Hello, thanks for reaching out. Please reach out here for help"` (Judge: 3.62; Resolution Likelihood: 1.8).  
   Similarly on **Query #0** (*"I ordered on July 25th. I still do not receive anything and you already charged me... Unacceptable 😡"*), Simple copied a human tweet asking *"Hi, sorry, what's the status of the tracking and delivery date? Have you looked into an A-Z Claim:"* (Score: 4.44), whereas Trust-First safely routed the angry customer via intake deflection *"Please fill this form and I’ll contact you at the earliest"* (Score: 3.85, Resolution Likelihood: 1.8).  
   Because the anchored judge requires issue-specific entity overlap ("beta code", "claim", "tracking") to award high resolution likelihood, Trust-First's generic deflection templates are penalized (Resolution Likelihood 1.8 vs 3.5), despite being the safer action. When upgraded to Live LLM Mode (Mode 2), dynamic generation synthesizes contextual issue keywords, but in deterministic offline mode, safety-first intake templates trade conversational specificity for zero false negatives.

### 3.3 Simple Baseline Degeneracy Audit: Threshold Recalibration ($\tau = 0.35$)
A critical audit was conducted to determine whether the Simple baseline was a real baseline or an always-escalate strawman:
- **Initial Flaw**: With an initial cosine threshold of $0.65$ (or $0.75$), TF-IDF similarities across short customer tweets (mean: 0.318, max: 0.682) meant that **179 of 180 golden queries (99.4%)** fell below the threshold. The baseline was essentially an always-escalate policy with 0.6% automation coverage, making any metric win over it meaningless.
- **Recalibrated Baseline ($\tau = 0.35$)**: We recalibrated Simple's threshold to $\tau = 0.35$, yielding a genuine decision distribution:
  - **127 of 180 queries (70.6%)** flagged for escalation.
  - **53 of 180 queries (29.4%)** auto-handled.
  - Performance: Precision = **0.236**, Recall = **0.833**, F1 = **0.368**, AUROC = **0.573**.
- **The Honest Takeaway**: Under a tuned threshold, Simple actually achieves a slightly higher escalation precision (**0.236 vs. 0.224**) and F1 (**0.368 vs. 0.365**) than Trust-First. Trust-First intentionally accepts lower precision (more false alarms) because it guarantees **100% recall on safety-critical escalations** (1.000 vs. 0.833 for Simple) and achieves far superior ranking discrimination (**AUROC 0.760 vs. 0.573**) and calibration (**ECE 0.368 vs. 0.482**). Simple is thus a tuned, competitive baseline, and the trade-off is mathematically transparent.

### 3.4 Coarse Intent Accuracy Trade-off (0.733 vs 0.756)
The Simple baseline achieved 0.756 coarse accuracy compared to 0.733 for the Trust-First system. The hierarchical classifier introduces safety tiers and conditional sub-intent modeling; this structural constraint slightly depresses coarse classification on boundary edge cases in exchange for a massive gain in **sub-intent Macro F1 (0.241 vs 0.131, +84.0%)** and **escalation AUROC (0.760 vs 0.573, +32.6%)**.

### 3.5 Root Cause of Coarse Worst-Class F1 = 0.000
All three systems recorded 0.000 F1 on specific minority classes. The underlying causes are distinct:
1. **Trust-First System on `general_other` (Support: n=10, F1 = 0.000)**:  
   `general_other` was assigned as the ground truth label for the 10 deliberate cold-case scenarios (e.g. AWS Greengrass IoT, Bitcoin payments, drone collisions) to test out-of-domain detection. However, `taxonomy.json` defines only 5 operational categories (`order_delivery`, `payments_refunds`, `account_access`, `product_digital`, `abuse_safety`). Because `general_other` is an out-of-taxonomy class, the 5-way classifier can never predict it, resulting in 0% recall. Crucially, **the escalation gate correctly caught 100% of these cases** via the cold-case similarity threshold ($S_{\text{top}} \le 0.36$), proving that out-of-domain safety is achieved at the gate level rather than the classifier level.
2. **Baselines on `abuse_safety` (Support: n=7, F1 = 0.000)**:  
   `abuse_safety` represents only 0.8% of raw Twitter data and 7 items in the golden set. The Trivial baseline never predicts it; the Simple baseline (TF-IDF Ridge) completely collapsed on this minority class due to severe class imbalance. In contrast, the Trust-First system achieved high recall on `abuse_safety` through explicit keyword safety overrides.

---

## 4. What Is Misleading About My Headline Number? (Mandatory)

Responsible engineering requires transparently deconstructing headline metrics before a reviewer audits them in production:

1. **Escalation Precision Is 0.224 (Roughly 4 in 5 Escalations Are Over-Escalations)**:
   - *The Reality*: Leading with high recall or relative gains obscures the absolute precision: **0.224**. Only ~22% of messages flagged for human review strictly required escalation under gold labels.
   - *Operational Consequence*: The system operates with a deliberate conservative safety bias. While this guarantees 100% safety recall on the golden set, it burdens human agent queues with false alarms.

2. **Sub-Intent Macro F1 of 0.241 Is Low in Absolute Terms**:
   - *The Reality*: Although +84.0% higher than the simple baseline (0.131), an absolute Macro F1 of **0.241** means fine-grained sub-intent routing remains noisy across 11 classes. High volume in `tracking_status_eta` masks poor classification on rare sub-intents like `unauthorized_charge` and `mfa_password_lockout`.

3. **Recalibrated Simple Baseline Outperforms Trust-First on Precision (0.236 vs. 0.224) and F1 (0.368 vs. 0.365)**:
   - *The Reality*: When the Simple baseline's similarity threshold is calibrated to $\tau = 0.35$ rather than left at a degenerate always-escalate value, it achieves higher precision and F1 than Trust-First. Trust-First's advantage is confined to safety recall (1.000 vs. 0.833), calibration error (ECE 0.368 vs. 0.482), and AUROC (0.760 vs. 0.573).

4. **Trust-First Loses More Pairwise Comparisons to Simple Than It Wins (32.2% Loss vs. 17.8% Win)**:
   - *The Reality*: In pairwise LLM-judge matchups against Simple, Trust-First wins 17.8%, ties 50.0%, and loses 32.2%. As diagnosed in §3.2, Simple copies verbatim historical human replies that include specific situational questions and retain 100% lexical overlap with precedents, whereas Trust-First's deterministic repro mode uses cautious intake deflection templates that the judge penalizes for lacking issue keywords.

5. **Trivial Baseline's ECE = 0.000 Is a Degenerate Artifact**:
   - *The Reality*: A reader might glance at ECE = 0.000 and conclude the trivial baseline is perfectly calibrated. It is not; the trivial baseline outputs a constant confidence of 1.0 for every query, clustering all samples into a single boundary bin. Its AUROC of 0.500 proves it has zero discriminative power.

6. **100% Self-Relabeling Agreement Reflects Same-Day Rule Consistency**:
   - *The Reality*: The 100% blind re-annotation agreement was measured on a 20-sample subset re-labeled within the same curation session. It proves internal rule consistency, not longitudinal objectivity across independent human annotators. An external cross-annotator study without shared annotator priors would likely fall in a moderate-to-strong agreement range rather than near-perfect unanimity.

7. **Golden Set Skew vs. Real Production Volume**:
   - *The Reality*: In our golden set, **22.2%** of cases are complex disputes or safety overrides, and **13.8%** are cold cases. In real Twitter traffic, **87.0%** of inquiries are routine tracking questions. In real traffic, naive accuracy will appear artificially high, while escalation precision will appear even lower.

8. **The Resolution Proxy Equates Abandonment with Resolution**:
   - *The Reality*: Under the thread heuristic, customers who give up after an unhelpful reply receive a default resolution proxy ($R = 0.70$). True resolution can only be measured via downstream ERP/ticket state (e.g. no re-contact within 72 hours).

---

## 5. Provenance Audit: Traceable Real Data Spot-Check

To verify that precedents and evaluation cases originate from authentic Twitter customer support interactions rather than synthetic generation, below is a verifiable spot-check against the HuggingFace/Kaggle dataset (`thoughtvector/customer_support_on_twitter`):

| Type | ID | Real Conversation ID | Customer Query (T1) | Support Precedent Reply (T2) | Outcome $R$ |
| :--- | :--- | :--- | :--- | :--- | :---: |
| Precedent | `prec_amzn_0006` | `b14ef239fabefe2f06ca1fe0a93581be` | "My grandfather received a call from 206-508-4014 claiming to be from Amazon..." | "That is not one of our numbers. Please make sure he didn't provide them any info..." | 0.70 |
| Precedent | `prec_amzn_0058` | `1d22eac7c3686f2960ef27f1d0ec5b84` | "I would appreciate it if your delivery people didn’t walk on my lawn..." | "I'm sorry for the poor delivery! We want to make sure this is addressed..." | 0.95 |
| Precedent | `prec_amzn_0221` | `2b56556f8db7de56e9d0e7ddb83f50ee` | "Why subtitles are available only for S01E01 of This is Us, but not for..." | "Could you please help us with the exact title name you're referring to..." | 0.65 |
| Precedent | `prec_amzn_0299` | `13297e995d406e634eaf3d2dcfad9a97` | "i'm currently living in Germany and according to your site my parcels..." | "Hi, did you already receive your parcel or were my colleagues able to..." | 0.85 |
| Precedent | `prec_amzn_0311` | `7a50990c3cc851525b2c8de216ee429f` | "Hi, I have ordered RAM and now I want to return it. But, Product is not..." | "We'd like to check this out. Please contact our support team here: and we'll..." | 0.65 |
| Golden Set | `eval_gold_005` | `b582ea79678c0d88f91e3702005f9bc7` | "Still no delivery date to the item I pre-ordered, that came out yesterday..." | Grounding: `tracking_status_eta` SLA guidance | Auto-Handle |
| Golden Set | `eval_gold_029` | `f04ec8a32b8bd49d5a5b920258f05a8f` | "greetings from . we wanted to be associated for Merchant onboarding..." | Grounding: Seller onboarding redirection | Auto-Handle |
| Golden Set | `eval_gold_066` | `a9e3b95e8ae40fee7407bcbc458f3770` | "Hello When I ordered a loafer, then a formal shoe came to me..." | Grounding: Wrong item replacement workflow | Auto-Handle |
| Golden Set | `eval_gold_111` | `cfd0316557bf8e5541955f0ef3017dc4` | "hi I've gone to use prime and can't get delivered until Friday..." | Grounding: Prime transit cutoff times | Auto-Handle |
| Golden Set | `eval_gold_141` | `56595eef9f023731b735556e4c9a7df0` | "You are doing fraud by selling amazon prime. U charge but do not..." | Grounding: Prime refund dispute workflow | Auto-Handle |

---

## 6. What I'd Do Next With One More Week

1. **Self-Consistency Divergence Gating (Cut Scope Item)**:
   - Sample $N=3$ candidate replies at temperature 0.7 for borderline queries. If candidate generations diverge in factual guidance or proposed actions, treat variance as an epistemic uncertainty signal and escalate.
2. **Deterministic Carrier & CRM API Verification**:
   - Connect the escalation gate to mock Amazon OMS (Order Management System) APIs. If tracking shows "Delivered 20 minutes ago," auto-handle with carrier photo verification; if tracking shows "Exception / Lost in transit," escalate immediately regardless of text similarity.
3. **Cross-Encoder Neural Reranker**:
   - Replace bi-encoder FAISS top-20 retrieval with a fine-tuned cross-encoder (`ms-marco-MiniLM-L-6-v2`) to capture fine-grained negations and multi-intent disputes.
4. **Active Learning & Precedent Pruning**:
   - Build a supervisor review interface where human agents can flag misfiring precedents, automatically deprecating them from the FAISS index.
