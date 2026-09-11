# Citations & Prior Work Attribution

This system builds upon and explicitly cites the following prior research, datasets, and architectural methodologies:

### 1. Self-Citation: Epistemic-Calibrated Adaptive Retrieval (ECA-RAG)
- **Author Prior Work**: Jaiyandh A. S.
- **Reference Concept**: *ECA-RAG: Epistemic-Calibrated Adaptive Retrieval-Augmented Generation*
- **Specific Technique Reused**:
  - Calibrated confidence-gated adaptive-k retrieval.
  - Instead of pulling a static top-$k$ precedent set, the retrieval volume dynamically expands ($k \in [2, 5]$) when top candidate outcome scores exhibit variance or disagreement, and contracts ($k=2$) when precedents show unanimous resolution consensus.
  - Consensus agreement scoring across the adaptively retrieved set, serving as the primary feature for posterior probability calibration.

### 2. Dataset Provenance: Customer Support on Twitter
- **Curator**: ThoughtVector / Kaggle (`thoughtvector/customer_support_on_twitter`)
- **Hugging Face Mirror**: `TNE-AI/customer-support-on-twitter-conversation`
- **Volume & Brand**: 81,092 real multi-turn conversations for `@AmazonHelp` (highest-volume support brand in the corpus).
- **Turn Structure Used**: Reconstructed $(T_1 \to T_2 \to T_3)$ conversational threads to compute empirical post-reply resolution signals from customer follow-up tweets.

### 3. Probability Calibration: Platt Scaling
- **Citation**: Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*. Advances in Large Margin Classifiers, 10(3), 61–74.
- **Application**: Logistic sigmoid transformation applied over empirical precedent agreement and similarity features to yield well-calibrated posterior probabilities $P(\text{successful auto-handle})$.

### 4. Vector Search & Similarity Indexing: FAISS
- **Citation**: Johnson, J., Douze, M., & Jégou, H. (2017). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data, 7(3), 535–547.
- **Application**: Normalized inner product indexing (`faiss.IndexFlatIP`) for dense cosine similarity retrieval of customer query embeddings.

### 5. Semantic Dense Representations: Sentence-BERT
- **Citation**: Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP).
- **Model Used**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense semantic embeddings).
