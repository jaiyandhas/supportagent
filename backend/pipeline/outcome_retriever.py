"""Outcome-Weighted Precedent Retrieval Engine for AmazonHelp.

Stage 1: Candidate retrieval via FAISS (top-N=20).
Stage 2: Outcome reranking combining semantic similarity and historical thread resolution score.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import faiss
import numpy as np

from backend.models.schemas import PrecedentCandidate
from backend.pipeline.classifier import get_embedder


class OutcomeWeightedRetriever:
    def __init__(
        self,
        precedents_path: str = "data/historical_precedents.json",
        alpha: float = 0.60,
        candidate_pool_size: int = 20
    ):
        self.precedents_path = precedents_path
        self.alpha = alpha  # Weight for semantic similarity vs (1 - alpha) for resolution score
        self.candidate_pool_size = candidate_pool_size
        self.embedder = get_embedder()

        self.precedents: List[Dict] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self._build_index()

    def _build_index(self):
        """Load precedents and build normalized FAISS IndexFlatIP."""
        if not os.path.exists(self.precedents_path):
            raise FileNotFoundError(f"Precedents file not found: {self.precedents_path}")

        with open(self.precedents_path, "r", encoding="utf-8") as f:
            self.precedents = json.load(f)

        queries = [p["customer_query"] for p in self.precedents]
        embeddings = self.embedder.encode(queries, convert_to_numpy=True, show_progress_bar=False)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def retrieve_candidates(
        self,
        query: str,
        target_sub_intent: Optional[str] = None,
        top_k: int = 5,
        custom_alpha: Optional[float] = None
    ) -> List[PrecedentCandidate]:
        """Execute two-stage retrieval: candidate fetch followed by outcome reranking."""
        alpha = custom_alpha if custom_alpha is not None else self.alpha
        q_emb = self.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(q_emb)

        # Stage 1: Retrieve top-N candidates from FAISS
        pool_n = min(self.candidate_pool_size, len(self.precedents))
        distances, indices = self.index.search(q_emb, pool_n)

        raw_indices = indices[0]
        raw_sims = distances[0]

        candidates = []
        for rank, (idx, sim) in enumerate(zip(raw_indices, raw_sims)):
            if idx < 0 or idx >= len(self.precedents):
                continue
            prec = self.precedents[idx]
            sim_score = float(sim)
            res_score = float(prec.get("resolution_score", 0.50))

            # Intent boost bonus (0.05) if candidate matches predicted sub-intent
            intent_boost = 0.05 if (target_sub_intent and prec.get("sub_intent") == target_sub_intent) else 0.0

            # Stage 2: Outcome Reranking Formula
            reranked = (alpha * sim_score) + ((1.0 - alpha) * res_score) + intent_boost

            candidate = PrecedentCandidate(
                precedent_id=prec["precedent_id"],
                customer_query=prec["customer_query"],
                support_reply=prec["support_reply"],
                customer_followup=prec.get("customer_followup"),
                resolution_score=round(res_score, 2),
                resolution_rationale=prec.get("resolution_rationale", ""),
                coarse_intent=prec.get("coarse_intent", "order_delivery"),
                sub_intent=prec.get("sub_intent", "tracking_status_eta"),
                semantic_similarity=round(sim_score, 3),
                reranked_score=round(reranked, 3)
            )
            candidates.append(candidate)

        # Sort by reranked score descending
        candidates.sort(key=lambda c: c.reranked_score, reverse=True)

        return candidates[:top_k]
