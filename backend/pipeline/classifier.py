"""Hierarchical Intent Classifier for AmazonHelp Support Agent.

Predicts coarse category first, then sub-intent conditional on coarse category.
Logs confidence at both levels.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from backend.models.schemas import IntentPrediction

# Singleton embedder
_EMBEDDER = None

def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBEDDER


class HierarchicalIntentClassifier:
    def __init__(self, taxonomy_path: str = "data/taxonomy.json", precedents_path: str = "data/historical_precedents.json"):
        self.taxonomy_path = taxonomy_path
        self.precedents_path = precedents_path
        self.embedder = get_embedder()

        with open(taxonomy_path, "r", encoding="utf-8") as f:
            self.taxonomy = json.load(f)

        self.coarse_categories = list(self.taxonomy["coarse_categories"].keys())
        self.sub_intents_by_coarse = {
            c: list(data["sub_intents"].keys())
            for c, data in self.taxonomy["coarse_categories"].items()
        }

        self.coarse_model: Optional[LogisticRegression] = None
        self.sub_models: Dict[str, LogisticRegression] = {}
        self._train_from_precedents()

    def _train_from_precedents(self):
        """Train lightweight logistic regression models from historical precedents and taxonomy prototypes."""
        if not os.path.exists(self.precedents_path):
            raise FileNotFoundError(f"Precedents file not found: {self.precedents_path}")

        with open(self.precedents_path, "r", encoding="utf-8") as f:
            precedents = json.load(f)

        # Build training examples
        texts = []
        coarse_labels = []
        sub_labels = []

        for p in precedents:
            texts.append(p["customer_query"])
            coarse_labels.append(p["coarse_intent"])
            sub_labels.append(p["sub_intent"])

        # Augment with prototype descriptions from taxonomy for robust generalization
        for c, c_info in self.taxonomy["coarse_categories"].items():
            texts.append(c_info["safety_tier_description"])
            coarse_labels.append(c)
            sub_labels.append(list(c_info["sub_intents"].keys())[0])

            for s, s_info in c_info["sub_intents"].items():
                texts.append(s_info["display_name"] + ". " + s_info["split_rationale"])
                coarse_labels.append(c)
                sub_labels.append(s)

        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # Train coarse model
        self.coarse_model = LogisticRegression(max_iter=500, class_weight='balanced')
        self.coarse_model.fit(embeddings, coarse_labels)

        # Train sub-intent models per coarse category
        for c in self.coarse_categories:
            c_indices = [i for i, label in enumerate(coarse_labels) if label == c]
            if len(c_indices) > 0:
                c_subs = [sub_labels[i] for i in c_indices]
                unique_subs = list(set(c_subs))
                if len(unique_subs) > 1:
                    sub_model = LogisticRegression(max_iter=500, class_weight='balanced')
                    sub_model.fit(embeddings[c_indices], c_subs)
                    self.sub_models[c] = sub_model
                else:
                    self.sub_models[c] = None

    def _check_safety_overrides(self, text: str) -> Optional[Tuple[str, str, float, float]]:
        """High-precision keyword safety overrides for critical threats."""
        t = text.lower()
        if any(w in t for w in ['scam', 'phishing', 'fake call', 'impersonat', 'suspicious number', 'scammer']):
            return "abuse_safety", "phishing_scam_report", 0.99, 0.98
        if any(w in t for w in ['lawyer', 'sue you', 'legal action', 'police', 'court', 'threaten', 'consumer court']):
            return "abuse_safety", "harassment_threatening_conduct", 0.99, 0.98
        if any(w in t for w in ['hacked', 'compromised', 'unauthorized access', 'account taken over']):
            return "account_access", "compromised_account_hijack", 0.99, 0.98
        if any(w in t for w in ['otp', 'mfa', 'locked out of account', 'two-step', 'password reset']):
            return "account_access", "mfa_password_lockout", 0.98, 0.95
        return None

    def predict(self, query: str) -> IntentPrediction:
        """Predict hierarchical intent with dual-level confidence logging."""
        # 1. Check safety rule triggers
        override = self._check_safety_overrides(query)
        if override:
            coarse, sub, c_conf, s_conf = override
            is_override = coarse in ["account_access", "abuse_safety"]
            return IntentPrediction(
                coarse_intent=coarse,
                coarse_confidence=c_conf,
                sub_intent=sub,
                sub_confidence=s_conf,
                is_hard_override=is_override
            )

        # 2. Predict coarse intent with embedding
        q_emb = self.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
        coarse_probs = self.coarse_model.predict_proba(q_emb)[0]
        coarse_classes = self.coarse_model.classes_
        top_coarse_idx = int(np.argmax(coarse_probs))
        pred_coarse = coarse_classes[top_coarse_idx]
        coarse_conf = float(coarse_probs[top_coarse_idx])

        # 3. Predict sub-intent conditioned on coarse
        pred_sub = self.sub_intents_by_coarse[pred_coarse][0]
        sub_conf = 1.0

        if pred_coarse in self.sub_models and self.sub_models[pred_coarse] is not None:
            sub_probs = self.sub_models[pred_coarse].predict_proba(q_emb)[0]
            sub_classes = self.sub_models[pred_coarse].classes_
            top_sub_idx = int(np.argmax(sub_probs))
            pred_sub = sub_classes[top_sub_idx]
            sub_conf = float(sub_probs[top_sub_idx])

        is_hard_override = pred_coarse in ["account_access", "abuse_safety"]

        return IntentPrediction(
            coarse_intent=pred_coarse,
            coarse_confidence=round(coarse_conf, 3),
            sub_intent=pred_sub,
            sub_confidence=round(sub_conf, 3),
            is_hard_override=is_hard_override
        )
