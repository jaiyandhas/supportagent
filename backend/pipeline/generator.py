"""Reply Generator for AmazonHelp Support Agent.

Supports dual generation modes:
1. Live LLM Mode: Uses Gemini or OpenAI when API keys are configured.
2. Deterministic Repro Mode: High-fidelity grounded synthesis adapting verified historical precedents without external keys.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
import requests

from backend.models.schemas import EscalationDecision, IntentPrediction, PrecedentCandidate


class ReplyGenerator:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Gemini REST API if key is available."""
        if not self.gemini_api_key:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 250
            }
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass
        return None

    def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI REST API if key is available."""
        if not self.openai_api_key:
            return None
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are AmazonHelp, Amazon's official customer support agent on Twitter. Draft a concise, empathetic, policy-grounded reply."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 200
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
        return None

    def _generate_deterministic_reply(
        self,
        query: str,
        intent: IntentPrediction,
        escalation: EscalationDecision,
        precedents: List[PrecedentCandidate]
    ) -> str:
        """Deterministic grounded synthesis adapting verified historical precedents."""
        if escalation.hard_override_triggered:
            if intent.coarse_intent == "abuse_safety":
                return (
                    "Thank you for bringing this to our attention. Amazon takes trust and security matters very seriously. "
                    "We have routed this report directly to our Trust & Safety security specialists for immediate review. "
                    "Please do not share sensitive personal information or OTPs over untrusted channels."
                )
            elif intent.coarse_intent == "account_access":
                return (
                    "We understand how concerning account access issues are. Because this involves account security and credential protection, "
                    "we have escalated your case directly to our Account Security Specialist team. "
                    "Please check your registered email for secure identity verification steps."
                )

        if escalation.is_cold_case:
            return (
                "We appreciate you reaching out regarding this issue. Because this inquiry involves unique circumstances with no standard automated precedent, "
                "we have escalated your request to a senior customer care specialist to ensure it is handled thoroughly. "
                "An agent will review and reply directly."
            )

        if escalation.should_escalate:
            return (
                "We sincerely apologize for the inconvenience and frustration caused by this issue. "
                "Due to the complexities involved in resolving this matter, we have escalated your thread to a dedicated customer support specialist. "
                "A specialist will inspect your account details and provide direct assistance shortly."
            )

        # Ground on top-outcome precedent
        if precedents:
            top_prec = precedents[0]
            base_reply = top_prec.support_reply

            # Strip raw Twitter scrape artifacts (e.g. ^SI, ^SK, (2/3), dangling colons, person names)
            base_reply = re.sub(r'\^[A-Za-z]{1,4}\b', '', base_reply)
            base_reply = re.sub(r'\(?\b\d+/\d+\)?', '', base_reply)
            base_reply = re.sub(r':\s*(and|or|so|to|we|please)\b', r' \1', base_reply, flags=re.I)
            base_reply = re.sub(r',\s+[A-Z][a-z]+(?=[.!?])', '', base_reply)
            base_reply = re.sub(r'\s+', ' ', base_reply).strip()

            # Polish precedent into clean branded tone
            if not base_reply.startswith(("Hi", "Hello", "We're sorry", "Thanks for reaching")):
                return f"Hello, thanks for reaching out. {base_reply}"
            return base_reply

        return (
            "Thanks for contacting AmazonHelp. Please check your Orders page in the Amazon App for real-time tracking, "
            "or contact our Customer Support team directly via amazon.com/contact-us for personalized assistance."
        )

    def generate(
        self,
        query: str,
        intent: IntentPrediction,
        escalation: EscalationDecision,
        precedents: List[PrecedentCandidate],
        force_live_llm: bool = False
    ) -> Tuple[str, str, Dict]:
        """Generate draft reply grounded in precedents and return (reply, mode, debug_trace)."""
        prompt = (
            f"You are @AmazonHelp, Amazon's official customer support agent on Twitter.\n"
            f"Customer Query: '{query}'\n"
            f"Intent: {intent.coarse_intent} -> {intent.sub_intent}\n"
            f"Escalation Decision: {escalation.status} (Reason: {escalation.stated_reason})\n\n"
            f"Ground your reply strictly in the following historically resolved AmazonHelp precedent:\n"
        )
        if precedents:
            top = precedents[0]
            prompt += f"Precedent Reply: '{top.support_reply}' (Historical Resolution Score: {top.resolution_score})\n"
        prompt += (
            "\nGuidelines: Keep reply under 280 characters, empathetic, polite, actionable, and grounded in verified policy. "
            "Never hallucinate policy or make false refund promises."
        )

        llm_reply = None
        mode = "deterministic_repro"

        if self.gemini_api_key or self.openai_api_key or force_live_llm:
            llm_reply = self._call_gemini(prompt) or self._call_openai(prompt)
            if llm_reply:
                mode = "live_llm"

        final_reply = llm_reply if llm_reply else self._generate_deterministic_reply(query, intent, escalation, precedents)

        debug_trace = {
            "grounding_precedents": [
                {
                    "id": p.precedent_id,
                    "resolution_score": p.resolution_score,
                    "semantic_similarity": p.semantic_similarity,
                    "reranked_score": p.reranked_score
                }
                for p in precedents
            ],
            "generation_mode": mode,
            "escalation_reason": escalation.stated_reason
        }

        return final_reply, mode, debug_trace
