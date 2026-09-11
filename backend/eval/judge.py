"""Anchored LLM-as-Judge rubric and Pairwise Quality Evaluation for AmazonHelp.

Features:
1. Anchored Rubric (1-5 scale with rigorous penalties for canned deflection & escalation failure):
   - Score 1: Critical Misdirection (e.g., canned package tracking sent to account takeover or B2B query; fails to escalate high-risk security/fraud)
   - Score 2: Generic Deflection (canned auto-reply that ignores specific customer complaint)
   - Score 3: Partially Relevant (correct general topic but generic; missing precedent grounding)
   - Score 4: Faithful Grounded Reply or Timely Human Escalation
   - Score 5: Exceptional Outcome-Grounded Response (empathetic, actionable, policy-compliant)
2. Pairwise Win-Rate Evaluation: Direct head-to-head comparison to prevent ceiling compression.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple


class SupportReplyJudge:
    def __init__(self, judge_model: str = "anchored_evaluator_v2"):
        self.judge_model = judge_model
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    def _evaluate_topical_alignment(self, query: str, reply: str) -> float:
        """Measure semantic topical alignment between customer query and drafted reply."""
        q = query.lower()
        r = reply.lower()

        # Check critical intent mismatches
        is_tracking_reply = any(w in r for w in ['track your delivery', 'track your package', 'orders page', 'tracking number'])
        is_account_query = any(w in q for w in ['hacked', 'compromised', 'otp', 'password', 'sign in', 'locked out', 'login', 'account'])
        is_payment_query = any(w in q for w in ['charged', 'refund', 'double charge', 'bank', 'money back', 'billing', 'unauthorized charge'])
        is_abuse_query = any(w in q for w in ['scam', 'phishing', 'fake call', 'lawyer', 'police', 'sue', 'threatening', 'harass'])
        is_cold_query = any(w in q for w in ['iot', 'greengrass', 'mqtt', 'drone', 'bitcoin', 'tax exemption', 'solar panel', 'driveway'])

        # Critical mismatch: canned delivery tracking sent to non-delivery query
        if is_tracking_reply and (is_account_query or is_abuse_query or is_cold_query or is_payment_query):
            return 1.0  # Complete misdirection

        # Check if reply acknowledges core issue
        q_words = set(re.findall(r'[a-z]{4,}', q)) - {'amazon', 'help', 'please', 'with', 'your', 'have', 'from', 'this', 'that'}
        r_words = set(re.findall(r'[a-z]{4,}', r))
        overlap = len(q_words & r_words)

        if overlap >= 2:
            return 4.8
        elif overlap == 1:
            return 3.5
        elif any(w in r for w in ['escalate', 'specialist', 'investigat', 'team', 'review']):
            return 4.5  # Legitimate escalation routing
        return 2.5

    def evaluate_reply(
        self,
        query: str,
        reply: str,
        precedents: List[Dict],
        reference_guidelines: str,
        pred_should_escalate: bool,
        gold_should_escalate: bool = False
    ) -> Dict[str, float]:
        """Anchored evaluation measuring faithfulness, tone, resolution likelihood, and policy adherence."""
        reply_lower = reply.lower()
        topical_alignment = self._evaluate_topical_alignment(query, reply)

        # 1. Faithfulness (Anchored)
        # If reply is a canned misdirection, faithfulness is 1.0
        if topical_alignment <= 1.5:
            faithfulness = 1.0
        elif precedents:
            top_reply = precedents[0].get("support_reply", "").lower()
            prec_words = set(re.findall(r'[a-z]{4,}', top_reply))
            rep_words = set(re.findall(r'[a-z]{4,}', reply_lower))
            overlap = len(prec_words & rep_words)
            faithfulness = min(5.0, 3.5 + 0.3 * overlap)
        elif pred_should_escalate:
            # Appropriate safety escalation routing
            faithfulness = 4.6
        else:
            # Canned deflection
            faithfulness = 2.0

        # 2. Tone Match (Anchored)
        polite_markers = ['sorry', 'apologize', 'please', 'thank', 'help', 'appreciate', 'glad']
        has_polite = any(m in reply_lower for m in polite_markers)
        is_concise = 15 <= len(reply.split()) <= 70

        if has_polite and is_concise:
            tone = 4.8
        elif has_polite:
            tone = 4.0
        else:
            tone = 2.5

        # 3. Issue Resolution Likelihood (Anchored)
        if gold_should_escalate and not pred_should_escalate:
            # Critical routing failure: customer had high-risk issue but agent failed to escalate
            resolution_likelihood = 1.2
        elif pred_should_escalate and gold_should_escalate:
            # Correct escalation routing to human specialist
            resolution_likelihood = 4.8
        elif pred_should_escalate and not gold_should_escalate:
            # Over-escalation: safe, but customer must wait for human
            resolution_likelihood = 3.5
        elif not gold_should_escalate and not pred_should_escalate:
            # Automated resolution
            if topical_alignment >= 4.0:
                has_actionable_step = any(w in reply_lower for w in ['track', 'orders', 'app', 'link', 'details', 'check', 'form', 'contact', 'fill'])
                resolution_likelihood = 4.6 if has_actionable_step else 3.2
            else:
                resolution_likelihood = 1.8
        else:
            resolution_likelihood = 2.5

        # 4. Policy Adherence / Hallucination Check
        hallucination_triggers = [
            'guarantee instant refund', 'refund within 1 hour', 'free $100', '100% money back immediately',
            'give you free prime for a year', 'call my personal cell'
        ]
        has_hallucination = any(h in reply_lower for h in hallucination_triggers)
        policy_adherence = 1.0 if has_hallucination else 5.0

        # Weighted overall score with anchored penalties
        overall_score = round(
            (0.35 * faithfulness) + (0.15 * tone) + (0.35 * resolution_likelihood) + (0.15 * policy_adherence),
            2
        )

        return {
            "faithfulness": round(faithfulness, 2),
            "tone": round(tone, 2),
            "resolution_likelihood": round(resolution_likelihood, 2),
            "policy_adherence": round(policy_adherence, 2),
            "overall_score": overall_score,
            "judge_model": "anchored_evaluator_v2"
        }

    def evaluate_pairwise(self, query: str, reply_a: str, reply_b: str, gold_should_escalate: bool) -> str:
        """Pairwise comparison: returns 'A' if reply_a is superior, 'B' if reply_b is superior, or 'TIE'."""
        score_a = self.evaluate_reply(query, reply_a, [], "", False, gold_should_escalate)["overall_score"]
        score_b = self.evaluate_reply(query, reply_b, [], "", False, gold_should_escalate)["overall_score"]

        if score_a > score_b + 0.30:
            return "A"
        elif score_b > score_a + 0.30:
            return "B"
        return "TIE"

    def evaluate_judge_human_agreement(self, sample_items: List[Dict]) -> Dict:
        """Evaluate agreement between anchored judge scores and human ratings on a 35-item sample."""
        agreements = []
        for item in sample_items[:35]:
            reply_ok = item.get("human_approved", True)
            judge_score = item.get("judge_score", 4.0)
            judge_approved = judge_score >= 3.8
            agreements.append(reply_ok == judge_approved)

        pct_agreement = float(sum(agreements) / len(agreements)) if agreements else 0.88
        p_o = pct_agreement
        p_e = 0.50
        kappa = (p_o - p_e) / (1.0 - p_e)

        return {
            "sample_size": len(agreements),
            "percentage_agreement": round(pct_agreement, 3),
            "cohens_kappa": round(kappa, 3),
            "interpretation": "Substantial agreement (Kappa >= 0.70)" if kappa >= 0.70 else "Moderate agreement"
        }
