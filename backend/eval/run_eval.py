"""Comprehensive evaluation runner comparing Trivial, Simple, and Calibrated Trust-First Agents.

Produces all headline metrics, calibration curves, baseline tables, sensitivity sweeps,
and outputs data/eval_results.json for the demo UI and formal report.
"""

import json
import os
import time
from typing import Dict, List
import numpy as np

from backend.eval.baselines import SimpleBaselineAgent, TrivialBaselineAgent
from backend.eval.judge import SupportReplyJudge
from backend.eval.metrics import (
    compute_calibration_curve,
    compute_coverage_risk_curve,
    compute_escalation_metrics,
    compute_intent_metrics,
)
from backend.pipeline.engine import SupportAgentEngine


def evaluate_retrieval_impact(engine: SupportAgentEngine, gold_items: List[Dict]) -> Dict:
    """Compare retrieval relevance and resolution scores before vs after outcome reranking."""
    precedents = engine.retriever.precedents
    sample_queries = [g["customer_query"] for g in gold_items if not g.get("is_cold_case")][:60]
    target_subs = [g["true_sub_intent"] for g in gold_items if not g.get("is_cold_case")][:60]

    # Baseline: alpha = 1.0 (pure semantic similarity, NO outcome reranking)
    hits_before = 0
    res_scores_before = []
    for q, target in zip(sample_queries, target_subs):
        cands = engine.retriever.retrieve_candidates(q, target_sub_intent=target, top_k=3, custom_alpha=1.0)
        if cands and any(c.sub_intent == target for c in cands):
            hits_before += 1
        if cands:
            res_scores_before.append(cands[0].resolution_score)

    # Proposed: alpha = 0.6 (outcome-weighted reranking)
    hits_after = 0
    res_scores_after = []
    for q, target in zip(sample_queries, target_subs):
        cands = engine.retriever.retrieve_candidates(q, target_sub_intent=target, top_k=3, custom_alpha=0.6)
        if cands and any(c.sub_intent == target for c in cands):
            hits_after += 1
        if cands:
            res_scores_after.append(cands[0].resolution_score)

    # Alpha sensitivity study: sweep alpha from 0.0 to 1.0
    alpha_sweep = []
    for alpha_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        avg_res = []
        hits = 0
        for q, target in zip(sample_queries, target_subs):
            cands = engine.retriever.retrieve_candidates(q, target_sub_intent=target, top_k=3, custom_alpha=alpha_val)
            if cands and any(c.sub_intent == target for c in cands):
                hits += 1
            if cands:
                avg_res.append(cands[0].resolution_score)
        alpha_sweep.append({
            "alpha": alpha_val,
            "hit_rate": round(hits / len(sample_queries), 3),
            "mean_resolution_score": round(float(np.mean(avg_res)), 3)
        })

    return {
        "hit_rate_before_reranking_alpha_1_0": round(hits_before / len(sample_queries), 3),
        "hit_rate_after_reranking_alpha_0_6": round(hits_after / len(sample_queries), 3),
        "mean_resolution_before": round(float(np.mean(res_scores_before)), 3),
        "mean_resolution_after": round(float(np.mean(res_scores_after)), 3),
        "alpha_sensitivity_sweep": alpha_sweep
    }


def run_full_evaluation(golden_path: str = "data/golden_eval_set.json", output_path: str = "data/eval_results.json"):
    """Execute evaluation over all 180 golden test cases across Trivial, Simple, and Calibrated systems."""
    print("="*65)
    print("STARTING FULL EVALUATION HARNESS (AMAZONHELP SUPPORT AGENT)")
    print("="*65)

    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden evaluation set missing at {golden_path}. Run ingestion first.")

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    print(f"Loaded {len(golden_set)} golden evaluation cases.")

    # Initialize agents
    print("Initializing Trivial Baseline Agent...")
    trivial_agent = TrivialBaselineAgent()

    print("Initializing Simple Baseline Agent (TF-IDF + Top-1 NN)...")
    simple_agent = SimpleBaselineAgent()

    print("Initializing Calibrated Trust-First Agent...")
    calibrated_engine = SupportAgentEngine()

    judge = SupportReplyJudge()

    # Ground truth arrays
    gold_coarse = [g["true_coarse_intent"] for g in golden_set]
    gold_sub = [g["true_sub_intent"] for g in golden_set]
    gold_escalate = [g["gold_should_escalate"] for g in golden_set]

    # Evaluate each system
    systems = {
        "trivial_baseline": {"agent": trivial_agent, "results": []},
        "simple_baseline": {"agent": simple_agent, "results": []},
        "calibrated_trust_first": {"agent": calibrated_engine, "results": []}
    }

    start_eval_time = time.time()

    for sys_name, sys_dict in systems.items():
        print(f"Evaluating {sys_name} on {len(golden_set)} test items...")
        agent = sys_dict["agent"]
        res_list = []
        for item in golden_set:
            pred = agent.process_query(item["customer_query"])
            res_list.append(pred)
        sys_dict["results"] = res_list

    elapsed_eval = round(time.time() - start_eval_time, 2)
    print(f"Completed inference runs in {elapsed_eval}s.")

    # Metrics computation
    metrics_summary = {}

    for sys_name, sys_dict in systems.items():
        preds = sys_dict["results"]
        pred_coarse = [p.intent.coarse_intent for p in preds]
        pred_sub = [p.intent.sub_intent for p in preds]
        pred_escalate = [p.escalation.should_escalate for p in preds]
        pred_confs = [p.escalation.calibrated_confidence for p in preds]

        # Intent classification metrics
        coarse_metrics = compute_intent_metrics(gold_coarse, pred_coarse)
        sub_metrics = compute_intent_metrics(gold_sub, pred_sub)

        # Escalation routing metrics
        esc_metrics = compute_escalation_metrics(gold_escalate, pred_escalate, pred_confs)

        # Correct auto-handle ground truth (1 if safe auto-handle, 0 if it should have escalated)
        y_correct_autohandle = [0 if gold_esc else 1 for gold_esc in gold_escalate]
        # P(auto-handle) = confidence
        p_autohandle = pred_confs

        ece, brier, reliability_points = compute_calibration_curve(y_correct_autohandle, p_autohandle)
        cov_risk_points = compute_coverage_risk_curve(gold_escalate, pred_confs)

        # Judge scores on replies
        judge_scores = []
        for g_item, p_item in zip(golden_set, preds):
            j_eval = judge.evaluate_reply(
                g_item["customer_query"],
                p_item.drafted_reply,
                [cand.model_dump() for cand in p_item.precedents],
                g_item["reference_reply_guidelines"],
                p_item.escalation.should_escalate,
                gold_should_escalate=g_item["gold_should_escalate"]
            )
            judge_scores.append(j_eval)

        mean_faithfulness = float(np.mean([j["faithfulness"] for j in judge_scores]))
        mean_tone = float(np.mean([j["tone"] for j in judge_scores]))
        mean_res_likelihood = float(np.mean([j["resolution_likelihood"] for j in judge_scores]))
        mean_policy = float(np.mean([j["policy_adherence"] for j in judge_scores]))
        mean_overall_judge = float(np.mean([j["overall_score"] for j in judge_scores]))

        metrics_summary[sys_name] = {
            "intent_coarse": coarse_metrics,
            "intent_sub": sub_metrics,
            "escalation": esc_metrics,
            "calibration": {
                "ece": ece,
                "brier_score": brier,
                "reliability_curve": reliability_points
            },
            "coverage_risk_curve": cov_risk_points,
            "judge_rubric": {
                "faithfulness": round(mean_faithfulness, 2),
                "tone": round(mean_tone, 2),
                "resolution_likelihood": round(mean_res_likelihood, 2),
                "policy_adherence": round(mean_policy, 2),
                "overall_score": round(mean_overall_judge, 2)
            }
        }

    # Pairwise win-rate comparison: Trust-First vs Trivial and Trust-First vs Simple
    trust_preds = systems["calibrated_trust_first"]["results"]
    triv_preds = systems["trivial_baseline"]["results"]
    simp_preds = systems["simple_baseline"]["results"]

    pairwise_vs_trivial = {"win": 0, "tie": 0, "loss": 0}
    pairwise_vs_simple = {"win": 0, "tie": 0, "loss": 0}

    for g_item, t_p, tr_p, s_p in zip(golden_set, triv_preds, trust_preds, simp_preds):
        q = g_item["customer_query"]
        gold_esc = g_item["gold_should_escalate"]
        score_triv = judge.evaluate_reply(q, t_p.drafted_reply, [], "", t_p.escalation.should_escalate, gold_esc)["overall_score"]
        score_simp = judge.evaluate_reply(q, s_p.drafted_reply, [c.model_dump() for c in s_p.precedents], "", s_p.escalation.should_escalate, gold_esc)["overall_score"]
        score_trust = judge.evaluate_reply(q, tr_p.drafted_reply, [c.model_dump() for c in tr_p.precedents], "", tr_p.escalation.should_escalate, gold_esc)["overall_score"]

        # vs Trivial
        if score_trust > score_triv + 0.25:
            pairwise_vs_trivial["win"] += 1
        elif score_triv > score_trust + 0.25:
            pairwise_vs_trivial["loss"] += 1
        else:
            pairwise_vs_trivial["tie"] += 1

        # vs Simple
        if score_trust > score_simp + 0.25:
            pairwise_vs_simple["win"] += 1
        elif score_simp > score_trust + 0.25:
            pairwise_vs_simple["loss"] += 1
        else:
            pairwise_vs_simple["tie"] += 1

    total_pairs = len(golden_set)
    pairwise_results = {
        "vs_trivial": {
            "win_rate": round(pairwise_vs_trivial["win"] / total_pairs, 3),
            "tie_rate": round(pairwise_vs_trivial["tie"] / total_pairs, 3),
            "loss_rate": round(pairwise_vs_trivial["loss"] / total_pairs, 3),
            "counts": pairwise_vs_trivial
        },
        "vs_simple": {
            "win_rate": round(pairwise_vs_simple["win"] / total_pairs, 3),
            "tie_rate": round(pairwise_vs_simple["tie"] / total_pairs, 3),
            "loss_rate": round(pairwise_vs_simple["loss"] / total_pairs, 3),
            "counts": pairwise_vs_simple
        }
    }

    # Retrieval impact study
    print("Evaluating retrieval hit-rate impact (before vs. after outcome reranking)...")
    retrieval_impact = evaluate_retrieval_impact(calibrated_engine, golden_set)

    # Human-Judge Agreement study
    print("Evaluating empirical agreement between judge scores and human ratings (N=45)...")
    judge_agreement = judge.evaluate_judge_human_agreement("data/judge_human_eval_dataset.json")

    final_report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "golden_set_size": len(golden_set),
        "cold_case_count": sum(1 for g in golden_set if g.get("is_cold_case")),
        "borderline_case_count": sum(1 for g in golden_set if g.get("is_borderline")),
        "systems": metrics_summary,
        "pairwise_comparison": pairwise_results,
        "retrieval_impact": retrieval_impact,
        "judge_human_agreement": judge_agreement
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report_data, f, indent=2)
    print(f"\nSaved complete evaluation report to {output_path}")

    # Print comparative results table
    print("\n" + "="*80)
    print("HEADLINE COMPARATIVE EVALUATION RESULTS")
    print("="*80)
    print(f"{'Metric':35s} | {'Trivial Baseline':16s} | {'Simple Baseline':16s} | {'Trust-First (Ours)':16s}")
    print("-" * 88)

    t_m = metrics_summary["trivial_baseline"]
    s_m = metrics_summary["simple_baseline"]
    c_m = metrics_summary["calibrated_trust_first"]

    rows = [
        ("Coarse Intent Accuracy", f"{t_m['intent_coarse']['accuracy']:.3f}", f"{s_m['intent_coarse']['accuracy']:.3f}", f"{c_m['intent_coarse']['accuracy']:.3f}"),
        ("Coarse Worst-Class F1", f"{t_m['intent_coarse']['worst_class_f1']:.3f} ({t_m['intent_coarse']['worst_class']})", f"{s_m['intent_coarse']['worst_class_f1']:.3f} ({s_m['intent_coarse']['worst_class']})", f"{c_m['intent_coarse']['worst_class_f1']:.3f} ({c_m['intent_coarse']['worst_class']})"),
        ("Sub-Intent Macro F1", f"{t_m['intent_sub']['macro_f1']:.3f}", f"{s_m['intent_sub']['macro_f1']:.3f}", f"{c_m['intent_sub']['macro_f1']:.3f}"),
        ("Sub-Intent Worst-Class F1", f"{t_m['intent_sub']['worst_class_f1']:.3f}", f"{s_m['intent_sub']['worst_class_f1']:.3f}", f"{c_m['intent_sub']['worst_class_f1']:.3f}"),
        ("Escalation Precision", f"{t_m['escalation']['precision']:.3f}", f"{s_m['escalation']['precision']:.3f}", f"{c_m['escalation']['precision']:.3f}"),
        ("Escalation Recall", f"{t_m['escalation']['recall']:.3f}", f"{s_m['escalation']['recall']:.3f}", f"{c_m['escalation']['recall']:.3f}"),
        ("Escalation F1", f"{t_m['escalation']['f1']:.3f}", f"{s_m['escalation']['f1']:.3f}", f"{c_m['escalation']['f1']:.3f}"),
        ("Escalation AUROC", f"{t_m['escalation']['auroc']:.3f}", f"{s_m['escalation']['auroc']:.3f}", f"{c_m['escalation']['auroc']:.3f}"),
        ("Expected Calibration Error (ECE) ↓", f"{t_m['calibration']['ece']:.3f}", f"{s_m['calibration']['ece']:.3f}", f"{c_m['calibration']['ece']:.3f}"),
        ("Brier Score Loss ↓", f"{t_m['calibration']['brier_score']:.3f}", f"{s_m['calibration']['brier_score']:.3f}", f"{c_m['calibration']['brier_score']:.3f}"),
        ("Judge Overall Quality (1-5) ↑", f"{t_m['judge_rubric']['overall_score']:.2f}", f"{s_m['judge_rubric']['overall_score']:.2f}", f"{c_m['judge_rubric']['overall_score']:.2f}"),
        ("Judge Faithfulness (1-5) ↑", f"{t_m['judge_rubric']['faithfulness']:.2f}", f"{s_m['judge_rubric']['faithfulness']:.2f}", f"{c_m['judge_rubric']['faithfulness']:.2f}"),
    ]

    for label, val_t, val_s, val_c in rows:
        print(f"{label:35s} | {val_t:16s} | {val_s:16s} | {val_c:16s}")
    print("="*80)

    print("\nRETRIEVAL OUTCOME RERANKING HIT-RATE IMPACT:")
    print(f"  • Top-3 Hit-Rate Before Reranking (alpha=1.0): {retrieval_impact['hit_rate_before_reranking_alpha_1_0']:.1%}")
    print(f"  • Top-3 Hit-Rate After Reranking (alpha=0.6):  {retrieval_impact['hit_rate_after_reranking_alpha_0_6']:.1%}")
    print(f"  • Mean Precedent Resolution Score: Before={retrieval_impact['mean_resolution_before']:.2f} -> After={retrieval_impact['mean_resolution_after']:.2f}")
    print(f"  • Judge-Human Agreement: Pearson r={judge_agreement['pearson_r']:.3f}, QWK={judge_agreement['quadratic_weighted_kappa']:.3f}, MAE={judge_agreement['mean_absolute_error']:.3f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_full_evaluation()
