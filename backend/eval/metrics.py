"""Comprehensive evaluation metrics for AmazonHelp Trust-First Support Agent.

Includes:
- Hierarchical intent accuracy and F1 (including WORST-CLASS F1)
- Escalation Precision, Recall, F1, and AUROC
- Calibration metrics: Expected Calibration Error (ECE) and Brier Score
- Reliability diagram points (binned confidence vs actual empirical accuracy)
- Coverage vs. Risk curve (sweeping calibrated thresholds)
- Retrieval hit-rate and outcome alignment comparison before/after reranking
- Alpha sensitivity study
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)


def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """Compute overall accuracy, macro F1, and identify the worst-performing class."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class_f1 = {}
    for cls_name, metrics in report.items():
        if isinstance(metrics, dict) and "f1-score" in metrics and cls_name not in ["macro avg", "weighted avg"]:
            per_class_f1[cls_name] = round(float(metrics["f1-score"]), 3)

    # Worst class
    worst_class = None
    worst_f1 = 1.0
    worst_support = 0
    for cls_name, score in per_class_f1.items():
        if score < worst_f1:
            worst_f1 = score
            worst_class = cls_name
            worst_support = int(report[cls_name].get("support", 0))

    return {
        "accuracy": round(acc, 3),
        "macro_f1": round(macro_f1, 3),
        "per_class_f1": per_class_f1,
        "worst_class": worst_class,
        "worst_class_f1": round(worst_f1, 3),
        "worst_class_support": worst_support
    }


def compute_calibration_curve(
    y_true_correct: List[int],
    y_probs: List[float],
    n_bins: int = 10
) -> Tuple[float, float, List[Dict]]:
    """Compute Expected Calibration Error (ECE), Brier score, and reliability diagram bins.
    
    y_true_correct: 1 if auto-handle was correct (gold_should_escalate == False), 0 if it was an error.
    y_probs: calibrated confidence P(auto-handle).
    """
    probs = np.array(y_probs)
    trues = np.array(y_true_correct)

    brier = float(brier_score_loss(trues, probs))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bin_indices = np.digitize(probs, bins) - 1

    total_samples = len(probs)
    ece = 0.0
    reliability_points = []

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(np.sum(mask))
        if count > 0:
            bin_conf = float(np.mean(probs[mask]))
            bin_acc = float(np.mean(trues[mask]))
            weight = count / total_samples
            ece += weight * abs(bin_acc - bin_conf)
            reliability_points.append({
                "bin_index": i,
                "bin_center": round(float(bin_centers[i]), 2),
                "mean_confidence": round(bin_conf, 3),
                "empirical_accuracy": round(bin_acc, 3),
                "sample_count": count
            })
        else:
            reliability_points.append({
                "bin_index": i,
                "bin_center": round(float(bin_centers[i]), 2),
                "mean_confidence": round(float(bin_centers[i]), 2),
                "empirical_accuracy": 0.0,
                "sample_count": 0
            })

    return round(float(ece), 3), round(brier, 3), reliability_points


def compute_coverage_risk_curve(
    gold_should_escalate: List[bool],
    predicted_confidences: List[float],
    threshold_steps: int = 20
) -> List[Dict]:
    """Compute Coverage vs. Risk curve across confidence thresholds."""
    trues_escalate = np.array(gold_should_escalate)
    confs = np.array(predicted_confidences)
    total = len(gold_should_escalate)

    thresholds = np.linspace(0.05, 0.95, threshold_steps)
    points = []

    for tau in thresholds:
        # Auto-handled if confidence >= tau
        auto_handled_mask = confs >= tau
        num_auto = int(np.sum(auto_handled_mask))
        coverage = num_auto / total

        if num_auto > 0:
            # Risk: proportion of auto-handled queries that should have escalated (critical error)
            critical_errors = int(np.sum(trues_escalate[auto_handled_mask]))
            risk = critical_errors / num_auto
        else:
            risk = 0.0

        points.append({
            "threshold": round(float(tau), 2),
            "coverage": round(float(coverage), 3),
            "risk": round(float(risk), 3),
            "auto_handled_count": num_auto
        })

    return points


def compute_escalation_metrics(gold_should_escalate: List[bool], pred_should_escalate: List[bool], confidences: List[float]) -> Dict:
    """Compute classification metrics for escalation gate."""
    y_true = [1 if e else 0 for e in gold_should_escalate]
    y_pred = [1 if e else 0 for e in pred_should_escalate]

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        # Higher escalation probability = 1 - confidence
        esc_probs = [1.0 - c for c in confidences]
        auroc = float(roc_auc_score(y_true, esc_probs))
    except Exception:
        auroc = 0.50

    return {
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "auroc": round(auroc, 3)
    }
