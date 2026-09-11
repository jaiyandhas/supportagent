"""Automated consistency tests ensuring documentation and UI never drift from eval_results.json."""

import json
import os
import re
import pytest

from backend.pipeline.engine import SupportAgentEngine


def test_eval_results_exist_and_valid():
    eval_path = "data/eval_results.json"
    assert os.path.exists(eval_path), "eval_results.json must exist"
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "systems" in data
    assert "trivial_baseline" in data["systems"]
    assert "simple_baseline" in data["systems"]
    assert "calibrated_trust_first" in data["systems"]
    assert "pairwise_comparison" in data
    assert "retrieval_impact" in data


def test_no_scrape_artifacts_in_deterministic_drafts():
    """Verify that Mode 1 deterministic generator never leaks raw scrape artifacts."""
    engine = SupportAgentEngine()
    with open("data/golden_eval_set.json", "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    # Test across all golden set queries
    for item in golden_set:
        res = engine.process_query(item["customer_query"])
        reply = res.drafted_reply

        # Twitter thread counters like (1/2), 2/3, etc.
        assert not re.search(r"\(?\b\d+/\d+\)?", reply), f"Leaked thread marker in: {reply}"
        # Agent handle signatures like ^SI, ^SK, ^amzn
        assert not re.search(r"\^[A-Za-z]{1,4}\b", reply), f"Leaked agent signature in: {reply}"
        # Trailing colon or unreplaced placeholder
        assert not reply.endswith(" :"), f"Trailing colon in: {reply}"


def test_docs_and_eval_metrics_consistency():
    """Verify that README.md and REPORT.md tables match the exact numbers in eval_results.json."""
    with open("data/eval_results.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    triv_judge = eval_data["systems"]["trivial_baseline"]["judge_rubric"]["overall_score"]
    simp_judge = eval_data["systems"]["simple_baseline"]["judge_rubric"]["overall_score"]
    trust_judge = eval_data["systems"]["calibrated_trust_first"]["judge_rubric"]["overall_score"]

    with open("README.md", "r", encoding="utf-8") as f:
        readme_content = f.read()

    with open("REPORT.md", "r", encoding="utf-8") as f:
        report_content = f.read()

    # Assert judge numbers appear accurately in README and REPORT
    assert f"{triv_judge:.2f}" in readme_content, f"README missing trivial judge score {triv_judge:.2f}"
    assert f"{simp_judge:.2f}" in readme_content, f"README missing simple judge score {simp_judge:.2f}"
    assert f"{trust_judge:.2f}" in readme_content, f"README missing trust-first judge score {trust_judge:.2f}"

    assert f"{triv_judge:.2f}" in report_content, f"REPORT missing trivial judge score {triv_judge:.2f}"
    assert f"{simp_judge:.2f}" in report_content, f"REPORT missing simple judge score {simp_judge:.2f}"
    assert f"{trust_judge:.2f}" in report_content, f"REPORT missing trust-first judge score {trust_judge:.2f}"
