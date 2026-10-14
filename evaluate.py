"""
Evaluation harness and trustworthiness evaluation runner for FreightTiger Assistant.
Usage:
    python evaluate.py
"""

from typing import List, Dict
import pandas as pd

from src.data_loader import load_context_notes
from src.note_retrieval import NoteRetriever
from src.explanation_generator import generate_deterministic_explanation
from src.config import default_config


def run_evaluation() -> Dict[str, any]:
    notes_df = load_context_notes(default_config.CONTEXT_NOTES_FILE, default_config.REQUIRED_NOTE_COLUMNS)
    retriever = NoteRetriever(notes_df)

    eval_dataset = [
        {
            "id": "CASE-01",
            "description": "Legitimate regional festival surcharge on Ahmedabad-Mumbai corridor",
            "route": "Ahmedabad-Mumbai",
            "week_of": "2025-01-20",
            "is_anomaly": True,
            "diff_hist_pct": 29.5,
            "diff_peer_pct": 22.5,
            "expected_flag": "No (justified)",
            "expected_note_id": "N002",
        },
        {
            "id": "CASE-02",
            "description": "Heavy highway flooding on Chennai-Bangalore (Week 1 of disruption)",
            "route": "Chennai-Bangalore",
            "week_of": "2025-02-24",
            "is_anomaly": True,
            "diff_hist_pct": 31.5,
            "diff_peer_pct": 38.7,
            "expected_flag": "No (justified)",
            "expected_note_id": "N001",
        },
        {
            "id": "CASE-03",
            "description": "Heavy highway flooding on Chennai-Bangalore (Week 2 of disruption)",
            "route": "Chennai-Bangalore",
            "week_of": "2025-03-03",
            "is_anomaly": True,
            "diff_hist_pct": 22.8,
            "diff_peer_pct": 32.0,
            "expected_flag": "No (justified)",
            "expected_note_id": "N001",
        },
        {
            "id": "CASE-04",
            "description": "Unexplained cost surge on Delhi-Jaipur with no relevant notes",
            "route": "Delhi-Jaipur",
            "week_of": "2024-11-11",
            "is_anomaly": True,
            "diff_hist_pct": 35.5,
            "diff_peer_pct": 21.0,
            "expected_flag": "Yes",
            "expected_note_id": "",
        },
        {
            "id": "CASE-05",
            "description": "Unexplained surge on Mumbai-Pune where closest note (N006) states stable demand",
            "route": "Mumbai-Pune",
            "week_of": "2025-09-15",
            "is_anomaly": True,
            "diff_hist_pct": 9.2,
            "diff_peer_pct": 23.6,
            "expected_flag": "Yes",
            "expected_note_id": "",
        },
        {
            "id": "CASE-06",
            "description": "Adversarial: Note N005 mentions maintenance on Mumbai-Delhi, but notes costs were unaffected",
            "route": "Mumbai-Delhi",
            "week_of": "2024-07-29",
            "is_anomaly": True,
            "diff_hist_pct": 18.0,
            "diff_peer_pct": 12.0,
            "expected_flag": "Yes",
            "expected_note_id": "",
        },
        {
            "id": "CASE-07",
            "description": "Adversarial: Route mismatch - Chennai-Bangalore note N001 applied to Mumbai-Pune corridor",
            "route": "Mumbai-Pune",
            "week_of": "2025-02-24",
            "is_anomaly": True,
            "diff_hist_pct": 25.0,
            "diff_peer_pct": 22.0,
            "expected_flag": "Yes",
            "expected_note_id": "",
        },
        {
            "id": "CASE-08",
            "description": "Adversarial: Date mismatch - Note N002 (Jan 2025) applied to Jan 2024",
            "route": "Ahmedabad-Mumbai",
            "week_of": "2024-01-22",
            "is_anomaly": True,
            "diff_hist_pct": 20.0,
            "diff_peer_pct": 18.0,
            "expected_flag": "Yes",
            "expected_note_id": "",
        },
        {
            "id": "CASE-09",
            "description": "Adversarial: Fleet tracking mandate N010 where costs absorbed without rate change",
            "route": "Delhi-Chennai",
            "week_of": "2025-10-27",
            "is_anomaly": True,
            "diff_hist_pct": 16.0,
            "diff_peer_pct": 15.0,
            "expected_flag": "Yes",
            "expected_note_id": "",
        },
        {
            "id": "CASE-10",
            "description": "Normal route-week within historical and peer baseline range",
            "route": "Kolkata-Bhubaneswar",
            "week_of": "2024-05-13",
            "is_anomaly": False,
            "diff_hist_pct": 1.5,
            "diff_peer_pct": -0.8,
            "expected_flag": "No",
            "expected_note_id": "",
        },
    ]

    total = len(eval_dataset)
    passed = 0
    hallucination_violations = 0
    results = []

    for case in eval_dataset:
        ev = retriever.retrieve_evidence(case["route"], case["week_of"]) if case["is_anomaly"] else None
        res = generate_deterministic_explanation(
            route=case["route"],
            week_of=case["week_of"],
            cost_per_tonne_km=3.5,
            diff_hist_pct=case["diff_hist_pct"],
            diff_peer_pct=case["diff_peer_pct"],
            is_candidate_anomaly=case["is_anomaly"],
            evidence=ev,
        )

        flag_correct = (res.flagged == case["expected_flag"])
        note_correct = (res.matched_note_id == case["expected_note_id"])
        is_pass = flag_correct and note_correct

        if not note_correct and res.matched_note_id != "":
            hallucination_violations += 1

        if is_pass:
            passed += 1

        results.append({
            "id": case["id"],
            "desc": case["description"],
            "expected": f"flag={case['expected_flag']}, note={case['expected_note_id'] or 'None'}",
            "actual": f"flag={res.flagged}, note={res.matched_note_id or 'None'}",
            "passed": is_pass,
        })

    return {
        "total_test_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "accuracy_pct": (passed / total) * 100.0,
        "hallucination_violations": hallucination_violations,
        "results": results,
    }


def main():
    print("=" * 75)
    print("FREIGHTTIGER SMART ASSISTANT — EVALUATION & TRUST HARNESS")
    print("=" * 75)

    eval_res = run_evaluation()
    print(f"Total Test Cases:          {eval_res['total_test_cases']}")
    print(f"Passed:                    {eval_res['passed_cases']}")
    print(f"Failed:                    {eval_res['failed_cases']}")
    print(f"Accuracy:                  {eval_res['accuracy_pct']:.1f}%")
    print(f"Hallucination Violations:  {eval_res['hallucination_violations']} (Must be 0)")
    print("-" * 75)
    print("TEST CASE BREAKDOWN:")
    for r in eval_res["results"]:
        status = "PASS [OK]" if r["passed"] else "FAIL [X]"
        print(f"[{status}] {r['id']}: {r['desc']}")
        print(f"       Expected: {r['expected']}")
        print(f"       Actual:   {r['actual']}")
    print("=" * 75)


if __name__ == "__main__":
    main()
