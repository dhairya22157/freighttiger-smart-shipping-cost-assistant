"""
Evaluation and trustworthiness test harness for FreightTiger Assistant.
Evaluates synthetic adversarial cases:
- Route mismatch in notes
- Date window mismatch
- Unaffected / negative notes
- Missing supporting evidence
- Valid justification validation
"""

import pandas as pd
import pytest

from src.note_retrieval import NoteRetriever
from src.explanation_generator import generate_deterministic_explanation


@pytest.fixture
def retriever():
    # Load default notes
    from src.data_loader import load_context_notes
    from src.config import default_config
    return NoteRetriever(load_context_notes(default_config.CONTEXT_NOTES_FILE, default_config.REQUIRED_NOTE_COLUMNS))


def test_adversarial_route_mismatch(retriever):
    """Note applies to Chennai-Bangalore, tested against Mumbai-Delhi."""
    ev = retriever.retrieve_evidence("Mumbai-Delhi", "2025-02-24")
    res = generate_deterministic_explanation(
        route="Mumbai-Delhi",
        week_of="2025-02-24",
        cost_per_tonne_km=3.5,
        diff_hist_pct=30.0,
        diff_peer_pct=25.0,
        is_candidate_anomaly=True,
        evidence=ev,
    )
    assert res.flagged == "Yes"
    assert res.matched_note_id == ""
    assert "N001" not in res.matched_note_id


def test_adversarial_date_mismatch(retriever):
    """Note applies to Ahmedabad-Mumbai on 2025-01-20, tested against 2024-06-10."""
    ev = retriever.retrieve_evidence("Ahmedabad-Mumbai", "2024-06-10")
    res = generate_deterministic_explanation(
        route="Ahmedabad-Mumbai",
        week_of="2024-06-10",
        cost_per_tonne_km=3.5,
        diff_hist_pct=30.0,
        diff_peer_pct=25.0,
        is_candidate_anomaly=True,
        evidence=ev,
    )
    assert res.flagged == "Yes"
    assert res.matched_note_id == ""
    assert "N002" not in res.matched_note_id


def test_adversarial_non_affecting_notes(retriever):
    """Notes stating costs were unaffected (N005, N010) must never justify a spike."""
    for route, week, non_justifying_id in [("Mumbai-Delhi", "2024-07-29", "N005"), ("Mumbai-Pune", "2025-10-27", "N010")]:
        ev = retriever.retrieve_evidence(route, week)
        res = generate_deterministic_explanation(
            route=route,
            week_of=week,
            cost_per_tonne_km=3.0,
            diff_hist_pct=20.0,
            diff_peer_pct=22.0,
            is_candidate_anomaly=True,
            evidence=ev,
        )
        assert res.flagged == "Yes"
        assert res.matched_note_id == ""


def test_labeled_ground_truth_accuracy(retriever):
    """
    Evaluates against explicit labeled ground truth test cases.
    """
    test_cases = [
        # (route, week, is_anom, expected_flag, expected_note)
        ("Ahmedabad-Mumbai", "2025-01-20", True, "No (justified)", "N002"),
        ("Chennai-Bangalore", "2025-02-24", True, "No (justified)", "N001"),
        ("Chennai-Bangalore", "2025-03-03", True, "No (justified)", "N001"),
        ("Delhi-Jaipur", "2024-11-11", True, "Yes", ""),
        ("Mumbai-Pune", "2025-09-15", True, "Yes", ""),
        ("Mumbai-Pune", "2024-03-04", False, "No", ""),
    ]

    passed = 0
    for route, week, is_anom, exp_flag, exp_note in test_cases:
        ev = retriever.retrieve_evidence(route, week) if is_anom else None
        res = generate_deterministic_explanation(
            route=route,
            week_of=week,
            cost_per_tonne_km=3.5,
            diff_hist_pct=25.0 if is_anom else 2.0,
            diff_peer_pct=25.0 if is_anom else 1.0,
            is_candidate_anomaly=is_anom,
            evidence=ev,
        )
        assert res.flagged == exp_flag, f"Failed flag for {route} {week}: expected {exp_flag}, got {res.flagged}"
        assert res.matched_note_id == exp_note, f"Failed note for {route} {week}: expected {exp_note}, got {res.matched_note_id}"
        passed += 1

    assert passed == len(test_cases)
