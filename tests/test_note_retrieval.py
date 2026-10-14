"""
Unit tests for context-note retrieval, date filtering, and causal guardrails.
"""

import pandas as pd
import pytest

from src.data_loader import load_context_notes
from src.note_retrieval import NoteRetriever
from src.config import default_config


@pytest.fixture
def retriever():
    notes_df = load_context_notes(default_config.CONTEXT_NOTES_FILE, default_config.REQUIRED_NOTE_COLUMNS)
    return NoteRetriever(notes_df)


def test_n002_justifies_ahmedabad_mumbai_festival(retriever):
    evidence = retriever.retrieve_evidence("Ahmedabad-Mumbai", "2025-01-20")
    assert evidence.is_justified is True
    assert evidence.matched_note_id == "N002"
    assert "festival" in evidence.matched_note_text.lower()


def test_n001_justifies_chennai_bangalore_multiweek_flooding(retriever):
    # Week 1 of flooding: 2025-02-24
    ev1 = retriever.retrieve_evidence("Chennai-Bangalore", "2025-02-24")
    assert ev1.is_justified is True
    assert ev1.matched_note_id == "N001"

    # Week 2 of flooding: 2025-03-03 (within Feb 24 - Mar 8 window)
    ev2 = retriever.retrieve_evidence("Chennai-Bangalore", "2025-03-03")
    assert ev2.is_justified is True
    assert ev2.matched_note_id == "N001"

    # After flood resolved: 2025-03-17 (N009 confirms returned to normal)
    ev_post = retriever.retrieve_evidence("Chennai-Bangalore", "2025-03-17")
    assert ev_post.is_justified is False
    assert ev_post.matched_note_id is None


def test_route_mismatch_never_justifies(retriever):
    # Chennai-Bangalore flooding must never justify Mumbai-Delhi or Delhi-Jaipur
    ev = retriever.retrieve_evidence("Mumbai-Delhi", "2025-02-24")
    assert ev.is_justified is False
    assert ev.matched_note_id != "N001"


def test_unaffected_notes_never_justify(retriever):
    # N005: Mumbai-Delhi maintenance did not significantly affect costs
    ev_n005 = retriever.retrieve_evidence("Mumbai-Delhi", "2024-07-29")
    assert ev_n005.is_justified is False
    assert ev_n005.matched_note_id is None

    # N010: Tracking mandate costs absorbed by transporters
    ev_n010 = retriever.retrieve_evidence("Mumbai-Pune", "2025-10-27")
    assert ev_n010.is_justified is False
    assert ev_n010.matched_note_id is None


def test_sample_output_retrieval_scenarios(retriever):
    # Sample Row 2: Delhi-Jaipur 2024-11-11 -> No note found
    ev2 = retriever.retrieve_evidence("Delhi-Jaipur", "2024-11-11")
    assert ev2.is_justified is False
    assert ev2.matched_note_id is None

    # Sample Row 4: Mumbai-Pune 2025-09-15 -> Closest note is N006, but does not justify
    ev4 = retriever.retrieve_evidence("Mumbai-Pune", "2025-09-15")
    assert ev4.is_justified is False
    assert ev4.matched_note_id is None
    assert ev4.closest_note_id == "N006"
