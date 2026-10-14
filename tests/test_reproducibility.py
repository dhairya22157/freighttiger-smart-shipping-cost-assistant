"""
Unit test for 3-run reproducibility.
"""

from reproducibility_check import run_reproducibility_check


def test_three_untouched_runs_are_identical():
    is_reproducible = run_reproducibility_check()
    assert is_reproducible is True
