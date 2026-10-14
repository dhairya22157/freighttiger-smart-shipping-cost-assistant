"""
3-Run Reproducibility Verification Script for FreightTiger Assistant.
Executes the pipeline three untouched times on the exact same input,
performs strict element-wise diffing on all required fields, and reports the results.
Usage:
    python reproducibility_check.py
"""

from pathlib import Path
import sys
import pandas as pd

from src.config import Config, default_config
from src.pipeline import ShippingCostAssistantPipeline


def run_reproducibility_check() -> bool:
    print("=" * 75)
    print("FREIGHTTIGER SMART ASSISTANT — 3-RUN REPRODUCIBILITY VERIFICATION")
    print("=" * 75)

    run_files = []
    run_dfs = []

    for i in range(1, 4):
        run_file = default_config.OUTPUT_DIR / f"reproducibility_run_{i}.csv"
        run_files.append(run_file)
        print(f"\n[RUN {i}/3] Executing pipeline...")

        pipeline = ShippingCostAssistantPipeline(config=default_config)
        df, summary = pipeline.run(output_path=run_file, output_mode="all")
        run_dfs.append(df)
        print(f" -> Completed. Saved {len(df)} rows to {run_file}")

    print("\n" + "=" * 75)
    print("STRICT DIFFERENTIAL COMPARISON ACROSS RUNS 1, 2, AND 3:")
    print("=" * 75)

    df1, df2, df3 = run_dfs[0], run_dfs[1], run_dfs[2]

    # Verify identical row counts
    assert len(df1) == len(df2) == len(df3), (
        f"Row count mismatch! Run 1: {len(df1)}, Run 2: {len(df2)}, Run 3: {len(df3)}"
    )
    print(f"[PASS] Row count matches across all runs: {len(df1)} rows.")

    # Fields requiring strict equality
    fields_to_compare = [
        ("route", "Route Identifiers"),
        ("week_of", "Week Identifiers"),
        ("cost_per_tonne_km", "Weekly Cost per Tonne-Km"),
        ("vs_own_history", "Baseline A (vs Own History)"),
        ("vs_similar_routes", "Baseline B (vs Peer Routes)"),
        ("flagged", "Anomaly Verdict Flag"),
        ("matched_note_id", "Cited Matched Note ID"),
        ("reason", "Explanation Reason Text"),
    ]

    all_identical = True

    for col, desc in fields_to_compare:
        match_1_2 = (df1[col] == df2[col]).all()
        match_1_3 = (df1[col] == df3[col]).all()

        if match_1_2 and match_1_3:
            print(f"[PASS] {desc} ('{col}'): 100% IDENTICAL across all 3 runs (0 diffs).")
        else:
            diff_1_2 = (df1[col] != df2[col]).sum()
            diff_1_3 = (df1[col] != df3[col]).sum()
            print(f"[FAIL] {desc} ('{col}') had differences: Run 1 vs 2 = {diff_1_2}, Run 1 vs 3 = {diff_1_3}")
            all_identical = False

    print("-" * 75)
    if all_identical:
        print("OVERALL REPRODUCIBILITY RESULT: 100% REPRODUCIBLE (PASS)")
        print("All numbers, flags, cited notes, and explanations match bit-for-bit across all 3 runs.")
    else:
        print("OVERALL REPRODUCIBILITY RESULT: FAILED (Differences detected)")
    print("=" * 75)

    return all_identical


if __name__ == "__main__":
    success = run_reproducibility_check()
    sys.exit(0 if success else 1)
