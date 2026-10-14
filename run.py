"""
Execution script for FreightTiger Smart Shipping Cost Assistant.
Usage:
    python run.py
    python run.py --output-mode flagged_only
    python run.py --output-mode all
"""

import argparse
from pathlib import Path
from src.config import Config
from src.pipeline import ShippingCostAssistantPipeline


def main():
    parser = argparse.ArgumentParser(description="FreightTiger Smart Shipping Cost Assistant")
    parser.add_argument(
        "--output-mode",
        choices=["all", "flagged_only"],
        default="all",
        help="Whether to output all route-weeks ('all') or only candidate anomalies ('flagged_only'). Default: all",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path for exported CSV. Defaults to outputs/shipping_cost_analysis.csv",
    )
    parser.add_argument(
        "--hist-threshold",
        type=float,
        default=15.0,
        help="Historical cost rise anomaly threshold percentage. Default: 15.0",
    )
    parser.add_argument(
        "--peer-threshold",
        type=float,
        default=20.0,
        help="Peer route cost difference threshold percentage. Default: 20.0",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="deterministic",
        help="Explanation provider: 'deterministic' (default, free/offline) or 'groq'.",
    )

    args = parser.parse_args()

    config = Config(
        HIST_DIFF_THRESHOLD_PCT=args.hist_threshold,
        PEER_DIFF_THRESHOLD_PCT=args.peer_threshold,
        OUTPUT_MODE=args.output_mode,
        LLM_PROVIDER=args.provider,
    )

    print("=" * 70)
    print("FREIGHTTIGER SMART SHIPPING COST ASSISTANT")
    print("=" * 70)
    print(f"Shipments File:      {config.SHIPMENTS_FILE}")
    print(f"Context Notes File:  {config.CONTEXT_NOTES_FILE}")
    print(f"Historical Threshold:{config.HIST_DIFF_THRESHOLD_PCT}% (over trailing {config.HISTORICAL_ROLLING_WEEKS}w rolling avg)")
    print(f"Peer Threshold:      {config.PEER_DIFF_THRESHOLD_PCT}% (above same-week peers)")
    print(f"Output Mode:         {config.OUTPUT_MODE}")
    print(f"Provider:            {config.LLM_PROVIDER}")
    print("-" * 70)

    pipeline = ShippingCostAssistantPipeline(config=config)
    output_target = Path(args.output_path) if args.output_path else config.DEFAULT_OUTPUT_FILE
    df, summary = pipeline.run(output_path=output_target, output_mode=args.output_mode)

    print("\nEXECUTION SUMMARY:")
    print(f"- Total Route-Weeks Evaluated: {summary['total_route_weeks']}")
    print(f"- Candidate Anomalies Detected: {summary['candidate_anomalies']}")
    print(f"  * Justified (explained by note): {summary['justified_anomalies']}")
    print(f"  * Unexplained (flagged for review): {summary['unexplained_anomalies']}")
    print(f"- Normal Weeks: {summary['normal_weeks']}")
    print(f"- Rows Written to Output CSV: {summary['output_rows_written']}")
    print(f"- Primary Output File: {summary['output_path']}")
    print(f"- Full Analysis File:  {summary['all_output_path']}")
    print(f"- Flagged-Only File:   {summary['flagged_output_path']}")

    # Display sample rows from output
    print("\nSAMPLE OUTPUT ROWS (Top 5):")
    sample_cols = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "flagged", "matched_note_id"]
    print(df[sample_cols].head(5).to_string(index=False))

    token_info = summary["token_summary"]
    print("\nTOKEN & COST ACCOUNTING:")
    print(f"- Total LLM Calls:      {token_info['total_calls']}")
    print(f"- Prompt Tokens:        {token_info['total_prompt_tokens']}")
    print(f"- Completion Tokens:    {token_info['total_completion_tokens']}")
    print(f"- Total Tokens:         {token_info['total_tokens']}")
    print(f"- Estimated Cost (USD): ${token_info['total_cost_usd']:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
