"""
End-to-end pipeline for FreightTiger Smart Shipping Cost Assistant.
Integrates data loading, preprocessing, cost analysis, anomaly detection,
note retrieval, grounded explanation generation, and contract-compliant output generation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

from src.config import Config, default_config
from src.data_loader import load_shipments, load_context_notes, generate_dataset_summary
from src.preprocessing import preprocess_shipments, preprocess_notes
from src.cost_analysis import compute_all_metrics
from src.anomaly_detection import AnomalyDetectionPolicy, detect_anomalies
from src.note_retrieval import NoteRetriever
from src.explanation_generator import ExplanationGenerator
from src.output_writer import write_output_csv, CONTRACT_COLUMNS
from src.token_cost_logger import TokenCostLogger


class ShippingCostAssistantPipeline:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self.logger = TokenCostLogger(log_path=self.config.TOKEN_LOG_FILE)
        self.explanation_gen = ExplanationGenerator(
            provider=self.config.LLM_PROVIDER,
            model=self.config.LLM_MODEL,
            logger=self.logger,
        )

    def run(self, output_path: Optional[Path] = None, output_mode: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, any]]:
        mode = output_mode or self.config.OUTPUT_MODE
        target_path = output_path or self.config.DEFAULT_OUTPUT_FILE

        # 1. Load data
        shipments_raw = load_shipments(self.config.SHIPMENTS_FILE, self.config.REQUIRED_SHIPMENT_COLUMNS)
        notes_raw = load_context_notes(self.config.CONTEXT_NOTES_FILE, self.config.REQUIRED_NOTE_COLUMNS)

        # 2. Preprocess
        shipments_df = preprocess_shipments(shipments_raw)
        notes_df = preprocess_notes(notes_raw)

        # 3. Compute weekly metrics & baselines A and B
        weekly_metrics = compute_all_metrics(shipments_df, window_size=self.config.HISTORICAL_ROLLING_WEEKS)

        # 4. Anomaly detection
        policy = AnomalyDetectionPolicy(
            hist_threshold_pct=self.config.HIST_DIFF_THRESHOLD_PCT,
            peer_threshold_pct=self.config.PEER_DIFF_THRESHOLD_PCT,
            require_both=self.config.REQUIRE_BOTH_ANOMALY_CONDITIONS,
        )
        analyzed_df = detect_anomalies(weekly_metrics, policy=policy)

        # 5. Note retrieval & Grounded explanation generation
        retriever = NoteRetriever(notes_df)

        flags = []
        note_ids = []
        reasons = []

        for _, row in analyzed_df.iterrows():
            route = row["route"]
            week_of = row["week_of"]
            cost = row["cost_per_tonne_km"]
            diff_h = row["diff_hist_pct"]
            diff_p = row["diff_peer_pct"]
            is_anomaly = row["is_candidate_anomaly"]

            evidence = None
            if is_anomaly:
                evidence = retriever.retrieve_evidence(route, week_of)

            res = self.explanation_gen.generate(
                route=route,
                week_of=week_of,
                cost_per_tonne_km=cost,
                diff_hist_pct=diff_h,
                diff_peer_pct=diff_p,
                is_candidate_anomaly=is_anomaly,
                evidence=evidence,
            )

            flags.append(res.flagged)
            note_ids.append(res.matched_note_id)
            reasons.append(res.reason)

        analyzed_df["flagged"] = flags
        analyzed_df["matched_note_id"] = note_ids
        analyzed_df["reason"] = reasons

        # 6. Save token/cost logs
        self.logger.save_log(self.config.TOKEN_LOG_FILE)

        # 7. Export outputs
        # Always save complete analysis (728 rows) and flagged only
        all_output_file = self.config.OUTPUT_DIR / "shipping_cost_analysis_all.csv"
        flagged_output_file = self.config.FLAGGED_ONLY_OUTPUT_FILE

        write_output_csv(analyzed_df, all_output_file)

        flagged_df = analyzed_df[analyzed_df["flagged"] != "No"].copy()
        write_output_csv(flagged_df, flagged_output_file)

        # Write primary target output
        primary_raw = flagged_df if mode == "flagged_only" else analyzed_df
        primary_df = primary_raw[CONTRACT_COLUMNS].copy()
        primary_df["cost_per_tonne_km"] = primary_df["cost_per_tonne_km"].apply(lambda x: f"{float(x):.2f}")
        write_output_csv(primary_df, target_path)

        # Summary metadata
        summary = {
            "total_route_weeks": len(analyzed_df),
            "candidate_anomalies": int(analyzed_df["is_candidate_anomaly"].sum()),
            "justified_anomalies": int((analyzed_df["flagged"] == "No (justified)").sum()),
            "unexplained_anomalies": int((analyzed_df["flagged"] == "Yes").sum()),
            "normal_weeks": int((analyzed_df["flagged"] == "No").sum()),
            "output_rows_written": len(primary_df),
            "output_path": str(target_path),
            "all_output_path": str(all_output_file),
            "flagged_output_path": str(flagged_output_file),
            "token_summary": self.logger.get_summary(),
        }

        return primary_df, summary
