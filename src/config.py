"""
Configuration settings for FreightTiger Smart Shipping Cost Assistant.
Loads environment variables and sets deterministic defaults.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List


@dataclass
class Config:
    # Project paths
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    OUTPUT_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs")
    LOG_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "logs")

    # File paths
    SHIPMENTS_FILE: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "shipment_records.csv"
    )
    CONTEXT_NOTES_FILE: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "context_notes.csv"
    )
    SAMPLE_OUTPUT_FILE: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "sample_output_format_v2.csv"
    )
    DEFAULT_OUTPUT_FILE: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs" / "shipping_cost_analysis.csv"
    )
    FLAGGED_ONLY_OUTPUT_FILE: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs" / "flagged_anomalies_only.csv"
    )
    TOKEN_LOG_FILE: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "logs" / "token_cost_log.json"
    )

    # Historical Baseline A parameters
    HISTORICAL_ROLLING_WEEKS: int = 8

    # Anomaly Detection Policy Thresholds
    # PDF: 'Flag any route where the cost is rising and looks out of the ordinary compared to its own history or to similar routes.'
    HIST_DIFF_THRESHOLD_PCT: float = 15.0  # Cost rose >= 15% above own 8-week history
    PEER_DIFF_THRESHOLD_PCT: float = 20.0  # Cost rose >= 20% above peers this week
    REQUIRE_BOTH_ANOMALY_CONDITIONS: bool = False  # PDF specifies 'or', but configurable

    # Output Mode: 'all' (all 728 route-weeks) or 'flagged_only' (only anomalous/investigated weeks)
    OUTPUT_MODE: str = os.getenv("OUTPUT_MODE", "all")

    # Expected column schemas
    REQUIRED_SHIPMENT_COLUMNS: List[str] = field(
        default_factory=lambda: [
            "shipment_id",
            "origin",
            "destination",
            "route_type",
            "material",
            "quantity_tonnes",
            "distance_km",
            "freight_cost_inr",
            "shipment_date",
            "transporter",
        ]
    )

    REQUIRED_NOTE_COLUMNS: List[str] = field(
        default_factory=lambda: ["note_id", "date", "applies_to", "note"]
    )

    OUTPUT_COLUMNS: List[str] = field(
        default_factory=lambda: [
            "route",
            "week_of",
            "cost_per_tonne_km",
            "vs_own_history",
            "vs_similar_routes",
            "flagged",
            "matched_note_id",
            "reason",
        ]
    )

    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "deterministic")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_TEMPERATURE: float = 0.0

    def __post_init__(self):
        # Fallback to 'AI Intern Case Study' directory if 'data' folder files are missing
        alt_data_dir = self.BASE_DIR / "AI Intern Case Study"
        if not self.SHIPMENTS_FILE.exists() and (alt_data_dir / "shipment_records.csv").exists():
            self.SHIPMENTS_FILE = alt_data_dir / "shipment_records.csv"
        if not self.CONTEXT_NOTES_FILE.exists() and (alt_data_dir / "context_notes.csv").exists():
            self.CONTEXT_NOTES_FILE = alt_data_dir / "context_notes.csv"
        if not self.SAMPLE_OUTPUT_FILE.exists() and (alt_data_dir / "sample_output_format_v2.csv").exists():
            self.SAMPLE_OUTPUT_FILE = alt_data_dir / "sample_output_format_v2.csv"

        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)


default_config = Config()
