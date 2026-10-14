"""
Unit tests for data loader and preprocessing.
Tests Monday-to-Sunday weekly bucketing, boundaries, tonne-km, and input validation.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.data_loader import (
    load_shipments,
    load_context_notes,
    generate_dataset_summary,
    DataValidationError,
)
from src.preprocessing import get_monday_week_of, preprocess_shipments, preprocess_notes
from src.config import default_config


def test_monday_week_of_boundaries():
    # 2024-01-01 is a Monday
    assert get_monday_week_of(pd.Timestamp("2024-01-01")) == "2024-01-01"

    # 2024-01-03 is a Wednesday
    assert get_monday_week_of(pd.Timestamp("2024-01-03")) == "2024-01-01"

    # 2024-01-07 is a Sunday -> must bucket to Monday 2024-01-01
    assert get_monday_week_of(pd.Timestamp("2024-01-07")) == "2024-01-01"

    # 2024-01-08 is the next Monday
    assert get_monday_week_of(pd.Timestamp("2024-01-08")) == "2024-01-08"

    # Year boundary: 2024-12-30 (Monday) to 2025-01-05 (Sunday)
    assert get_monday_week_of(pd.Timestamp("2024-12-31")) == "2024-12-30"
    assert get_monday_week_of(pd.Timestamp("2025-01-01")) == "2024-12-30"
    assert get_monday_week_of(pd.Timestamp("2025-01-05")) == "2024-12-30"

    # Month boundary: 2024-02-29 (Thursday) -> Monday 2024-02-26
    assert get_monday_week_of(pd.Timestamp("2024-02-29")) == "2024-02-26"
    assert get_monday_week_of(pd.Timestamp("2024-03-01")) == "2024-02-26"


def test_load_actual_datasets():
    shipments_df = load_shipments(default_config.SHIPMENTS_FILE, default_config.REQUIRED_SHIPMENT_COLUMNS)
    assert len(shipments_df) == 2940
    assert (shipments_df["quantity_tonnes"] > 0).all()
    assert (shipments_df["distance_km"] > 0).all()
    assert (shipments_df["freight_cost_inr"] > 0).all()

    notes_df = load_context_notes(default_config.CONTEXT_NOTES_FILE, default_config.REQUIRED_NOTE_COLUMNS)
    assert len(notes_df) == 10
    assert "note_id" in notes_df.columns

    summary = generate_dataset_summary(shipments_df, notes_df)
    assert summary["shipment_records_count"] == 2940
    assert summary["context_notes_count"] == 10
    assert len(summary["unique_routes"]) == 7


def test_validation_errors_on_invalid_inputs(tmp_path: Path):
    # Test missing column
    bad_csv = tmp_path / "bad_shipments.csv"
    bad_csv.write_text("origin,destination\nMumbai,Pune\n")
    with pytest.raises(DataValidationError, match="Missing required columns"):
        load_shipments(bad_csv, default_config.REQUIRED_SHIPMENT_COLUMNS)

    # Test non-positive quantity
    bad_val_csv = tmp_path / "bad_values.csv"
    bad_df = pd.DataFrame([{
        "shipment_id": "S1",
        "origin": "Mumbai",
        "destination": "Pune",
        "route_type": "Short",
        "material": "Cement",
        "quantity_tonnes": -5.0,  # Invalid negative
        "distance_km": 150.0,
        "freight_cost_inr": 5000,
        "shipment_date": "2024-01-02",
        "transporter": "Carrier"
    }])
    bad_df.to_csv(bad_val_csv, index=False)
    with pytest.raises(DataValidationError, match="non-positive"):
        load_shipments(bad_val_csv, default_config.REQUIRED_SHIPMENT_COLUMNS)


def test_preprocessing_shipments():
    sample_df = pd.DataFrame([{
        "shipment_id": "S1",
        "origin": "Mumbai",
        "destination": "Pune",
        "route_type": "Short",
        "material": "Cement",
        "quantity_tonnes": 10.0,
        "distance_km": 100.0,
        "freight_cost_inr": 2000,
        "shipment_date": "2024-01-06"  # Saturday
    }])
    processed = preprocess_shipments(sample_df)
    assert processed.loc[0, "route"] == "Mumbai-Pune"
    assert processed.loc[0, "week_of"] == "2024-01-01"  # Monday of that week
    assert processed.loc[0, "tonne_km"] == 1000.0
    assert processed.loc[0, "cost_per_tonne_km"] == 2.0
