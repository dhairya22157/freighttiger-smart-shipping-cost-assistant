"""
Data loading and validation module.
Enforces strict schema validation, positive numeric values, and valid date parsing.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


class DataValidationError(Exception):
    """Raised when dataset fails schema, integrity, or domain validation checks."""
    pass


def load_shipments(file_path: Path, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Loads shipment records from CSV and applies strict validation.

    Rules:
    - File must exist.
    - Required columns must all be present.
    - quantity_tonnes, distance_km, freight_cost_inr must be positive numbers.
    - shipment_date must be valid YYYY-MM-DD dates.
    - route_type must be one of 'Short', 'Medium', 'Long'.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Shipment records file not found at: {file_path}")

    df = pd.read_csv(file_path)

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns in shipment records: {sorted(list(missing_cols))}"
            )

    # Check for nulls in critical fields
    critical_cols = ["shipment_id", "origin", "destination", "route_type", "quantity_tonnes", "distance_km", "freight_cost_inr", "shipment_date"]
    null_counts = df[critical_cols].isnull().sum()
    if null_counts.any():
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise DataValidationError(f"Null values found in critical shipment columns: {bad_cols}")

    # Validate numeric types and bounds
    for col in ["quantity_tonnes", "distance_km", "freight_cost_inr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            raise DataValidationError(f"Column '{col}' contains non-numeric values.")
        non_positive = (df[col] <= 0).sum()
        if non_positive > 0:
            raise DataValidationError(
                f"Column '{col}' contains {non_positive} non-positive values (must be > 0)."
            )

    # Validate date parsing
    try:
        df["parsed_shipment_date"] = pd.to_datetime(df["shipment_date"], format="%Y-%m-%d", errors="raise")
    except Exception as e:
        raise DataValidationError(f"Invalid date format in 'shipment_date' column. Expected YYYY-MM-DD: {e}")

    # Validate route_type
    valid_route_types = {"Short", "Medium", "Long"}
    invalid_types = set(df["route_type"].unique()) - valid_route_types
    if invalid_types:
        raise DataValidationError(f"Invalid route_type values found: {invalid_types}. Allowed: {valid_route_types}")

    return df


def load_context_notes(file_path: Path, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Loads context notes from CSV and validates schema and date formats.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Context notes file not found at: {file_path}")

    df = pd.read_csv(file_path)

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns in context notes: {sorted(list(missing_cols))}"
            )

    # Ensure required fields are not empty
    for col in ["note_id", "date", "applies_to", "note"]:
        if df[col].isnull().any():
            raise DataValidationError(f"Context notes contain empty entries in '{col}'.")

    # Validate date parsing
    try:
        df["parsed_date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
    except Exception as e:
        raise DataValidationError(f"Invalid date in context notes 'date' column: {e}")

    return df


def generate_dataset_summary(shipments_df: pd.DataFrame, notes_df: pd.DataFrame) -> Dict[str, any]:
    """
    Generates a structured summary report of the datasets for auditing and documentation.
    """
    shipments_df["route"] = shipments_df["origin"] + "-" + shipments_df["destination"]

    summary = {
        "shipment_records_count": len(shipments_df),
        "shipment_date_min": str(shipments_df["shipment_date"].min()),
        "shipment_date_max": str(shipments_df["shipment_date"].max()),
        "unique_origins": sorted(shipments_df["origin"].unique().tolist()),
        "unique_destinations": sorted(shipments_df["destination"].unique().tolist()),
        "unique_routes": sorted(shipments_df["route"].unique().tolist()),
        "route_types": shipments_df["route_type"].value_counts().to_dict(),
        "total_freight_spend_inr": float(shipments_df["freight_cost_inr"].sum()),
        "total_tonnes_moved": float(shipments_df["quantity_tonnes"].sum()),
        "context_notes_count": len(notes_df),
        "notes_routes_mentioned": sorted(notes_df["applies_to"].unique().tolist()),
    }
    return summary
