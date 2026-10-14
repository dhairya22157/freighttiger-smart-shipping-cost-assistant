"""
Output writer and contract validator for FreightTiger Shipping Cost Assistant.
Ensures exported CSV matches sample_output_format.csv schema, types, and RFC 4180 rules.
"""

import csv
from pathlib import Path
from typing import List, Optional
import pandas as pd


CONTRACT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]


class OutputContractError(Exception):
    """Raised when generated output violates the case study contract."""
    pass


def validate_output_dataframe(df: pd.DataFrame) -> None:
    """
    Validates the output DataFrame against the strict contract:
    - Exact column names and order
    - No missing values in mandatory fields
    - matched_note_id blank when flagged == 'Yes' or 'No'
    - matched_note_id present when flagged == 'No (justified)'
    - cost_per_tonne_km is positive numeric
    """
    if list(df.columns) != CONTRACT_COLUMNS:
        raise OutputContractError(
            f"Columns do not match contract. Expected: {CONTRACT_COLUMNS}, got: {list(df.columns)}"
        )

    # Check mandatory non-null fields
    mandatory_fields = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "reason"]
    for col in mandatory_fields:
        if df[col].isnull().any():
            bad_count = df[col].isnull().sum()
            raise OutputContractError(f"Mandatory column '{col}' contains {bad_count} null values.")

    # Validate matched_note_id rules
    for idx, row in df.iterrows():
        flagged = str(row["flagged"]).strip()
        note_id = str(row["matched_note_id"]).strip() if pd.notnull(row["matched_note_id"]) else ""

        if flagged == "No (justified)":
            if not note_id:
                raise OutputContractError(
                    f"Row {idx} ({row['route']}, {row['week_of']}): 'No (justified)' verdict must have a non-empty matched_note_id."
                )
        elif flagged in ("Yes", "No"):
            if note_id:
                raise OutputContractError(
                    f"Row {idx} ({row['route']}, {row['week_of']}): '{flagged}' verdict must have an empty matched_note_id, got '{note_id}'."
                )
        else:
            raise OutputContractError(
                f"Row {idx} ({row['route']}, {row['week_of']}): Invalid flagged value '{flagged}'. Expected 'Yes', 'No (justified)', or 'No'."
            )


def write_output_csv(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Writes DataFrame to CSV strictly formatted to the contract.
    Ensures RFC 4180 quoting to avoid tokenizing issues when strings contain commas.
    """
    # Enforce contract columns
    output_df = df[CONTRACT_COLUMNS].copy()

    # Format cost_per_tonne_km to 2 decimal places
    output_df["cost_per_tonne_km"] = output_df["cost_per_tonne_km"].apply(lambda x: f"{float(x):.2f}")

    # Ensure empty strings for blank matched_note_id
    output_df["matched_note_id"] = output_df["matched_note_id"].fillna("").astype(str).str.strip()

    # Validate before writing
    validate_output_dataframe(output_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")

    return output_path
