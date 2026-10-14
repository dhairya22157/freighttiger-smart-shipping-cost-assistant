"""
Unit tests for exact output CSV schema, column order, data types, and RFC 4180 parsing.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.output_writer import CONTRACT_COLUMNS, validate_output_dataframe, OutputContractError
from src.pipeline import ShippingCostAssistantPipeline
from src.config import default_config


def test_output_csv_conforms_to_contract():
    pipeline = ShippingCostAssistantPipeline(default_config)
    output_df, summary = pipeline.run()

    # 1. Exact columns and order
    assert list(output_df.columns) == CONTRACT_COLUMNS

    # 2. Validate using our validator
    validate_output_dataframe(output_df)

    # 3. Verify file on disk can be read back cleanly with pandas without parsing errors
    disk_path = Path(summary["output_path"])
    assert disk_path.exists()

    read_back_df = pd.read_csv(disk_path)
    assert list(read_back_df.columns) == CONTRACT_COLUMNS
    assert len(read_back_df) == len(output_df)

    # 4. Check flagged values are strictly in allowed set
    allowed_flags = {"Yes", "No (justified)", "No"}
    actual_flags = set(read_back_df["flagged"].unique())
    assert actual_flags.issubset(allowed_flags)

    # 5. Check matched_note_id consistency
    for _, row in read_back_df.iterrows():
        flag = row["flagged"]
        note_id = str(row["matched_note_id"]) if pd.notnull(row["matched_note_id"]) else ""
        if flag == "No (justified)":
            assert len(note_id) > 0
            assert note_id.startswith("N")
        else:
            assert note_id == "" or note_id == "nan"


def test_contract_validation_rejects_malformed_dataframe():
    # Test column mismatch
    bad_df = pd.DataFrame({"wrong_col": [1, 2]})
    with pytest.raises(OutputContractError, match="Columns do not match contract"):
        validate_output_dataframe(bad_df)

    # Test missing mandatory field
    bad_row_df = pd.DataFrame([{
        "route": "Delhi-Jaipur",
        "week_of": "2024-01-01",
        "cost_per_tonne_km": 4.17,
        "vs_own_history": "+10%",
        "vs_similar_routes": "+5%",
        "flagged": "Yes",
        "matched_note_id": "",
        "reason": None,  # Null reason
    }])
    with pytest.raises(OutputContractError, match="contains 1 null values"):
        validate_output_dataframe(bad_row_df)

    # Test unjustified row with non-empty matched_note_id
    bad_id_df = pd.DataFrame([{
        "route": "Delhi-Jaipur",
        "week_of": "2024-01-01",
        "cost_per_tonne_km": 4.17,
        "vs_own_history": "+10%",
        "vs_similar_routes": "+5%",
        "flagged": "Yes",
        "matched_note_id": "N001",  # Illegal for 'Yes'
        "reason": "Some reason",
    }])
    with pytest.raises(OutputContractError, match="must have an empty matched_note_id"):
        validate_output_dataframe(bad_id_df)
