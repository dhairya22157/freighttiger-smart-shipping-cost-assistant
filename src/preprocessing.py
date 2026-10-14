"""
Preprocessing module for FreightTiger Shipping Cost Assistant.
Handles route normalization, Monday-to-Sunday weekly bucketing, and tonne-km calculations.
"""

from datetime import datetime
import pandas as pd


def get_monday_week_of(dt: pd.Timestamp) -> str:
    """
    Given a timestamp or datetime, returns the ISO date string (YYYY-MM-DD)
    of the Monday of that week.
    Monday is weekday 0, Sunday is weekday 6.
    """
    monday = dt - pd.Timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d")


def preprocess_shipments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares shipment records:
    - Generates 'route' = '{origin}-{destination}'
    - Parses shipment_date to datetime if not already done
    - Computes 'week_of' (Monday date string)
    - Computes 'tonne_km' = quantity_tonnes * distance_km
    - Computes individual 'cost_per_tonne_km' = freight_cost_inr / tonne_km
    """
    df = df.copy()

    # Route identity
    df["route"] = df["origin"].astype(str).str.strip() + "-" + df["destination"].astype(str).str.strip()

    # Ensure datetime
    if "parsed_shipment_date" not in df.columns:
        df["parsed_shipment_date"] = pd.to_datetime(df["shipment_date"], format="%Y-%m-%d")

    # Compute Monday week_of
    df["week_of"] = df["parsed_shipment_date"].apply(get_monday_week_of)

    # Tonne-km and individual ratio
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    df["cost_per_tonne_km"] = df["freight_cost_inr"] / df["tonne_km"]

    return df


def preprocess_notes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares context notes:
    - Parses date
    - Computes note's Monday 'week_of'
    - Normalizes applies_to
    """
    df = df.copy()
    if "parsed_date" not in df.columns:
        df["parsed_date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    df["week_of"] = df["parsed_date"].apply(get_monday_week_of)
    df["applies_to"] = df["applies_to"].astype(str).str.strip()
    df["note"] = df["note"].astype(str).str.strip()

    return df
