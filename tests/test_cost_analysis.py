"""
Unit tests for weekly cost aggregation and comparison baselines A and B.
Verifies weighted vs unweighted aggregation, no-look-ahead rolling window,
peer isolation, and sample dataset ground truth numbers.
"""

import numpy as np
import pandas as pd
import pytest

from src.cost_analysis import (
    aggregate_weekly_costs,
    compute_baseline_a_history,
    compute_baseline_b_peers,
    compute_all_metrics,
)
from src.data_loader import load_shipments
from src.preprocessing import preprocess_shipments
from src.config import default_config


def test_weighted_vs_unweighted_aggregation():
    """
    Ensures that weekly_cost_per_tonne_km = sum(cost) / sum(quantity * distance)
    and does NOT compute unweighted average of ratios.
    """
    df = pd.DataFrame([
        {
            "shipment_id": "S1",
            "origin": "CityA",
            "destination": "CityB",
            "route": "CityA-CityB",
            "route_type": "Short",
            "quantity_tonnes": 10.0,
            "distance_km": 100.0,
            "freight_cost_inr": 2000.0,  # ratio = 2.0
            "week_of": "2024-01-01",
        },
        {
            "shipment_id": "S2",
            "origin": "CityA",
            "destination": "CityB",
            "route": "CityA-CityB",
            "route_type": "Short",
            "quantity_tonnes": 100.0,
            "distance_km": 100.0,
            "freight_cost_inr": 10000.0,  # ratio = 1.0
            "week_of": "2024-01-01",
        },
    ])
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]

    weekly = aggregate_weekly_costs(df)
    assert len(weekly) == 1

    expected_cost = (2000.0 + 10000.0) / (1000.0 + 10000.0)  # 12000 / 11000 ≈ 1.090909
    assert np.isclose(weekly.loc[0, "cost_per_tonne_km"], expected_cost)
    # Confirm it is NOT the unweighted average of ratios (1.5)
    assert not np.isclose(weekly.loc[0, "cost_per_tonne_km"], 1.5)


def test_baseline_a_no_lookahead_and_window():
    """
    Verifies trailing 8-week window:
    - Week 1 has no prior observations (np.nan)
    - Week 2 uses Week 1
    - Week 9 uses Weeks 1-8
    - Week 10 uses Weeks 2-9
    - Never uses current or future weeks
    """
    weeks = [f"2024-01-{i:02d}" for i in range(1, 11)]
    costs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    weekly = pd.DataFrame({
        "route": ["Route1"] * 10,
        "route_type": ["Short"] * 10,
        "week_of": weeks,
        "cost_per_tonne_km": costs,
    })

    result = compute_baseline_a_history(weekly, window_size=8)

    # Week 1: index 0 -> no prior weeks
    assert pd.isna(result.loc[0, "hist_avg_8w"])
    assert result.loc[0, "hist_obs_count"] == 0

    # Week 2: index 1 -> prior: [1.0] -> avg = 1.0
    assert np.isclose(result.loc[1, "hist_avg_8w"], 1.0)
    assert result.loc[1, "hist_obs_count"] == 1

    # Week 8: index 7 -> prior: costs[0:7] = 1..7 -> avg = 4.0
    assert np.isclose(result.loc[7, "hist_avg_8w"], 4.0)
    assert result.loc[7, "hist_obs_count"] == 7

    # Week 9: index 8 -> prior: costs[0:8] = 1..8 -> avg = 4.5
    assert np.isclose(result.loc[8, "hist_avg_8w"], 4.5)
    assert result.loc[8, "hist_obs_count"] == 8

    # Week 10: index 9 -> prior: costs[1:9] = 2..9 -> avg = 5.5
    assert np.isclose(result.loc[9, "hist_avg_8w"], 5.5)
    assert result.loc[9, "hist_obs_count"] == 8


def test_baseline_b_peer_isolation():
    """
    Verifies that Baseline B:
    - Only includes peers in the same week with the same route_type
    - Excludes the route itself
    - Returns np.nan when peer group is empty
    """
    weekly = pd.DataFrame([
        # Short routes in week 1
        {"route": "RouteA", "route_type": "Short", "week_of": "2024-01-01", "cost_per_tonne_km": 3.0},
        {"route": "RouteB", "route_type": "Short", "week_of": "2024-01-01", "cost_per_tonne_km": 5.0},
        # Long route in week 1 (only 1 Long route -> no peers)
        {"route": "RouteC", "route_type": "Long", "week_of": "2024-01-01", "cost_per_tonne_km": 2.0},
        # RouteA in week 2 (only 1 Short route in week 2)
        {"route": "RouteA", "route_type": "Short", "week_of": "2024-01-08", "cost_per_tonne_km": 4.0},
    ])

    result = compute_baseline_b_peers(weekly)

    # RouteA in week 1 peer is RouteB (cost 5.0)
    row_a = result[(result["route"] == "RouteA") & (result["week_of"] == "2024-01-01")].iloc[0]
    assert np.isclose(row_a["peer_avg"], 5.0)
    assert row_a["peer_count"] == 1

    # RouteB in week 1 peer is RouteA (cost 3.0)
    row_b = result[(result["route"] == "RouteB") & (result["week_of"] == "2024-01-01")].iloc[0]
    assert np.isclose(row_b["peer_avg"], 3.0)
    assert row_b["peer_count"] == 1

    # RouteC in week 1 has no Long peers -> np.nan
    row_c = result[(result["route"] == "RouteC") & (result["week_of"] == "2024-01-01")].iloc[0]
    assert pd.isna(row_c["peer_avg"])
    assert row_c["peer_count"] == 0

    # RouteA in week 2 has no Short peers in week 2 -> np.nan
    row_a_w2 = result[(result["route"] == "RouteA") & (result["week_of"] == "2024-01-08")].iloc[0]
    assert pd.isna(row_a_w2["peer_avg"])
    assert row_a_w2["peer_count"] == 0


def test_match_sample_output_ground_truth():
    """
    Loads the real shipment dataset and verifies against the exact values in sample_output_format_v2.csv:
    1. Delhi-Jaipur (2024-11-11): 4.17, +35.5% vs past average, +21.0% vs similar-length routes
    2. Ahmedabad-Mumbai (2025-01-20): 3.29, +29.5% vs past average, +22.5% vs similar-length routes
    3. Mumbai-Pune (2025-09-15): 3.98, +9.2% vs past average, +23.6% vs similar-length routes
    """
    df_raw = load_shipments(default_config.SHIPMENTS_FILE, default_config.REQUIRED_SHIPMENT_COLUMNS)
    df_prep = preprocess_shipments(df_raw)
    metrics_df = compute_all_metrics(df_prep, window_size=8)

    # 1. Delhi-Jaipur 2024-11-11
    dj = metrics_df[(metrics_df["route"] == "Delhi-Jaipur") & (metrics_df["week_of"] == "2024-11-11")].iloc[0]
    assert np.isclose(round(dj["cost_per_tonne_km"], 2), 4.17)
    assert dj["vs_own_history"] == "+35.5% vs this route's past average"
    assert dj["vs_similar_routes"] == "+21.0% vs similar-length routes this week"

    # 2. Ahmedabad-Mumbai 2025-01-20
    am = metrics_df[(metrics_df["route"] == "Ahmedabad-Mumbai") & (metrics_df["week_of"] == "2025-01-20")].iloc[0]
    assert np.isclose(round(am["cost_per_tonne_km"], 2), 3.29)
    assert am["vs_own_history"] == "+29.5% vs this route's past average"
    assert am["vs_similar_routes"] == "+22.5% vs similar-length routes this week"

    # 3. Mumbai-Pune 2025-09-15
    mp = metrics_df[(metrics_df["route"] == "Mumbai-Pune") & (metrics_df["week_of"] == "2025-09-15")].iloc[0]
    assert np.isclose(round(mp["cost_per_tonne_km"], 2), 3.98)
    assert mp["vs_own_history"] == "+9.2% vs this route's past average"
    assert mp["vs_similar_routes"] == "+23.6% vs similar-length routes this week"
