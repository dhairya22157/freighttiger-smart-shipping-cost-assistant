"""
Cost analysis module for FreightTiger Shipping Cost Assistant.
Computes weekly route cost metrics and strictly adheres to:
- Baseline A: Trailing 8-week rolling average of prior weekly costs (no look-ahead).
- Baseline B: Same-week peer route average sharing the same route_type (excluding self).
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def aggregate_weekly_costs(df_shipments: pd.DataFrame) -> pd.DataFrame:
    """
    Groups shipment records by route, route_type, and week_of.
    Computes weekly aggregate cost per tonne-kilometre:
        weekly_cost_per_tonne_km = sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)

    Preserves total shipment count, total freight cost, and total tonne-km.
    """
    if "route" not in df_shipments.columns or "week_of" not in df_shipments.columns:
        raise ValueError("DataFrame must contain 'route' and 'week_of' columns before aggregation.")

    if "tonne_km" not in df_shipments.columns:
        df_shipments = df_shipments.copy()
        df_shipments["tonne_km"] = df_shipments["quantity_tonnes"] * df_shipments["distance_km"]

    # Verify positive denominators
    if (df_shipments["tonne_km"] <= 0).any():
        raise ValueError("Found non-positive tonne_km in shipment records.")

    weekly = (
        df_shipments.groupby(["route", "route_type", "week_of"], as_index=False)
        .agg(
            total_freight_cost=("freight_cost_inr", "sum"),
            total_tonne_km=("tonne_km", "sum"),
            shipment_count=("shipment_id", "count"),
        )
    )

    # Calculate weekly cost per tonne-km
    weekly["cost_per_tonne_km"] = weekly["total_freight_cost"] / weekly["total_tonne_km"]

    # Ensure chronological ordering by route and week_of
    weekly["parsed_week_of"] = pd.to_datetime(weekly["week_of"])
    weekly = weekly.sort_values(["route", "parsed_week_of"]).reset_index(drop=True)
    weekly = weekly.drop(columns=["parsed_week_of"])

    return weekly


def compute_baseline_a_history(
    weekly_df: pd.DataFrame, window_size: int = 8
) -> pd.DataFrame:
    """
    Baseline A: vs. own history
    Calculates the trailing rolling average cost per tonne-km for each route,
    using strictly prior weeks (no look-ahead).

    Rules:
    - Never include the current week's cost.
    - Never use future data.
    - If fewer than `window_size` prior weeks exist, use all available prior weeks.
    - If no prior observations exist (first week), value is np.nan.
    - Does not pad or extrapolate missing weeks.
    - Records the number of prior observations used for auditing.
    """
    df = weekly_df.copy()
    df["hist_avg_8w"] = np.nan
    df["hist_obs_count"] = 0

    for route, grp in df.groupby("route", sort=False):
        indices = grp.index.tolist()
        costs = grp["cost_per_tonne_km"].values

        for i, idx in enumerate(indices):
            if i == 0:
                # No prior observation exists
                df.loc[idx, "hist_avg_8w"] = np.nan
                df.loc[idx, "hist_obs_count"] = 0
            else:
                # Window of up to `window_size` strictly prior weeks
                start_i = max(0, i - window_size)
                prior_window = costs[start_i:i]
                df.loc[idx, "hist_avg_8w"] = float(np.mean(prior_window))
                df.loc[idx, "hist_obs_count"] = len(prior_window)

    # Calculate percentage difference vs history
    df["diff_hist_pct"] = np.where(
        df["hist_avg_8w"].notnull() & (df["hist_avg_8w"] > 0),
        ((df["cost_per_tonne_km"] - df["hist_avg_8w"]) / df["hist_avg_8w"]) * 100.0,
        np.nan,
    )

    return df


def compute_baseline_b_peers(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline B: vs. similar routes
    Calculates the average cost per tonne-km in that same week across all OTHER routes
    sharing the same route_type (Short / Medium / Long).

    Rules:
    - Same week only.
    - Same route_type only.
    - Exclude the route itself.
    - If peer group is empty, value is np.nan (not zero).
    - Records peer count for auditing.
    """
    df = weekly_df.copy()
    df["peer_avg"] = np.nan
    df["peer_count"] = 0

    for (week_of, route_type), grp in df.groupby(["week_of", "route_type"], sort=False):
        routes_in_grp = grp["route"].unique()
        if len(routes_in_grp) <= 1:
            # No peers available for comparison
            continue

        for idx, row in grp.iterrows():
            peers = grp[grp["route"] != row["route"]]
            if len(peers) > 0:
                df.loc[idx, "peer_avg"] = float(peers["cost_per_tonne_km"].mean())
                df.loc[idx, "peer_count"] = int(len(peers))

    # Calculate percentage difference vs peer average
    df["diff_peer_pct"] = np.where(
        df["peer_avg"].notnull() & (df["peer_avg"] > 0),
        ((df["cost_per_tonne_km"] - df["peer_avg"]) / df["peer_avg"]) * 100.0,
        np.nan,
    )

    return df


def format_comparison_str(pct_val: Optional[float], comparison_type: str) -> str:
    """
    Formats the numeric percentage difference to match sample_output_format.csv:
    - vs_own_history: '+35.5% vs this route\'s past average'
    - vs_similar_routes: '+21.0% vs similar-length routes this week'
    """
    if pct_val is None or pd.isna(pct_val):
        if comparison_type == "own_history":
            return "N/A (no prior history)"
        else:
            return "N/A (no peer routes)"

    sign = "+" if pct_val > 0 else ""
    formatted_pct = f"{sign}{pct_val:.1f}%"

    if comparison_type == "own_history":
        return f"{formatted_pct} vs this route's past average"
    elif comparison_type == "similar_routes":
        return f"{formatted_pct} vs similar-length routes this week"
    else:
        return formatted_pct


def compute_all_metrics(
    df_shipments: pd.DataFrame, window_size: int = 8
) -> pd.DataFrame:
    """
    Full pipeline function to calculate weekly route costs and both baselines.
    Adds formatted comparison string columns matching the output contract:
    - 'vs_own_history'
    - 'vs_similar_routes'
    """
    weekly = aggregate_weekly_costs(df_shipments)
    weekly = compute_baseline_a_history(weekly, window_size=window_size)
    weekly = compute_baseline_b_peers(weekly)

    # Format contract columns
    weekly["vs_own_history"] = weekly["diff_hist_pct"].apply(
        lambda x: format_comparison_str(x, "own_history")
    )
    weekly["vs_similar_routes"] = weekly["diff_peer_pct"].apply(
        lambda x: format_comparison_str(x, "similar_routes")
    )

    return weekly
