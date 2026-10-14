"""
Anomaly detection module for FreightTiger Shipping Cost Assistant.
Detects rising costs that look out of the ordinary compared to history or peers.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class AnomalyDetectionPolicy:
    """
    Configurable anomaly detection policy.

    Rationale:
    - hist_threshold_pct (default: 15.0%): Routes exhibiting a >= 15% cost rise over their
      trailing 8-week historical rolling baseline represent sharp, out-of-the-ordinary increases.
    - peer_threshold_pct (default: 20.0%): Routes costing >= 20% more than other routes of the
      same route_type in the same week (while also rising, diff_hist > 0) are significantly detached from peer pricing.
    - require_both (default: False): As specified by the PDF ('compared to its own history OR to similar routes'),
      an anomaly can be triggered if either condition is met.
    - min_hist_observations (default: 1): Minimum historical weeks required to evaluate Baseline A.
    """
    hist_threshold_pct: float = 15.0
    peer_threshold_pct: float = 20.0
    require_both: bool = False
    min_hist_observations: int = 1


def evaluate_route_week_anomaly(
    diff_hist_pct: Optional[float],
    diff_peer_pct: Optional[float],
    hist_obs_count: int,
    policy: AnomalyDetectionPolicy,
) -> Tuple[bool, str]:
    """
    Evaluates whether a single route-week is an anomaly.

    Returns:
        (is_anomaly, trigger_reason)
    """
    hist_valid = (
        diff_hist_pct is not None
        and not pd.isna(diff_hist_pct)
        and hist_obs_count >= policy.min_hist_observations
    )
    peer_valid = diff_peer_pct is not None and not pd.isna(diff_peer_pct)

    # Check Condition 1: Historical surge
    # Must be rising and exceeding the historical threshold
    hist_anomaly = hist_valid and (diff_hist_pct >= policy.hist_threshold_pct)

    # Check Condition 2: Peer detachment
    # Must be rising compared to history (diff_hist_pct > 0 if history is available)
    # and exceeding the peer threshold
    is_rising = (diff_hist_pct > 0) if hist_valid else True
    peer_anomaly = peer_valid and is_rising and (diff_peer_pct >= policy.peer_threshold_pct)

    if policy.require_both:
        if hist_anomaly and peer_anomaly:
            return True, "history_and_peer"
        return False, "none"
    else:
        if hist_anomaly and peer_anomaly:
            return True, "both"
        elif hist_anomaly:
            return True, "history_only"
        elif peer_anomaly:
            return True, "peer_only"
        else:
            return False, "none"


def detect_anomalies(
    df: pd.DataFrame, policy: Optional[AnomalyDetectionPolicy] = None
) -> pd.DataFrame:
    """
    Applies anomaly detection policy across all route-weeks.
    Appends:
    - 'is_candidate_anomaly': bool
    - 'anomaly_trigger': str ('both', 'history_only', 'peer_only', 'none')
    """
    if policy is None:
        policy = AnomalyDetectionPolicy()

    df = df.copy()
    is_anomalies = []
    triggers = []

    for _, row in df.iterrows():
        is_anom, trig = evaluate_route_week_anomaly(
            diff_hist_pct=row.get("diff_hist_pct"),
            diff_peer_pct=row.get("diff_peer_pct"),
            hist_obs_count=int(row.get("hist_obs_count", 0)),
            policy=policy,
        )
        is_anomalies.append(is_anom)
        triggers.append(trig)

    df["is_candidate_anomaly"] = is_anomalies
    df["anomaly_trigger"] = triggers

    return df
