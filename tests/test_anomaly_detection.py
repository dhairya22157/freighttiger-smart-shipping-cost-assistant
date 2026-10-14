"""
Unit tests for anomaly detection policy and triggers.
"""

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import (
    AnomalyDetectionPolicy,
    evaluate_route_week_anomaly,
    detect_anomalies,
)


def test_evaluate_route_week_anomaly_triggers():
    policy = AnomalyDetectionPolicy(hist_threshold_pct=15.0, peer_threshold_pct=20.0, require_both=False)

    # 1. Both triggered: hist = 30%, peer = 25%
    is_anom, trig = evaluate_route_week_anomaly(30.0, 25.0, hist_obs_count=5, policy=policy)
    assert is_anom is True
    assert trig == "both"

    # 2. History only: hist = 20%, peer = 5%
    is_anom, trig = evaluate_route_week_anomaly(20.0, 5.0, hist_obs_count=4, policy=policy)
    assert is_anom is True
    assert trig == "history_only"

    # 3. Peer only: hist = 8% (rising), peer = 22%
    is_anom, trig = evaluate_route_week_anomaly(8.0, 22.0, hist_obs_count=8, policy=policy)
    assert is_anom is True
    assert trig == "peer_only"

    # 4. Normal: hist = 2%, peer = 5%
    is_anom, trig = evaluate_route_week_anomaly(2.0, 5.0, hist_obs_count=8, policy=policy)
    assert is_anom is False
    assert trig == "none"

    # 5. Falling cost: hist = -10%, peer = 25% -> cost is falling, so not a rising anomaly
    is_anom, trig = evaluate_route_week_anomaly(-10.0, 25.0, hist_obs_count=8, policy=policy)
    assert is_anom is False
    assert trig == "none"


def test_unavailable_baselines():
    policy = AnomalyDetectionPolicy(hist_threshold_pct=15.0, peer_threshold_pct=20.0, require_both=False)

    # Historical unavailable (first week, hist_obs_count = 0)
    # If peer is high:
    is_anom, trig = evaluate_route_week_anomaly(np.nan, 25.0, hist_obs_count=0, policy=policy)
    assert is_anom is True
    assert trig == "peer_only"

    # Peer unavailable (np.nan), history is high:
    is_anom, trig = evaluate_route_week_anomaly(25.0, np.nan, hist_obs_count=5, policy=policy)
    assert is_anom is True
    assert trig == "history_only"

    # Both unavailable:
    is_anom, trig = evaluate_route_week_anomaly(np.nan, np.nan, hist_obs_count=0, policy=policy)
    assert is_anom is False
    assert trig == "none"


def test_require_both_policy():
    strict_policy = AnomalyDetectionPolicy(hist_threshold_pct=15.0, peer_threshold_pct=20.0, require_both=True)

    # History high, peer low -> should NOT trigger
    is_anom, _ = evaluate_route_week_anomaly(25.0, 5.0, hist_obs_count=5, policy=strict_policy)
    assert is_anom is False

    # Both high -> should trigger
    is_anom, trig = evaluate_route_week_anomaly(25.0, 25.0, hist_obs_count=5, policy=strict_policy)
    assert is_anom is True
    assert trig == "history_and_peer"
