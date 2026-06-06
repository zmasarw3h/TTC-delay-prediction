from __future__ import annotations

import importlib

import pandas as pd
import pytest

from src.api.historical_lookup import HistoricalFeatureLookup


@pytest.fixture()
def historical_csv(tmp_path):
    path = tmp_path / "modeling_dataset.csv"
    pd.DataFrame(
        [
            {
                "ts": "2024-01-01T08:00:00",
                "mode": "bus",
                "Route": "29",
                "Direction": "N",
                "Incident": "Mechanical",
                "Location": "Dufferin Station",
                "Min Delay": 10,
            },
            {
                "ts": "2024-01-20T08:00:00",
                "mode": "bus",
                "Route": "29",
                "Direction": "N",
                "Incident": "Mechanical",
                "Location": "Dufferin Station",
                "Min Delay": 30,
            },
            {
                "ts": "2024-02-10T08:00:00",
                "mode": "bus",
                "Route": "29",
                "Direction": "N",
                "Incident": "Operations",
                "Location": "Dufferin Station",
                "Min Delay": 60,
            },
            {
                "ts": "2024-02-15T08:00:00",
                "mode": "bus",
                "Route": "29",
                "Direction": "S",
                "Incident": "Mechanical",
                "Location": "Dufferin Station",
                "Min Delay": 90,
            },
            {
                "ts": "2024-02-20T08:00:00",
                "mode": "bus",
                "Route": "29",
                "Direction": "N",
                "Incident": "Mechanical",
                "Location": "Dufferin Station",
                "Min Delay": 120,
            },
            {
                "ts": "2024-02-20T08:30:00",
                "mode": "bus",
                "Route": "29",
                "Direction": "N",
                "Incident": "Mechanical",
                "Location": "Dufferin Station",
                "Min Delay": 240,
            },
            {
                "ts": "2024-02-21T08:00:00",
                "mode": "bus",
                "Route": "29",
                "Direction": "N",
                "Incident": "Mechanical",
                "Location": "Dufferin Station",
                "Min Delay": 180,
            },
        ]
    ).to_csv(path, index=False)
    return path


def compute(path, timestamp="2024-02-20T08:00:00", **overrides):
    payload = {
        "mode": "bus",
        "Route": "29",
        "Direction": "N",
        "Incident": "Mechanical",
        "Location": "Dufferin Stn",
        "timestamp": timestamp,
    }
    payload.update(overrides)
    return HistoricalFeatureLookup(path).compute(payload)


def test_lookup_computes_prior_route_mean_delay_only_before_prediction_timestamp(historical_csv):
    result = compute(historical_csv)

    assert result.features["prior_route_mean_delay"] == pytest.approx((10 + 30 + 60 + 90) / 4)


def test_lookup_excludes_same_timestamp_rows(historical_csv):
    result = compute(historical_csv)

    assert result.features["prior_route_incident_mean_delay"] == pytest.approx((10 + 30 + 90) / 3)
    assert result.features["prior_route_incident_count"] == 3


def test_lookup_excludes_future_rows(historical_csv):
    result = compute(historical_csv)

    assert result.features["prior_global_mean_delay"] == pytest.approx((10 + 30 + 60 + 90) / 4)


def test_route_hour_7d_mean_uses_only_prior_7_day_same_route_hour_rows(historical_csv):
    result = compute(historical_csv)

    assert result.features["prior_route_hour_7d_mean_delay"] == 90


def test_30d_route_mean_excludes_rows_older_than_30_days(historical_csv):
    result = compute(historical_csv)

    assert result.features["prior_route_30d_mean_delay"] == pytest.approx((60 + 90) / 2)


def test_severe_rate_features_compute_prior_rates(historical_csv):
    result = compute(historical_csv)

    assert result.features["prior_route_30d_severe_rate_30"] == 1.0
    assert result.features["prior_route_30d_severe_rate_60"] == 1.0
    assert result.features["prior_incident_30d_severe_rate_30"] == 1.0
    assert result.features["prior_incident_30d_severe_rate_60"] == 1.0


def test_route_incident_count_returns_zero_without_prior_support(historical_csv):
    result = compute(historical_csv, Route="501")

    assert result.features["prior_route_incident_count"] == 0
    assert result.features["prior_route_incident_mean_delay"] is None


def test_location_mean_and_count_are_prior_only(historical_csv):
    result = compute(historical_csv)

    assert result.features["prior_location_mean_delay"] == pytest.approx((10 + 30 + 60 + 90) / 4)
    assert result.features["prior_location_count"] == 4


def test_historical_lookup_module_is_import_safe(monkeypatch):
    monkeypatch.delenv("TTC_HISTORICAL_FEATURE_DATA_PATH", raising=False)

    module = importlib.reload(importlib.import_module("src.api.historical_lookup"))

    assert module.HistoricalFeatureLookup().is_loaded is False
