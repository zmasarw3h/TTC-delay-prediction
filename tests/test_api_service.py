from __future__ import annotations

import importlib

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.features.build_features import FEATURE_COLUMNS


class FakeRegressor:
    def __init__(self) -> None:
        self.seen_columns: list[str] | None = None

    def predict(self, frame):
        self.seen_columns = list(frame.columns)
        return np.array([12.5])


class FakeClassifier:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.seen_columns: list[str] | None = None

    def predict_proba(self, frame):
        self.seen_columns = list(frame.columns)
        return np.array([[1.0 - self.probability, self.probability]])


@pytest.fixture()
def fake_artifact_path(tmp_path):
    path = tmp_path / "fake_calibrated_model.joblib"
    artifact = {
        "model_name": "fake_calibrated_two_output_model",
        "model_phase": "Phase 9 test",
        "expected_delay_regressor": FakeRegressor(),
        "calibrated_risk_classifiers": {
            30: FakeClassifier(0.35),
            60: FakeClassifier(0.08),
        },
        "selected_calibration_methods": {
            "30": {"calibration_method": "sigmoid"},
            "60": {"calibration_method": "isotonic"},
        },
        "selected_probability_thresholds": {
            "30": {"probability_cutoff": 0.20},
            "60": {"probability_cutoff": 0.30},
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": "Min Delay",
        "risk_band_definitions": {
            "low": {"min_probability": 0.0, "max_probability": 0.10},
            "medium": {"min_probability": 0.10, "max_probability": 0.30},
            "high": {"min_probability": 0.30, "max_probability": 1.0},
        },
        "known_categories": {
            "mode": ["bus", "streetcar"],
            "Route": ["29"],
            "Direction": ["N", "S"],
            "Incident": ["Mechanical"],
            "Location": ["Dufferin Station"],
        },
        "metadata": {"notes": ["Fake artifact for API unit tests."]},
    }
    joblib.dump(artifact, path)
    return path


@pytest.fixture()
def client(fake_artifact_path, monkeypatch):
    monkeypatch.setenv("TTC_MODEL_ARTIFACT_PATH", str(fake_artifact_path))
    app_module = importlib.import_module("src.api.app")
    app_module.prediction_service = app_module.CalibratedDelayPredictionService()
    return TestClient(app_module.app)


def valid_payload() -> dict:
    return {
        "mode": "bus",
        "Route": "29",
        "Direction": "N",
        "Incident": "Mechanical",
        "Location": "Dufferin Station",
        "hour": 8,
        "day_of_week": 1,
        "month": 2,
        "is_weekend": 0,
        "is_holiday": 0,
        "hour_sin": 0.866,
        "hour_cos": -0.5,
        "day_of_year": 35,
        "day_sin": 0.565,
        "day_cos": 0.825,
        "prior_route_mean_delay": 10.0,
        "prior_route_hour_mean_delay": 12.0,
        "prior_incident_mean_delay": 9.0,
        "prior_mode_mean_delay": 8.0,
        "prior_global_mean_delay": 7.0,
        "prior_route_hour_7d_mean_delay": 11.0,
    }


def test_health_works(client, fake_artifact_path):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_artifact_loaded": False,
        "artifact_path": str(fake_artifact_path),
    }


def test_model_info_works(client):
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "fake_calibrated_two_output_model"
    assert body["model_phase"] == "Phase 9 test"
    assert body["feature_columns"] == FEATURE_COLUMNS
    assert body["target_column"] == "Min Delay"
    assert body["risk_thresholds"] == [30, 60]
    assert body["selected_calibration_methods"]["30"]["calibration_method"] == "sigmoid"
    assert "raw TTC incident records" in " ".join(body["notes_limitations"])


def test_predict_delay_returns_expected_response_shape(client):
    response = client.post("/predict-delay", json=valid_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_delay_minutes"] == 12.5
    assert body["calibrated_severe_delay_probability_30"] == 0.35
    assert body["risk_band_30"] == "high"
    assert body["severe_delay_prediction_30"] == 1
    assert body["selected_probability_cutoff_30"] == 0.2
    assert body["calibrated_severe_delay_probability_60"] == 0.08
    assert body["risk_band_60"] == "low"
    assert body["severe_delay_prediction_60"] == 0
    assert body["selected_probability_cutoff_60"] == 0.3
    assert body["model_name"] == "fake_calibrated_two_output_model"
    assert body["model_phase"] == "Phase 9 test"


def test_leakage_fields_are_rejected(client):
    payload = valid_payload()
    payload["Min Delay"] = 15

    response = client.post("/predict-delay", json=payload)

    assert response.status_code == 422
    assert "leakage-sensitive" in response.text


def test_unsupported_mode_is_rejected(client):
    payload = valid_payload()
    payload["mode"] = "subway"

    response = client.post("/predict-delay", json=payload)

    assert response.status_code == 422
    assert "Unsupported mode" in response.text


def test_timestamp_derived_time_fields_work(client):
    payload = valid_payload()
    for field in [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "day_of_year",
        "day_sin",
        "day_cos",
        "is_holiday",
    ]:
        payload.pop(field)
    payload["timestamp"] = "2024-02-03T08:30:00"

    response = client.post("/predict-delay", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert any("Derived missing time fields" in warning for warning in body["warnings"])
    assert any("is_holiday was missing and set to 0" in warning for warning in body["warnings"])


def test_feature_order_matches_artifact_feature_columns(client):
    response = client.post("/predict-delay", json=valid_payload())

    assert response.status_code == 200
    app_module = importlib.import_module("src.api.app")
    artifact = app_module.prediction_service.artifact
    assert artifact["expected_delay_regressor"].seen_columns == FEATURE_COLUMNS
    assert artifact["calibrated_risk_classifiers"][30].seen_columns == FEATURE_COLUMNS
    assert artifact["calibrated_risk_classifiers"][60].seen_columns == FEATURE_COLUMNS


def test_missing_historical_numeric_features_return_warnings(client):
    payload = valid_payload()
    payload["prior_route_mean_delay"] = None
    payload["prior_route_hour_7d_mean_delay"] = None

    response = client.post("/predict-delay", json=payload)

    assert response.status_code == 200
    assert any("Missing historical prior-delay feature" in warning for warning in response.json()["warnings"])


def test_unknown_categorical_values_warn_without_crashing(client):
    payload = valid_payload()
    payload["Route"] = "999"
    payload["Incident"] = "Unlisted"
    payload["Location"] = "Unknown stop"

    response = client.post("/predict-delay", json=payload)

    assert response.status_code == 200
    warning_text = " ".join(response.json()["warnings"])
    assert "Route '999'" in warning_text
    assert "Incident 'Unlisted'" in warning_text
    assert "Location 'Unknown stop'" in warning_text


def test_app_import_is_safe(monkeypatch):
    monkeypatch.delenv("TTC_MODEL_ARTIFACT_PATH", raising=False)

    app_module = importlib.reload(importlib.import_module("src.api.app"))

    assert app_module.app is not None
    assert not app_module.prediction_service.is_loaded
