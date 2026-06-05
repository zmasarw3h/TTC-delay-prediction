from __future__ import annotations

import importlib
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.input_validation import LEAKAGE_FIELDS
from src.features.build_features import FEATURE_COLUMNS


STATIC_DIR = Path(__file__).resolve().parents[1] / "src" / "api" / "static"


class FakeRegressor:
    def __init__(self) -> None:
        self.seen_columns: list[str] | None = None
        self.seen_frame = None

    def predict(self, frame):
        self.seen_columns = list(frame.columns)
        self.seen_frame = frame.copy()
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
            "Route": ["29", "501", "32A", "RAD", "Mechanical delay at station"],
            "Direction": ["N", "S", "E", "Mechanical delay at station"],
            "Incident": ["Mechanical", "Operations", "501"],
            "Location": [
                "Dufferin Station",
                "Queen Street West and Spadina Avenue",
            ],
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


def test_root_serves_demo_frontend(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Incident-time TTC delay prediction demo" in response.text
    assert "Model status" not in response.text
    assert "Calibrated two-output service" not in response.text
    assert response.text.count('data-preset="') == 2
    assert 'data-preset="bus"' in response.text
    assert 'data-preset="streetcar"' in response.text


def test_static_demo_files_are_served(client):
    for path, expected_text in [
        ("/static/styles.css", ".site-header"),
        ("/static/app.js", "POST"),
    ]:
        response = client.get(path)

        assert response.status_code == 200
        assert expected_text in response.text


def test_demo_static_files_exist():
    for filename in ["index.html", "styles.css", "app.js"]:
        assert (STATIC_DIR / filename).exists()


def test_demo_preset_payloads_do_not_include_leakage_fields():
    app_js = (STATIC_DIR / "app.js").read_text()

    for field in LEAKAGE_FIELDS:
        assert field not in app_js


def test_health_works(client, fake_artifact_path):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_artifact_loaded": False,
        "artifact_exists": True,
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


def test_model_options_returns_expected_structure(client):
    response = client.get("/model-options")

    assert response.status_code == 200
    body = response.json()
    assert body["modes"] == ["bus", "streetcar"]
    assert body["routes"] == ["29", "501", "32A", "RAD"]
    assert body["directions"] == ["N", "E", "S", "W"]
    assert "Mechanical" in body["incidents"]
    assert "Operations" in body["incidents"]
    assert "501" not in body["incidents"]
    assert "Dufferin Station" in body["locations"]
    assert body["counts"]["locations"] == 2
    assert "Mechanical delay at station" not in body["directions"]
    assert any("Excluded 1 non-route-like Route" in warning for warning in body["warnings"])


def test_model_options_directions_are_fixed_even_with_polluted_artifact(client):
    response = client.get("/model-options")

    assert response.status_code == 200
    body = response.json()
    assert body["directions"] == ["N", "E", "S", "W"]
    assert "Mechanical delay at station" not in body["directions"]


def test_match_location_handles_exact_match(client):
    response = client.post("/match-location", json={"location": "dufferin station"})

    assert response.status_code == 200
    body = response.json()
    assert body["original_location"] == "dufferin station"
    assert body["matched_location"] == "Dufferin Station"
    assert body["score"] == 100.0
    assert body["match_type"] == "exact"
    assert body["accepted_for_prediction"] is True
    assert body["warning"] is None


def test_match_location_handles_fuzzy_or_contains_match(client):
    response = client.post("/match-location", json={"location": "queen and spadina"})

    assert response.status_code == 200
    body = response.json()
    assert body["matched_location"] == "Queen Street West and Spadina Avenue"
    assert body["score"] >= 75.0
    assert body["match_type"] == "fuzzy"
    if body["score"] >= 90.0:
        assert body["accepted_for_prediction"] is True
    else:
        assert body["accepted_for_prediction"] is False
        assert "Possible location match" in body["warning"]


def test_match_location_handles_no_match(client):
    response = client.post("/match-location", json={"location": "Mars Base"})

    assert response.status_code == 200
    body = response.json()
    assert body["match_type"] == "none"
    assert body["accepted_for_prediction"] is False
    assert "No confident location match" in body["warning"]


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
    assert any("Derived is_holiday from timestamp" in warning for warning in body["warnings"])
    app_module = importlib.import_module("src.api.app")
    scored = app_module.prediction_service.artifact["expected_delay_regressor"].seen_frame.iloc[0]
    assert scored["hour"] == 8
    assert scored["day_of_week"] == 5
    assert scored["month"] == 2
    assert scored["is_weekend"] == 1
    assert scored["day_of_year"] == 34
    assert scored["is_holiday"] == 0


def test_timestamp_derives_holiday_flag(client):
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
    payload["timestamp"] = "2024-12-25T08:30:00"

    response = client.post("/predict-delay", json=payload)

    assert response.status_code == 200
    app_module = importlib.import_module("src.api.app")
    scored = app_module.prediction_service.artifact["expected_delay_regressor"].seen_frame.iloc[0]
    assert scored["is_holiday"] == 1


def test_provided_is_holiday_is_not_overwritten(client):
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
    ]:
        payload.pop(field)
    payload["timestamp"] = "2024-12-25T08:30:00"
    payload["is_holiday"] = 0

    response = client.post("/predict-delay", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert any("is_holiday was provided by caller and was not overwritten" in warning for warning in body["warnings"])
    app_module = importlib.import_module("src.api.app")
    scored = app_module.prediction_service.artifact["expected_delay_regressor"].seen_frame.iloc[0]
    assert scored["is_holiday"] == 0


def test_missing_timestamp_and_missing_is_holiday_falls_back_to_zero(client):
    payload = valid_payload()
    payload.pop("timestamp", None)
    payload.pop("is_holiday")

    response = client.post("/predict-delay", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert any(
        "is_holiday was missing and set to 0 because no timestamp was provided" in warning
        for warning in body["warnings"]
    )
    app_module = importlib.import_module("src.api.app")
    scored = app_module.prediction_service.artifact["expected_delay_regressor"].seen_frame.iloc[0]
    assert scored["is_holiday"] == 0


def test_invalid_timestamp_returns_clear_error(client):
    payload = valid_payload()
    payload["timestamp"] = "not-a-date"
    payload.pop("is_holiday")

    response = client.post("/predict-delay", json=payload)

    assert response.status_code == 422
    assert "Invalid timestamp" in response.text


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
