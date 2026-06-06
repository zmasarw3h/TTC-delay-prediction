from __future__ import annotations

import importlib
from pathlib import Path
import zipfile

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.input_validation import LEAKAGE_FIELDS
from src.features.build_features import FEATURE_COLUMNS, HISTORICAL_FEATURES


STATIC_DIR = Path(__file__).resolve().parents[1] / "src" / "api" / "static"
CURATED_INCIDENT_VALUES = [
    "Mechanical",
    "Utilized Off Route",
    "General Delay",
    "Late Leaving Garage",
    "Investigation",
    "Operations - Operator",
    "Operations",
    "Diversion",
    "Emergency Services",
    "Security",
    "Collision - TTC",
    "Collision - TTC Involved",
    "Road Blocked - NON-TTC Collision",
    "Held By",
    "Cleaning",
    "Cleaning - Unsanitary",
    "Vision",
    "Overhead",
    "Overhead - Pantograph",
    "Rail/Switches",
    "Other",
    "Unknown",
]


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
def fake_historical_path(tmp_path):
    path = tmp_path / "fake_modeling_dataset.csv"
    pd.DataFrame(
        [
            {
                "ts": "2024-01-01T08:00:00",
                "mode": "bus",
                "Route": "29",
                "Direction": "N",
                "Incident": "Mechanical",
                "Location": "DUFFERIN STATION",
                "Min Delay": 10,
                "hour": 8,
            },
            {
                "ts": "2024-01-02T08:00:00",
                "mode": "bus",
                "Route": "29",
                "Direction": "N",
                "Incident": "Mechanical",
                "Location": "DUFFERIN STATION",
                "Min Delay": 20,
                "hour": 8,
            },
            {
                "ts": "2024-01-03T09:00:00",
                "mode": "bus",
                "Route": "29",
                "Direction": "S",
                "Incident": "Operations",
                "Location": "DUFFERIN STATION",
                "Min Delay": 65,
                "hour": 9,
            },
        ]
    ).to_csv(path, index=False)
    return path


@pytest.fixture()
def client(fake_artifact_path, fake_historical_path, monkeypatch):
    monkeypatch.setenv("TTC_MODEL_ARTIFACT_PATH", str(fake_artifact_path))
    monkeypatch.setenv("TTC_HISTORICAL_FEATURE_DATA_PATH", str(fake_historical_path))
    monkeypatch.setenv("TTC_GTFS_ZIP_PATH", str(fake_artifact_path.parent / "missing_gtfs.zip"))
    app_module = importlib.import_module("src.api.app")
    app_module.prediction_service = app_module.CalibratedDelayPredictionService()
    app_module.load_route_metadata_index.cache_clear()
    app_module.load_route_stop_index.cache_clear()
    return TestClient(app_module.app)


@pytest.fixture()
def fake_gtfs_path(tmp_path):
    path = tmp_path / "fake_ttc_gtfs.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "routes.txt",
            "\n".join(
                [
                    "route_id,route_short_name,route_long_name,route_type",
                    "r1,1,Yonge-University Line,1",
                    "r4,4,Sheppard Line,3",
                    "r6,6,Finch West Line,0",
                    "r29,29,Dufferin,3",
                    "r501,501,Queen,0",
                ]
            ),
        )
        archive.writestr(
            "trips.txt",
            "\n".join(
                [
                    "route_id,service_id,trip_id",
                    "r1,weekday,t1",
                    "r4,weekday,t4",
                    "r6,weekday,t6",
                    "r29,weekday,t29",
                    "r29,weekday,t29s",
                    "r501,weekday,t501",
                    "r501,weekday,t501w",
                ]
            ),
        )
        archive.writestr(
            "stop_times.txt",
            "\n".join(
                [
                    "trip_id,arrival_time,departure_time,stop_id,stop_sequence",
                    "t1,08:00:00,08:00:00,s_subway,1",
                    "t4,08:00:00,08:00:00,s_sheppard,1",
                    "t6,08:00:00,08:00:00,s_finch,1",
                    "t29,08:00:00,08:00:00,s_dufferin_station,1",
                    "t29,08:02:00,08:02:00,s_dufferin_wilson,2",
                    "t29s,08:00:00,08:00:00,s_dufferin_wilson,1",
                    "t29s,08:02:00,08:02:00,s_dufferin_station,2",
                    "t501,08:00:00,08:00:00,s_queen_spadina,1",
                    "t501,08:02:00,08:02:00,s_queen_yonge,2",
                    "t501w,08:00:00,08:00:00,s_queen_yonge,1",
                    "t501w,08:02:00,08:02:00,s_queen_spadina,2",
                ]
            ),
        )
        archive.writestr(
            "stops.txt",
            "\n".join(
                [
                    "stop_id,stop_name,stop_lat,stop_lon",
                    "s_subway,Union Station,43.645,-79.380",
                    "s_sheppard,Sheppard-Yonge Station,43.761,-79.410",
                    "s_finch,Finch West Station,43.765,-79.491",
                    "s_dufferin_station,Dufferin Station,43.660,-79.435",
                    "s_dufferin_wilson,Dufferin St at Wilson Ave,43.730,-79.435",
                    "s_queen_spadina,Queen St West at Spadina Ave,43.648,-79.397",
                    "s_queen_yonge,Queen St East at Yonge St,43.653,-79.380",
                ]
            ),
        )
    return path


@pytest.fixture()
def client_with_gtfs(fake_artifact_path, fake_historical_path, fake_gtfs_path, monkeypatch):
    monkeypatch.setenv("TTC_MODEL_ARTIFACT_PATH", str(fake_artifact_path))
    monkeypatch.setenv("TTC_HISTORICAL_FEATURE_DATA_PATH", str(fake_historical_path))
    monkeypatch.setenv("TTC_GTFS_ZIP_PATH", str(fake_gtfs_path))
    app_module = importlib.import_module("src.api.app")
    app_module.prediction_service = app_module.CalibratedDelayPredictionService()
    app_module.load_route_metadata_index.cache_clear()
    app_module.load_route_stop_index.cache_clear()
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
        "prior_route_incident_mean_delay": 10.0,
        "prior_mode_incident_mean_delay": 10.0,
        "prior_route_direction_mean_delay": 10.0,
        "prior_route_incident_count": 2,
        "prior_route_30d_mean_delay": 10.0,
        "prior_incident_30d_mean_delay": 10.0,
        "prior_route_30d_severe_rate_30": 0.1,
        "prior_incident_30d_severe_rate_30": 0.1,
        "prior_route_30d_severe_rate_60": 0.0,
        "prior_incident_30d_severe_rate_60": 0.0,
        "prior_location_mean_delay": 10.0,
        "prior_location_count": 2,
    }


def option_values(options: list[dict[str, str]]) -> list[str]:
    return [option["value"] for option in options]


def test_root_serves_demo_frontend(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Incident-time TTC delay prediction demo" in response.text
    assert "Model status" not in response.text
    assert "Calibrated two-output service" not in response.text
    assert 'data-preset="' not in response.text
    assert response.text.count('data-mode="') == 2
    assert 'data-mode="bus"' in response.text
    assert 'data-mode="streetcar"' in response.text


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


def test_demo_static_payload_fields_do_not_include_leakage_fields():
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
    assert "computes local historical features" in " ".join(body["notes_limitations"])


def test_model_options_returns_expected_structure(client):
    response = client.get("/model-options")

    assert response.status_code == 200
    body = response.json()
    assert body["modes"] == [
        {"value": "bus", "label": "Bus"},
        {"value": "streetcar", "label": "Streetcar"},
    ]
    assert body["routes"] == ["29", "501", "32A", "RAD"]
    assert option_values(body["directions"]) == ["N", "E", "S", "W", "B", "Unknown"]
    assert option_values(body["incidents"]) == CURATED_INCIDENT_VALUES
    assert "501" not in option_values(body["incidents"])
    assert body["locations"] == []
    assert body["counts"]["locations"] == 0
    assert body["counts"]["known_locations_server_side"] == 2
    assert "Mechanical delay at station" not in option_values(body["directions"])
    assert any("Excluded 1 non-route-like Route" in warning for warning in body["warnings"])


def test_model_options_directions_are_fixed_even_with_polluted_artifact(client):
    response = client.get("/model-options")

    assert response.status_code == 200
    body = response.json()
    assert option_values(body["directions"]) == ["N", "E", "S", "W", "B", "Unknown"]
    assert "Mechanical delay at station" not in option_values(body["directions"])


def test_model_options_route_options_are_route_like_only(client):
    response = client.get("/model-options")

    assert response.status_code == 200
    routes = response.json()["routes"]
    assert routes == ["29", "501", "32A", "RAD"]
    assert all(route.isalnum() and len(route) <= 6 for route in routes)


def test_model_options_returns_curated_incident_options_only(client):
    response = client.get("/model-options")

    assert response.status_code == 200
    incident_values = option_values(response.json()["incidents"])
    assert incident_values == CURATED_INCIDENT_VALUES
    assert "501" not in incident_values
    assert "Mechanical delay at station" not in incident_values


def test_route_options_fall_back_to_model_routes_without_gtfs(client):
    response = client.get("/route-options")

    assert response.status_code == 200
    body = response.json()
    assert body["gtfs_available"] is False
    assert body["routes"] == [
        {"value": "29", "label": "29", "mode": None},
        {"value": "501", "label": "501", "mode": None},
        {"value": "32A", "label": "32A", "mode": None},
        {"value": "RAD", "label": "RAD", "mode": None},
    ]
    assert "GTFS route-stop data is not configured" in body["warning"]


def test_route_options_use_gtfs_modes_when_available(client_with_gtfs):
    response = client_with_gtfs.get("/route-options")

    assert response.status_code == 200
    body = response.json()
    assert body["gtfs_available"] is True
    assert body["routes"] == [
        {"value": "29", "label": "29 - Dufferin", "mode": "bus"},
        {"value": "501", "label": "501 - Queen", "mode": "streetcar"},
    ]
    route_values = {route["value"] for route in body["routes"]}
    assert {"1", "4", "6"}.isdisjoint(route_values)


def test_route_locations_are_scoped_to_selected_route(client_with_gtfs):
    response = client_with_gtfs.get("/route-locations", params={"route": "29"})

    assert response.status_code == 200
    body = response.json()
    assert body["gtfs_available"] is True
    assert body["normalized_route"] == "29"
    assert body["mode"] == "bus"
    assert body["directions"] == [
        {"value": "N", "label": "North"},
        {"value": "S", "label": "South"},
        {"value": "B", "label": "Both / bidirectional"},
    ]
    assert body["count"] == 2
    assert sorted(location["value"] for location in body["locations"]) == [
        "DUFFERIN STATION",
        "DUFFERIN STREET AT WILSON AVENUE",
    ]


def test_route_locations_support_base_route_for_branch(client_with_gtfs):
    response = client_with_gtfs.get("/route-locations", params={"route": "29A"})

    assert response.status_code == 200
    body = response.json()
    assert body["normalized_route"] == "29"
    assert body["count"] == 2
    assert "Using base route 29 stop list" in body["warning"]


def test_route_locations_return_east_west_directions_for_streetcar_route(client_with_gtfs):
    response = client_with_gtfs.get("/route-locations", params={"route": "501"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "streetcar"
    assert body["directions"] == [
        {"value": "E", "label": "East"},
        {"value": "W", "label": "West"},
        {"value": "B", "label": "Both / bidirectional"},
    ]


def test_validate_route_location_accepts_stop_on_selected_route(client_with_gtfs):
    response = client_with_gtfs.post(
        "/validate-route-location",
        json={"route": "29", "location": "Dufferin Station"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["accepted_for_prediction"] is True
    assert body["normalized_route"] == "29"
    assert body["route_location"] == "DUFFERIN STATION"
    assert body["warning"] is None


def test_validate_route_location_rejects_stop_not_on_selected_route(client_with_gtfs):
    response = client_with_gtfs.post(
        "/validate-route-location",
        json={"route": "29", "location": "Queen St East at Yonge St"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["accepted_for_prediction"] is False
    assert body["normalized_route"] == "29"
    assert body["route_location"] is None
    assert "is not a stop on route 29" in body["warning"]


def test_match_location_handles_exact_match(client):
    response = client.post("/match-location", json={"location": "dufferin station"})

    assert response.status_code == 200
    body = response.json()
    assert body["original_location"] == "dufferin station"
    assert body["normalized_location"] == "DUFFERIN STATION"
    assert body["matched_location"] == "DUFFERIN STATION"
    assert body["score"] == 100.0
    assert body["match_type"] == "exact"
    assert body["accepted_for_prediction"] is True
    assert body["warning"] is None


def test_match_location_handles_fuzzy_or_contains_match(client):
    response = client.post("/match-location", json={"location": "queen and spadina"})

    assert response.status_code == 200
    body = response.json()
    assert body["matched_location"] == "QUEEN STREET WEST AND SPADINA AVENUE"
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


def test_match_location_malformed_request_returns_422_json(client):
    response = client.post("/match-location", json={"bad": "request"})

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert "detail" in response.json()


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


def test_predict_delay_works_with_basic_fields_and_timestamp(client):
    payload = {
        "mode": "bus",
        "Route": "29",
        "Direction": "N",
        "Incident": "Mechanical",
        "Location": "Dufferin Station",
        "timestamp": "2024-01-04T08:30:00",
    }

    response = client.post("/predict-delay", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_delay_minutes"] == 12.5
    assert any("Historical features were computed from prior local records" in warning for warning in body["warnings"])
    app_module = importlib.import_module("src.api.app")
    scored = app_module.prediction_service.artifact["expected_delay_regressor"].seen_frame.iloc[0]
    assert scored["prior_route_mean_delay"] == pytest.approx(95 / 3)
    assert scored["prior_route_incident_mean_delay"] == pytest.approx(15)
    assert scored["prior_route_incident_count"] == 2
    assert scored["prior_location_count"] == 3


def test_caller_provided_historical_override_is_respected(client):
    payload = {
        "mode": "bus",
        "Route": "29",
        "Direction": "N",
        "Incident": "Mechanical",
        "Location": "Dufferin Station",
        "timestamp": "2024-01-04T08:30:00",
        "prior_route_mean_delay": 99.0,
    }

    response = client.post("/predict-delay", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert any(
        "Using caller-provided historical feature override: prior_route_mean_delay" in warning
        for warning in body["warnings"]
    )
    app_module = importlib.import_module("src.api.app")
    scored = app_module.prediction_service.artifact["expected_delay_regressor"].seen_frame.iloc[0]
    assert scored["prior_route_mean_delay"] == 99.0
    assert scored["prior_route_incident_count"] == 2


def test_predict_delay_uses_normalized_categorical_values(client):
    payload = valid_payload()
    payload["Route"] = 29.0
    payload["Direction"] = "N/B"
    payload["Incident"] = "Mech"
    payload["Location"] = "Dufferin Stn"

    response = client.post("/predict-delay", json=payload)

    assert response.status_code == 200
    app_module = importlib.import_module("src.api.app")
    seen = app_module.prediction_service.artifact["expected_delay_regressor"].seen_frame
    assert seen.loc[0, "Route"] == "29"
    assert seen.loc[0, "Direction"] == "N"
    assert seen.loc[0, "Incident"] == "Mechanical"
    assert seen.loc[0, "Location"] == "DUFFERIN STATION"


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
    payload["timestamp"] = "2023-12-01T08:30:00"

    response = client.post("/predict-delay", json=payload)

    assert response.status_code == 200
    assert any("had no prior support" in warning for warning in response.json()["warnings"])


def test_historical_lookup_info_works(client, fake_historical_path):
    response = client.get("/historical-lookup-info")

    assert response.status_code == 200
    body = response.json()
    assert body["historical_data_path"] == str(fake_historical_path)
    assert body["row_count"] == 3
    assert body["available_historical_feature_names"] == HISTORICAL_FEATURES
    assert body["min_timestamp"].startswith("2024-01-01")
    assert body["max_timestamp"].startswith("2024-01-03")


def test_compute_historical_features_endpoint_works(client):
    response = client.post(
        "/compute-historical-features",
        json={
            "mode": "bus",
            "Route": "29",
            "Direction": "N",
            "Incident": "Mechanical",
            "Location": "Dufferin Station",
            "timestamp": "2024-01-04T08:30:00",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["computed_historical_features"]["prior_route_mean_delay"] == pytest.approx(95 / 3)
    assert body["computed_historical_features"]["prior_route_incident_count"] == 2
    assert body["normalized_input_values"]["Location"] == "DUFFERIN STATION"
    assert body["support_counts"]["prior_route_incident_count"] == 2


def test_unknown_categorical_values_warn_without_crashing(client):
    payload = valid_payload()
    payload["Route"] = "999"
    payload["Incident"] = "Unlisted"
    payload["Location"] = "Unknown stop"

    response = client.post("/predict-delay", json=payload)

    assert response.status_code == 200
    warning_text = " ".join(response.json()["warnings"])
    assert "Route '999'" in warning_text
    assert "Incident 'Other'" in warning_text
    assert "Location 'UNKNOWN STOP'" in warning_text


def test_app_import_is_safe(monkeypatch):
    monkeypatch.delenv("TTC_MODEL_ARTIFACT_PATH", raising=False)

    app_module = importlib.reload(importlib.import_module("src.api.app"))

    assert app_module.app is not None
    assert not app_module.prediction_service.is_loaded
