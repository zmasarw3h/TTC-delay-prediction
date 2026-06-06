"""Pydantic schemas for the TTC delay prediction API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.api.input_validation import LEAKAGE_FIELDS, normalize_mode


class EngineeredIncidentFeatures(BaseModel):
    """Engineered incident-time features expected by the calibrated model."""

    model_config = ConfigDict(extra="allow")

    mode: str | None = None
    Route: str | int | float | None = None
    Direction: str | None = None
    Incident: str | None = None
    Location: str | None = None
    timestamp: str | None = None

    hour: int | float | None = None
    day_of_week: int | float | None = None
    month: int | float | None = None
    is_weekend: int | float | None = None
    is_holiday: int | float | None = None
    hour_sin: float | None = None
    hour_cos: float | None = None
    day_of_year: int | float | None = None
    day_sin: float | None = None
    day_cos: float | None = None

    prior_route_mean_delay: float | None = None
    prior_route_hour_mean_delay: float | None = None
    prior_incident_mean_delay: float | None = None
    prior_mode_mean_delay: float | None = None
    prior_global_mean_delay: float | None = None
    prior_route_hour_7d_mean_delay: float | None = None
    prior_route_incident_mean_delay: float | None = None
    prior_mode_incident_mean_delay: float | None = None
    prior_route_direction_mean_delay: float | None = None
    prior_route_incident_count: float | None = None
    prior_route_30d_mean_delay: float | None = None
    prior_incident_30d_mean_delay: float | None = None
    prior_route_30d_severe_rate_30: float | None = None
    prior_incident_30d_severe_rate_30: float | None = None
    prior_route_30d_severe_rate_60: float | None = None
    prior_incident_30d_severe_rate_60: float | None = None
    prior_location_mean_delay: float | None = None
    prior_location_count: float | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_leakage_and_unsupported_mode(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        leakage_fields = LEAKAGE_FIELDS.intersection(data.keys())
        if leakage_fields:
            names = ", ".join(sorted(leakage_fields))
            raise ValueError(f"Payload includes leakage-sensitive field(s): {names}.")

        if "mode" in data:
            normalize_mode(data.get("mode"))
        return data


class HealthResponse(BaseModel):
    status: str
    model_artifact_loaded: bool
    artifact_exists: bool
    artifact_path: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_phase: str
    feature_columns: list[str]
    target_column: str
    risk_thresholds: list[int]
    selected_calibration_methods: dict[str, Any]
    selected_operating_cutoffs: dict[str, Any]
    risk_band_definitions: dict[str, Any]
    notes_limitations: list[str] = Field(default_factory=list)


class ModelOptionsResponse(BaseModel):
    modes: list[dict[str, str]]
    routes: list[str]
    directions: list[dict[str, str]]
    incidents: list[dict[str, str]]
    locations: list[str]
    warnings: list[str]
    counts: dict[str, int]


class LocationMatchRequest(BaseModel):
    location: str


class RouteOptionsResponse(BaseModel):
    routes: list[dict[str, Any]]
    gtfs_available: bool
    source_path: str | None = None
    warning: str | None = None


class RouteLocationsResponse(BaseModel):
    route: str
    normalized_route: str
    mode: str | None = None
    directions: list[dict[str, str]]
    locations: list[dict[str, str]]
    count: int
    gtfs_available: bool
    source_path: str | None = None
    warning: str | None = None


class RouteLocationValidationRequest(BaseModel):
    route: str
    location: str


class RouteLocationValidationResponse(BaseModel):
    route: str
    normalized_route: str
    original_location: str
    normalized_location: str
    route_location: str | None
    route_location_label: str | None
    accepted_for_prediction: bool
    warning: str | None


class LocationMatchResponse(BaseModel):
    original_location: str
    normalized_location: str
    matched_location: str | None
    score: float
    match_type: str
    warning: str | None
    accepted_for_prediction: bool


class DelayPredictionResponse(BaseModel):
    predicted_delay_minutes: float
    calibrated_severe_delay_probability_30: float
    risk_band_30: str
    severe_delay_prediction_30: int
    selected_probability_cutoff_30: float
    calibrated_severe_delay_probability_60: float
    risk_band_60: str
    severe_delay_prediction_60: int
    selected_probability_cutoff_60: float
    warnings: list[str]
    model_name: str
    model_phase: str


class HistoricalLookupInfoResponse(BaseModel):
    historical_data_path: str
    loaded: bool
    loadable: bool
    row_count: int | None = None
    min_timestamp: str | None = None
    max_timestamp: str | None = None
    available_historical_feature_names: list[str]
    notes_limitations: list[str]
    warnings: list[str] = Field(default_factory=list)


class HistoricalFeatureComputationResponse(BaseModel):
    computed_historical_features: dict[str, float | int | None]
    warnings: list[str]
    normalized_input_values: dict[str, Any]
    support_counts: dict[str, int]
    metadata: dict[str, Any]
