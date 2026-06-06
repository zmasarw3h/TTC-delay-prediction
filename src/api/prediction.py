"""Prediction service helpers for the calibrated TTC delay model artifact."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.api.feature_derivation import TIME_FEATURE_FIELDS, derive_time_features
from src.api.historical_lookup import HistoricalFeatureLookup
from src.api.input_validation import validate_model_features
from src.features.build_features import HISTORICAL_FEATURES


DEFAULT_MODEL_ARTIFACT_PATH = Path("artifacts/calibration/calibrated_two_output_model.joblib")
API_LIMITATION_NOTES = [
    "The API accepts basic incident details and computes local historical features when timestamp is provided.",
    "Historical prior-delay features are computed from records before the prediction moment.",
    "Weather enrichment is not implemented in this service.",
    "Missing historical features may still be imputed by the model pipeline, but reliability may be reduced.",
]


@dataclass(frozen=True)
class PredictionResult:
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


class CalibratedDelayPredictionService:
    """Load and score the calibrated two-output model artifact."""

    def __init__(
        self,
        artifact_path: str | Path | None = None,
        historical_lookup: HistoricalFeatureLookup | None = None,
    ) -> None:
        self.artifact_path = Path(
            artifact_path
            or os.environ.get("TTC_MODEL_ARTIFACT_PATH")
            or DEFAULT_MODEL_ARTIFACT_PATH
        )
        self.historical_lookup = historical_lookup or HistoricalFeatureLookup()
        self._artifact: dict[str, Any] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._artifact is not None

    @property
    def artifact(self) -> dict[str, Any]:
        if self._artifact is None:
            self._artifact = self._load_artifact()
        return self._artifact

    def _load_artifact(self) -> dict[str, Any]:
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                "Model artifact not found. Set TTC_MODEL_ARTIFACT_PATH or create "
                f"{self.artifact_path}."
            )
        artifact = joblib.load(self.artifact_path)
        if not isinstance(artifact, dict):
            raise ValueError("Model artifact must be a dictionary.")

        required_keys = {
            "expected_delay_regressor",
            "calibrated_risk_classifiers",
            "feature_columns",
            "target_column",
        }
        missing = sorted(required_keys - set(artifact.keys()))
        if missing:
            raise ValueError(f"Model artifact is missing required key(s): {missing}.")
        return artifact

    def model_info(self) -> dict[str, Any]:
        artifact = self.artifact
        metadata = artifact.get("metadata", {})
        notes = list(metadata.get("notes", [])) if isinstance(metadata, dict) else []
        notes.extend(API_LIMITATION_NOTES)
        thresholds = self.risk_thresholds

        return {
            "model_name": self.model_name,
            "model_phase": self.model_phase,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "risk_thresholds": thresholds,
            "selected_calibration_methods": artifact.get("selected_calibration_methods", {}),
            "selected_operating_cutoffs": artifact.get("selected_probability_thresholds", {}),
            "risk_band_definitions": artifact.get("risk_band_definitions", {}),
            "notes_limitations": notes,
        }

    @property
    def model_name(self) -> str:
        return str(self.artifact.get("model_name", "calibrated_two_output_delay_and_risk_model"))

    @property
    def model_phase(self) -> str:
        return str(self.artifact.get("model_phase", "Phase 7C"))

    @property
    def feature_columns(self) -> list[str]:
        columns = self.artifact.get("feature_columns", [])
        return [str(column) for column in columns]

    @property
    def target_column(self) -> str:
        return str(self.artifact.get("target_column", "Min Delay"))

    @property
    def risk_thresholds(self) -> list[int]:
        classifiers = self.artifact.get("calibrated_risk_classifiers", {})
        if classifiers:
            return sorted(int(threshold) for threshold in classifiers.keys())
        cutoffs = self.artifact.get("selected_probability_thresholds", {})
        return sorted(int(threshold) for threshold in cutoffs.keys())

    def predict(self, payload: Mapping[str, Any]) -> PredictionResult:
        payload_with_history, history_warnings = self._enrich_historical_features(payload)
        payload_with_time, warnings = _derive_time_fields(payload_with_history, self.feature_columns)
        warnings = history_warnings + warnings
        validation = validate_model_features(
            payload_with_time,
            self.feature_columns,
            known_categories=self.known_categories,
        )
        warnings.extend(validation.warnings)
        warnings.extend(_missing_historical_feature_warnings(validation.features))

        frame = pd.DataFrame([validation.features], columns=self.feature_columns)
        artifact = self.artifact
        delay_prediction = artifact["expected_delay_regressor"].predict(frame)
        predicted_delay = float(np.asarray(delay_prediction, dtype="float64").ravel()[0])

        probability_30 = self._predict_threshold_probability(30, frame)
        cutoff_30 = self._selected_cutoff(30)
        probability_60 = self._predict_threshold_probability(60, frame)
        cutoff_60 = self._selected_cutoff(60)

        return PredictionResult(
            predicted_delay_minutes=predicted_delay,
            calibrated_severe_delay_probability_30=probability_30,
            risk_band_30=_assign_risk_band(probability_30),
            severe_delay_prediction_30=int(probability_30 >= cutoff_30),
            selected_probability_cutoff_30=cutoff_30,
            calibrated_severe_delay_probability_60=probability_60,
            risk_band_60=_assign_risk_band(probability_60),
            severe_delay_prediction_60=int(probability_60 >= cutoff_60),
            selected_probability_cutoff_60=cutoff_60,
            warnings=warnings,
            model_name=self.model_name,
            model_phase=self.model_phase,
        )

    def compute_historical_features(self, payload: Mapping[str, Any]):
        return self.historical_lookup.compute(payload)

    def historical_lookup_info(self) -> dict[str, Any]:
        return self.historical_lookup.info()

    def _predict_threshold_probability(self, threshold: int, frame: pd.DataFrame) -> float:
        classifiers = self.artifact.get("calibrated_risk_classifiers", {})
        classifier = classifiers.get(threshold) or classifiers.get(str(threshold))
        if classifier is None:
            classifier = self.artifact.get(f"risk_classifier_{threshold}")
        if classifier is None:
            raise ValueError(f"Model artifact has no risk classifier for {threshold}+ minutes.")
        return _predict_positive_probability(classifier, frame)

    def _selected_cutoff(self, threshold: int) -> float:
        cutoffs = self.artifact.get("selected_probability_thresholds", {})
        cutoff_info = cutoffs.get(str(threshold)) or cutoffs.get(threshold) or {}
        return float(cutoff_info.get("probability_cutoff", 0.5))

    @property
    def known_categories(self) -> dict[str, list[str]] | None:
        explicit_categories = self.artifact.get("known_categories")
        if isinstance(explicit_categories, dict):
            return {
                str(column): [str(value) for value in values]
                for column, values in explicit_categories.items()
            }
        return _extract_known_categories(self.artifact.get("expected_delay_regressor"))

    def _enrich_historical_features(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
        enriched = dict(payload)
        feature_columns = set(self.feature_columns)
        model_historical_features = [
            feature for feature in HISTORICAL_FEATURES if feature in feature_columns
        ]
        missing_historical = [
            feature for feature in model_historical_features if enriched.get(feature) is None
        ]
        supplied_historical = [
            feature for feature in model_historical_features if enriched.get(feature) is not None
        ]
        warnings = [
            f"Using caller-provided historical feature override: {feature}."
            for feature in supplied_historical
        ]

        if not missing_historical:
            return enriched, warnings

        if enriched.get("timestamp") is None:
            required_time_features = [field for field in TIME_FEATURE_FIELDS if field in feature_columns]
            missing_time = [field for field in required_time_features if enriched.get(field) is None]
            detail = " Missing time feature(s): " + ", ".join(missing_time) + "." if missing_time else ""
            raise ValueError(
                "timestamp is required when historical or time features are missing. "
                "Provide timestamp, or provide all required time and historical model features manually."
                + detail
            )

        lookup_result = self.historical_lookup.compute(enriched)
        for feature in missing_historical:
            enriched[feature] = lookup_result.features.get(feature)
        warnings.extend(lookup_result.warnings)
        return enriched, warnings


def _derive_time_fields(
    payload: Mapping[str, Any],
    feature_columns: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    enriched = dict(payload)
    warnings: list[str] = []
    required_time_fields = TIME_FEATURE_FIELDS
    if feature_columns is not None:
        required_time_fields = [field for field in TIME_FEATURE_FIELDS if field in set(feature_columns)]
    timestamp = enriched.get("timestamp")
    if timestamp is not None:
        derived = derive_time_features(timestamp)
        missing_time_fields = [field for field in required_time_fields if enriched.get(field) is None]
        for field in missing_time_fields:
            enriched[field] = derived[field]
        if missing_time_fields:
            warnings.append("Derived missing time fields from timestamp.")

        if enriched.get("is_holiday") is None:
            enriched["is_holiday"] = derived["is_holiday"]
            warnings.append("Derived is_holiday from timestamp.")
        else:
            warnings.append("is_holiday was provided by caller and was not overwritten.")
    else:
        missing_time_fields = [field for field in required_time_fields if enriched.get(field) is None]
        if missing_time_fields:
            raise ValueError(
                "timestamp is required when time features are missing. "
                "Provide timestamp, or provide all required time model features manually."
            )
        if enriched.get("is_holiday") is None:
            enriched["is_holiday"] = 0
            warnings.append("is_holiday was missing and set to 0 because no timestamp was provided.")
    return enriched, warnings


def _missing_historical_feature_warnings(features: Mapping[str, Any]) -> list[str]:
    missing = [field for field in HISTORICAL_FEATURES if features.get(field) is None]
    if not missing:
        return []
    return [
        "Missing historical prior-delay feature(s) may reduce prediction reliability: "
        + ", ".join(missing)
        + "."
    ]


def _predict_positive_probability(model: Any, frame: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)
        arr = np.asarray(probabilities, dtype="float64")
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return float(np.clip(arr[0, 1], 0.0, 1.0))
        return float(np.clip(arr.ravel()[0], 0.0, 1.0))
    prediction = model.predict(frame)
    return float(np.clip(np.asarray(prediction, dtype="float64").ravel()[0], 0.0, 1.0))


def _extract_known_categories(model: Any) -> dict[str, list[str]] | None:
    pipeline = getattr(model, "pipeline", model)
    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor")
    transformers = getattr(preprocessor, "transformers_", None)
    if transformers is None:
        return None

    for _, transformer, columns in transformers:
        steps = getattr(transformer, "named_steps", {})
        onehot = steps.get("onehot")
        categories = getattr(onehot, "categories_", None)
        if categories is None:
            continue
        return {
            str(column): [str(value) for value in category_values]
            for column, category_values in zip(columns, categories)
        }
    return None


def _assign_risk_band(probability: float) -> str:
    if probability < 0.10:
        return "low"
    if probability < 0.30:
        return "medium"
    return "high"
