"""Input normalization helpers for future TTC delay prediction APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.features.build_features import MAIN_CATEGORICAL_FEATURES, NUMERIC_FEATURES


UNKNOWN_CATEGORY = "Unknown"
MISSING_STRINGS = {"", "nan", "none", "null"}
SUPPORTED_MODES = {"bus", "streetcar"}
COMMON_DIRECTIONS = {
    "N",
    "S",
    "E",
    "W",
    "B",
    "N/B",
    "S/B",
    "E/B",
    "W/B",
}
LEAKAGE_FIELDS = {
    "Min Delay",
    "Min Gap",
    "Vehicle",
    "source_file",
    "source_sheet",
    "severe_delay_15",
}


@dataclass(frozen=True)
class ValidationResult:
    """Normalized model features plus non-fatal validation warnings."""

    features: dict[str, Any]
    warnings: list[str]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_STRINGS
    return False


def normalize_categorical_string(
    value: Any,
    *,
    missing_value: str | None = UNKNOWN_CATEGORY,
) -> str | None:
    """Normalize a categorical value without changing meaningful category text."""
    if _is_missing(value):
        return missing_value
    return str(value).strip()


def normalize_route(value: Any) -> str:
    """Normalize TTC route values as string categories."""
    if _is_missing(value):
        return UNKNOWN_CATEGORY
    if isinstance(value, bool):
        raise ValueError("Route must be a route string or integer-like number.")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        raise ValueError("Route numeric values must be integer-like.")

    normalized = normalize_categorical_string(value)
    if normalized is None:
        return UNKNOWN_CATEGORY
    return normalized


def normalize_direction(value: Any) -> tuple[str, list[str]]:
    """Normalize direction while allowing unusual direction codes with a warning."""
    normalized = normalize_categorical_string(value)
    if normalized is None:
        normalized = UNKNOWN_CATEGORY

    warnings: list[str] = []
    if normalized == UNKNOWN_CATEGORY:
        warnings.append("Direction is missing and was set to Unknown.")
        return normalized, warnings

    normalized = normalized.upper()
    if normalized not in COMMON_DIRECTIONS:
        warnings.append(f"Direction '{normalized}' is not a common TTC direction code.")
    return normalized, warnings


def normalize_mode(value: Any) -> str:
    """Normalize and validate transit mode."""
    normalized = normalize_categorical_string(value, missing_value=None)
    if normalized is None:
        raise ValueError("mode is required and must be one of: bus, streetcar.")

    mode = normalized.lower()
    if mode not in SUPPORTED_MODES:
        raise ValueError("Unsupported mode. Expected one of: bus, streetcar.")
    return mode


def normalize_incident_or_location(value: Any) -> str:
    """Normalize incident and location text for model categorical features."""
    normalized = normalize_categorical_string(value)
    if normalized is None:
        return UNKNOWN_CATEGORY
    return normalized


def _normalize_numeric(value: Any) -> Any:
    if _is_missing(value):
        return None
    return value


def _warn_if_unknown_category(
    field_name: str,
    value: Any,
    known_categories: Mapping[str, Iterable[str]] | None,
) -> str | None:
    if known_categories is None or field_name not in known_categories:
        return None
    if value == UNKNOWN_CATEGORY:
        return None

    allowed = {str(category) for category in known_categories[field_name]}
    if str(value) not in allowed:
        return f"{field_name} '{value}' was not seen in the known category list."
    return None


def validate_model_features(
    payload: Mapping[str, Any],
    required_feature_columns: Iterable[str],
    *,
    categorical_columns: Iterable[str] = MAIN_CATEGORICAL_FEATURES,
    numeric_columns: Iterable[str] = NUMERIC_FEATURES,
    known_categories: Mapping[str, Iterable[str]] | None = None,
) -> ValidationResult:
    """Validate and normalize an engineered model feature payload.

    Missing categorical values become ``Unknown``. Missing numeric values remain
    ``None`` so the trained model pipeline can apply its numeric imputation.
    """
    leakage_fields = LEAKAGE_FIELDS.intersection(payload.keys())
    if leakage_fields:
        names = ", ".join(sorted(leakage_fields))
        raise ValueError(f"Payload includes leakage-sensitive field(s): {names}.")

    categorical_set = set(categorical_columns)
    numeric_set = set(numeric_columns)
    features: dict[str, Any] = {}
    warnings: list[str] = []

    for column in required_feature_columns:
        value = payload.get(column)

        if column == "mode":
            normalized = normalize_mode(value)
        elif column == "Route":
            normalized = normalize_route(value)
            if normalized == UNKNOWN_CATEGORY:
                warnings.append("Route is missing and was set to Unknown.")
        elif column == "Direction":
            normalized, direction_warnings = normalize_direction(value)
            warnings.extend(direction_warnings)
        elif column in {"Incident", "Location"}:
            normalized = normalize_incident_or_location(value)
            if normalized == UNKNOWN_CATEGORY:
                warnings.append(f"{column} is missing and was set to Unknown.")
        elif column in categorical_set:
            normalized = normalize_categorical_string(value)
            if normalized == UNKNOWN_CATEGORY:
                warnings.append(f"{column} is missing and was set to Unknown.")
        elif column in numeric_set:
            normalized = _normalize_numeric(value)
        else:
            normalized = _normalize_numeric(value)

        features[column] = normalized
        category_warning = _warn_if_unknown_category(column, normalized, known_categories)
        if category_warning is not None:
            warnings.append(category_warning)

    return ValidationResult(features=features, warnings=warnings)
