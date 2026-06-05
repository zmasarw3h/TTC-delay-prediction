import pytest

from src.api.input_validation import (
    UNKNOWN_CATEGORY,
    normalize_mode,
    normalize_route,
    validate_model_features,
)


REQUIRED_FEATURES = [
    "hour",
    "mode",
    "Route",
    "Direction",
    "Incident",
    "Location",
    "prior_route_mean_delay",
]


def test_missing_route_becomes_unknown():
    result = validate_model_features(
        {"mode": "bus", "Route": "", "Direction": "N"},
        REQUIRED_FEATURES,
    )

    assert result.features["Route"] == UNKNOWN_CATEGORY


def test_rad_route_is_preserved():
    assert normalize_route("RAD") == "RAD"


def test_numeric_route_becomes_string_category():
    assert normalize_route(29) == "29"
    assert normalize_route(29.0) == "29"


def test_missing_direction_becomes_unknown():
    result = validate_model_features(
        {"mode": "bus", "Route": "29", "Direction": None},
        REQUIRED_FEATURES,
    )

    assert result.features["Direction"] == UNKNOWN_CATEGORY


def test_mode_normalization_accepts_bus_and_streetcar():
    assert normalize_mode("Bus") == "bus"
    assert normalize_mode("streetcar") == "streetcar"


def test_unsupported_mode_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported mode"):
        normalize_mode("subway")


def test_leakage_fields_are_rejected():
    with pytest.raises(ValueError, match="leakage-sensitive"):
        validate_model_features(
            {"mode": "bus", "Route": "29", "Direction": "N", "Min Delay": 10},
            REQUIRED_FEATURES,
        )


def test_missing_numeric_historical_features_remain_none():
    result = validate_model_features(
        {"mode": "bus", "Route": "29", "Direction": "N"},
        REQUIRED_FEATURES,
    )

    assert result.features["prior_route_mean_delay"] is None


def test_unknown_categorical_values_warn_without_rejection():
    result = validate_model_features(
        {
            "mode": "bus",
            "Route": "29",
            "Direction": "Q",
            "Incident": "Unlisted",
            "Location": "Queen / Unknown",
        },
        REQUIRED_FEATURES,
        known_categories={"Incident": {"Delay", "Mechanical"}},
    )

    assert result.features["Direction"] == "Q"
    assert result.features["Incident"] == "Unlisted"
    assert any("Direction 'Q'" in warning for warning in result.warnings)
    assert any("Incident 'Unlisted'" in warning for warning in result.warnings)
