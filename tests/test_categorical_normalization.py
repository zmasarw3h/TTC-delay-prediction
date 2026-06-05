from __future__ import annotations

import importlib

import pandas as pd

from src.data.categorical_normalization import (
    UNKNOWN_CATEGORY,
    normalize_categorical_columns,
    normalize_direction,
    normalize_incident,
    normalize_location,
    normalize_route,
)


def test_direction_normalization_maps_common_variants():
    for value in ["n", "NB", "N/B", "north"]:
        assert normalize_direction(value) == "N"
    for value in ["E/B", "eb", "east"]:
        assert normalize_direction(value) == "E"
    for value in ["B/W", "Bothways", "W/B/E/B"]:
        assert normalize_direction(value) == "B"


def test_direction_normalization_maps_garbage_to_unknown():
    for value in ["Mechanical", "Jane", "mins late", "INC 7603241", None]:
        assert normalize_direction(value) == UNKNOWN_CATEGORY


def test_route_normalization_preserves_route_like_values():
    assert normalize_route(29) == "29"
    assert normalize_route(29.0) == "29"
    assert normalize_route("32A") == "32A"
    assert normalize_route("504B") == "504B"
    assert normalize_route("RAD") == "RAD"


def test_route_normalization_filters_long_location_like_values():
    assert normalize_route("Queen Street West and Spadina") == UNKNOWN_CATEGORY
    assert normalize_route("Jane") == UNKNOWN_CATEGORY


def test_incident_normalization_maps_variants_to_curated_categories():
    assert normalize_incident("Mech") == "Mechanical"
    assert normalize_incident("Ops") == "Operations"
    assert normalize_incident("Securitty") == "Security"
    assert normalize_incident(None) == UNKNOWN_CATEGORY
    assert normalize_incident("Unlisted rare label") == "Other"


def test_location_normalization_applies_safe_deterministic_cleanup():
    assert normalize_location("queen & spadina") == "QUEEN AND SPADINA"
    assert normalize_location("Kennedy Stn") == "KENNEDY STATION"
    assert normalize_location("  Queen    St   W  ") == "QUEEN STREET WEST"
    assert normalize_location(None) == UNKNOWN_CATEGORY


def test_dataframe_normalization_preserves_raw_columns_and_updates_modeling_columns():
    frame = pd.DataFrame(
        {
            "mode": ["Bus"],
            "Route": [29.0],
            "Direction": ["N/B"],
            "Incident": ["Mech"],
            "Location": ["Kennedy Stn"],
        }
    )

    normalized = normalize_categorical_columns(frame)

    assert normalized.loc[0, "mode"] == "bus"
    assert normalized.loc[0, "Route"] == "29"
    assert normalized.loc[0, "Direction"] == "N"
    assert normalized.loc[0, "Incident"] == "Mechanical"
    assert normalized.loc[0, "Location"] == "KENNEDY STATION"
    assert normalized.loc[0, "Route_raw"] == 29.0
    assert normalized.loc[0, "Direction_raw"] == "N/B"
    assert normalized.loc[0, "Incident_raw"] == "Mech"
    assert normalized.loc[0, "Location_raw"] == "Kennedy Stn"
    assert "Route_raw" not in frame.columns


def test_categorical_normalization_import_is_safe():
    module = importlib.import_module("src.data.categorical_normalization")

    assert module.UNKNOWN_CATEGORY == "Unknown"
