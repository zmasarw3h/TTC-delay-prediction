from __future__ import annotations

import importlib

import pandas as pd

from src.data.audit_categories import audit_dataframe, audit_value, is_route_like


def test_direction_audit_flags_garbage_direction_values():
    result = audit_value("Direction", "Mechanical delay at station", 3, 30.0)

    assert result.suspicious is True
    assert "longer than 3 characters" in result.reasons
    assert "contains incident or location word" in result.reasons


def test_direction_audit_treats_normalized_values_as_healthy():
    for value in ["N", "E", "S", "W", "B", "Unknown"]:
        result = audit_value("Direction", value, 1, 10.0)

        assert result.suspicious is False


def test_incident_audit_treats_curated_normalized_categories_as_healthy():
    for value in [
        "Utilized Off Route",
        "General Delay",
        "Late Leaving Garage",
        "Road Blocked - NON-TTC Collision",
    ]:
        result = audit_value("Incident", value, 1, 10.0)

        assert result.suspicious is False


def test_incident_audit_still_flags_unknown_rare_malformed_labels():
    result = audit_value("Incident", "Garage route collision note typo", 1, 10.0)

    assert result.suspicious is True
    assert "extremely rare incident label" in result.reasons


def test_route_audit_flags_long_text_route_values():
    result = audit_value("Route", "Queen Street West and Spadina", 2, 20.0)

    assert result.suspicious is True
    assert "long route value" in result.reasons
    assert "looks like a location or intersection" in result.reasons


def test_incident_audit_flags_route_like_incident_values():
    result = audit_value("Incident", "501", 1, 10.0)

    assert result.suspicious is True
    assert "route-like incident value" in result.reasons


def test_route_like_filter_accepts_expected_route_values():
    assert is_route_like("29")
    assert is_route_like("501")
    assert is_route_like("32A")
    assert is_route_like("RAD")
    assert not is_route_like("Queen Street West and Spadina")


def test_audit_dataframe_returns_column_summaries():
    frame = pd.DataFrame(
        {
            "mode": ["bus", "streetcar"],
            "Route": ["29", "Mechanical delay at station"],
            "Direction": ["N", "late garage"],
            "Incident": ["Mechanical", "501"],
            "Location": ["Dufferin Station", "54"],
        }
    )

    audit = audit_dataframe(frame, top_n=2)

    assert audit["summary"]["Route"]["suspicious_values_count"] == 1
    assert audit["summary"]["Direction"]["suspicious_values_count"] == 1
    assert audit["summary"]["Incident"]["suspicious_values_count"] == 1
    assert audit["summary"]["Location"]["suspicious_values_count"] == 1


def test_audit_categories_import_is_safe():
    module = importlib.import_module("src.data.audit_categories")

    assert module.DEFAULT_INPUT.name == "modeling_dataset.csv"
