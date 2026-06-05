"""Deterministic categorical normalization for TTC modeling features.

The functions in this module are import-safe and do not use target values.
They are intended for training feature generation and API input validation.
"""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


UNKNOWN_CATEGORY = "Unknown"
OTHER_CATEGORY = "Other"
NORMALIZED_CATEGORICAL_COLUMNS = ["mode", "Route", "Direction", "Incident", "Location"]
RAW_CATEGORICAL_COLUMNS = ["Route", "Direction", "Incident", "Location"]
DIRECTION_CATEGORIES = ["N", "E", "S", "W", "B", UNKNOWN_CATEGORY]
INCIDENT_CATEGORIES = [
    "Mechanical",
    "Utilized Off Route",
    "General Delay",
    "Late Leaving Garage",
    "Investigation",
    "Operations",
    "Operations - Operator",
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
    UNKNOWN_CATEGORY,
    OTHER_CATEGORY,
]

NULL_LIKE_STRINGS = {
    "",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "<na>",
    "unknown",
    "unk",
}
ROUTE_PATTERN = re.compile(r"^(?:\d{1,4}[A-Z]{0,2}|RAD)$")
ROUTE_TEXT_BLOCKLIST = {
    "ACCIDENT",
    "AVENUE",
    "BLOCKED",
    "BOULEVARD",
    "COLLISION",
    "DELAY",
    "DIVERSION",
    "DRIVE",
    "EMERGENCY",
    "GARAGE",
    "INCIDENT",
    "INTERSECTION",
    "LOOP",
    "MECHANICAL",
    "OPERATIONS",
    "OPERATOR",
    "ROAD",
    "SECURITY",
    "STATION",
    "STREET",
    "TERMINAL",
}

INCIDENT_ALIASES = {
    "MECH": "Mechanical",
    "MECHANICAL": "Mechanical",
    "MECHANICAL DELAY": "Mechanical",
    "UTILIZED OFF ROUTE": "Utilized Off Route",
    "OFF ROUTE": "Utilized Off Route",
    "GENERAL": "General Delay",
    "GENERAL DELAY": "General Delay",
    "DELAY": "General Delay",
    "LATE LEAVING GARAGE": "Late Leaving Garage",
    "LATE GARAGE": "Late Leaving Garage",
    "INVESTIGATION": "Investigation",
    "INVESTIGATING": "Investigation",
    "OPS": "Operations",
    "OPERATION": "Operations",
    "OPERATIONS": "Operations",
    "OPERATOR": "Operations - Operator",
    "OPERATIONS OPERATOR": "Operations - Operator",
    "OPERATIONS - OPERATOR": "Operations - Operator",
    "DIVERSION": "Diversion",
    "DIVERSIONS": "Diversion",
    "EMERGENCY": "Emergency Services",
    "EMS": "Emergency Services",
    "EMERGENCY SERVICES": "Emergency Services",
    "SECURITTY": "Security",
    "SECURITY": "Security",
    "COLLISION": "Collision - TTC",
    "ACCIDENT": "Collision - TTC",
    "TTC COLLISION": "Collision - TTC",
    "COLLISION - TTC": "Collision - TTC",
    "TTC INVOLVED": "Collision - TTC Involved",
    "COLLISION - TTC INVOLVED": "Collision - TTC Involved",
    "ROAD BLOCKED": "Road Blocked - NON-TTC Collision",
    "ROAD BLOCKED - NON-TTC COLLISION": "Road Blocked - NON-TTC Collision",
    "NON TTC COLLISION": "Road Blocked - NON-TTC Collision",
    "NON-TTC COLLISION": "Road Blocked - NON-TTC Collision",
    "HELD": "Held By",
    "HELD BY": "Held By",
    "CLEANING": "Cleaning",
    "CLEAN-UP": "Cleaning",
    "CLEAN UP": "Cleaning",
    "UNSANITARY": "Cleaning - Unsanitary",
    "CLEANING - UNSANITARY": "Cleaning - Unsanitary",
    "VISION": "Vision",
    "OVERHEAD": "Overhead",
    "PANTOGRAPH": "Overhead - Pantograph",
    "OVERHEAD - PANTOGRAPH": "Overhead - Pantograph",
    "RAIL": "Rail/Switches",
    "SWITCH": "Rail/Switches",
    "SWITCHES": "Rail/Switches",
    "RAIL/SWITCHES": "Rail/Switches",
}

LOCATION_ABBREVIATIONS = {
    "STN": "STATION",
    "STA": "STATION",
    "ST": "STREET",
    "AVE": "AVENUE",
    "RD": "ROAD",
    "BLVD": "BOULEVARD",
    "DR": "DRIVE",
    "CRES": "CRESCENT",
    "PKWY": "PARKWAY",
    "HWY": "HIGHWAY",
    "W": "WEST",
    "E": "EAST",
    "N": "NORTH",
    "S": "SOUTH",
}


def normalize_mode(value: Any, *, strict: bool = False) -> str:
    """Normalize TTC mode to ``bus``, ``streetcar``, or ``Unknown``."""
    text = _clean_text(value)
    if text is None:
        if strict:
            raise ValueError("mode is required and must be one of: bus, streetcar.")
        return UNKNOWN_CATEGORY

    mode = text.lower()
    if mode in {"bus", "streetcar"}:
        return mode
    if strict:
        raise ValueError("Unsupported mode. Expected one of: bus, streetcar.")
    return UNKNOWN_CATEGORY


def normalize_route(value: Any) -> str:
    """Normalize route identifiers while filtering obvious non-route text."""
    if _is_missing(value) or isinstance(value, bool):
        return UNKNOWN_CATEGORY
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return UNKNOWN_CATEGORY
        return str(int(value)) if value.is_integer() else UNKNOWN_CATEGORY

    text = _clean_text(value)
    if text is None:
        return UNKNOWN_CATEGORY
    numeric = _integer_like_text(text)
    if numeric is not None:
        return numeric

    candidate = text.upper().replace(" ", "")
    if candidate == "RAD":
        return "RAD"
    if not ROUTE_PATTERN.fullmatch(candidate):
        return UNKNOWN_CATEGORY
    if len(candidate) > 6:
        return UNKNOWN_CATEGORY
    if any(word in text.upper().split() for word in ROUTE_TEXT_BLOCKLIST):
        return UNKNOWN_CATEGORY
    return candidate


def normalize_direction(value: Any) -> str:
    """Normalize direction to N, E, S, W, B, or Unknown."""
    text = _clean_text(value)
    if text is None:
        return UNKNOWN_CATEGORY

    upper = text.upper()
    compact = re.sub(r"[^A-Z]", "", upper)
    if compact in {"N", "NB", "NORTH", "NBOUND", "NORTHBOUND"}:
        return "N"
    if compact in {"E", "EB", "EAST", "EBOUND", "EASTBOUND"}:
        return "E"
    if compact in {"S", "SB", "SOUTH", "SBOUND", "SOUTHBOUND"}:
        return "S"
    if compact in {"W", "WB", "WEST", "WBOUND", "WESTBOUND"}:
        return "W"
    if compact in {
        "B",
        "BW",
        "BOTH",
        "BOTHWAYS",
        "BOTHWAY",
        "BIDIRECTIONAL",
        "BIDIR",
        "MIXED",
        "WBE",
        "WBEB",
        "EBWB",
        "NBSB",
        "SBNB",
    }:
        return "B"
    if set(compact).issubset({"N", "E", "S", "W", "B"}) and len(
        {letter for letter in compact if letter in {"N", "E", "S", "W"}}
    ) >= 2:
        return "B"
    return UNKNOWN_CATEGORY


def normalize_incident(value: Any) -> str:
    """Normalize incident labels to curated operational categories."""
    text = _clean_text(value)
    if text is None:
        return UNKNOWN_CATEGORY

    key = _canonical_key(text)
    if key in INCIDENT_ALIASES:
        return INCIDENT_ALIASES[key]

    for category in INCIDENT_CATEGORIES:
        if key == _canonical_key(category):
            return category

    if "PANTOGRAPH" in key:
        return "Overhead - Pantograph"
    if "OVERHEAD" in key:
        return "Overhead"
    if "UNSANITARY" in key:
        return "Cleaning - Unsanitary"
    if "CLEAN" in key:
        return "Cleaning"
    if "SECUR" in key:
        return "Security"
    if "EMERGENCY" in key or key == "EMS":
        return "Emergency Services"
    if "OPERATOR" in key:
        return "Operations - Operator"
    if "OPS" in key or "OPERATION" in key:
        return "Operations"
    if "MECH" in key:
        return "Mechanical"
    if "DIVERSION" in key:
        return "Diversion"
    if "COLLISION" in key or "ACCIDENT" in key:
        return "Collision - TTC"
    return OTHER_CATEGORY


def normalize_location(value: Any) -> str:
    """Apply safe deterministic text cleanup to location text."""
    text = _clean_text(value)
    if text is None:
        return UNKNOWN_CATEGORY

    upper = text.upper()
    upper = re.sub(r"[&@/]", " AND ", upper)
    upper = re.sub(r"[.,;:()\\[\\]{}]", " ", upper)
    tokens = [LOCATION_ABBREVIATIONS.get(token, token) for token in upper.split()]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip() or UNKNOWN_CATEGORY


def normalize_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized modeling categorical columns.

    ``Route_raw``, ``Direction_raw``, ``Incident_raw``, and ``Location_raw`` are
    created when the source column exists and the raw column is not already
    present. The modeling columns are then overwritten with deterministic
    normalized values.
    """
    normalized = df.copy()
    for column in RAW_CATEGORICAL_COLUMNS:
        raw_column = f"{column}_raw"
        if column in normalized.columns and raw_column not in normalized.columns:
            normalized[raw_column] = normalized[column]

    if "mode" in normalized.columns:
        normalized["mode"] = normalized["mode"].map(normalize_mode).astype("string")
    if "Route" in normalized.columns:
        normalized["Route"] = normalized["Route"].map(normalize_route).astype("string")
    if "Direction" in normalized.columns:
        normalized["Direction"] = normalized["Direction"].map(normalize_direction).astype("string")
    if "Incident" in normalized.columns:
        normalized["Incident"] = normalized["Incident"].map(normalize_incident).astype("string")
    if "Location" in normalized.columns:
        normalized["Location"] = normalized["Location"].map(normalize_location).astype("string")
    return normalized


def _clean_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    if text.lower() in NULL_LIKE_STRINGS:
        return None
    return text


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _integer_like_text(value: str) -> str | None:
    if re.fullmatch(r"\d+", value):
        return str(int(value))
    if re.fullmatch(r"\d+\.0+", value):
        return str(int(float(value)))
    return None


def _canonical_key(value: str) -> str:
    text = value.upper().replace("&", " AND ")
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"[^A-Z0-9-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
