"""Model option discovery and location matching helpers for the demo UI."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Mapping

from src.data.audit_categories import audit_value, is_route_like
from src.data.categorical_normalization import (
    INCIDENT_CATEGORIES,
    OTHER_CATEGORY,
    UNKNOWN_CATEGORY,
    normalize_direction,
    normalize_incident,
    normalize_location,
    normalize_mode,
    normalize_route,
)

try:  # pragma: no cover - exercised when optional dependency is installed.
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover - fallback is covered in tests.
    fuzz = None
    process = None


OPTION_FIELDS = ("mode", "Route", "Direction", "Incident", "Location")
DEFAULT_MODES = ["bus", "streetcar"]
DEFAULT_DIRECTIONS = ["N", "E", "S", "W", "B"]
CURATED_INCIDENTS = [
    category
    for category in INCIDENT_CATEGORIES
    if category not in {UNKNOWN_CATEGORY, OTHER_CATEGORY}
]
HIGH_CONFIDENCE_LOCATION_SCORE = 90.0
MEDIUM_CONFIDENCE_LOCATION_SCORE = 75.0

ROAD_WORDS = {
    "st": "street",
    "ave": "avenue",
    "rd": "road",
    "blvd": "boulevard",
    "dr": "drive",
    "cres": "crescent",
    "w": "west",
    "e": "east",
    "n": "north",
    "s": "south",
}
MATCH_STOPWORDS = {"and", "at", "the", "of"}


@dataclass(frozen=True)
class LocationMatch:
    original_location: str
    matched_location: str | None
    score: float
    match_type: str
    warning: str | None
    accepted_for_prediction: bool


def model_options_from_categories(
    known_categories: Mapping[str, list[str]] | None,
) -> dict[str, Any]:
    """Build frontend category options from known model categories."""
    categories = known_categories or {}
    warnings: list[str] = []

    modes = sorted(
        {
            normalize_mode(value)
            for value in _category_values(categories, "mode")
            if normalize_mode(value) != UNKNOWN_CATEGORY
        }
        or set(DEFAULT_MODES)
    )
    directions = DEFAULT_DIRECTIONS
    raw_directions = _category_values(categories, "Direction")
    raw_routes = _category_values(categories, "Route")
    routes, route_warnings = _filtered_route_options(raw_routes)
    raw_incidents = _category_values(categories, "Incident")
    incidents = CURATED_INCIDENTS.copy()
    locations = _normalized_location_options(_category_values(categories, "Location"))
    warnings.extend(route_warnings)
    warnings.extend(_ignored_direction_warnings(raw_directions))
    warnings.extend(_ignored_incident_warnings(raw_incidents))

    for field_name, values in [("Route", routes), ("Location", locations)]:
        if not values:
            warnings.append(
                f"{field_name} options were not available from the model artifact."
            )

    options = {
        "modes": modes,
        "routes": routes,
        "directions": directions,
        "incidents": incidents,
        "locations": locations,
        "warnings": warnings,
    }
    options["counts"] = {
        "modes": len(modes),
        "routes": len(routes),
        "directions": len(directions),
        "incidents": len(incidents),
        "locations": len(locations),
    }
    return options


def match_location(
    location: str,
    known_locations: list[str],
) -> LocationMatch:
    """Match a free-form location string to a known model location."""
    original = str(location or "").strip()
    normalized_input = normalize_match_text(original)
    if not normalized_input:
        return LocationMatch(
            original_location=original,
            matched_location=None,
            score=0.0,
            match_type="none",
            warning="Location is empty; enter a location or continue with Unknown.",
            accepted_for_prediction=False,
        )

    normalized_locations = [
        (known_location, normalize_match_text(known_location))
        for known_location in known_locations
        if str(known_location).strip()
    ]
    if not normalized_locations:
        return LocationMatch(
            original_location=original,
            matched_location=None,
            score=0.0,
            match_type="none",
            warning="No known model locations are available for matching.",
            accepted_for_prediction=False,
        )

    for known_location, normalized_location in normalized_locations:
        if normalized_input == normalized_location:
            return LocationMatch(
                original_location=original,
                matched_location=known_location,
                score=100.0,
                match_type="exact",
                warning=None,
                accepted_for_prediction=True,
            )

    matched_location, score = _best_fuzzy_location_match(
        normalized_input,
        normalized_locations,
    )
    if matched_location is None or score < MEDIUM_CONFIDENCE_LOCATION_SCORE:
        return LocationMatch(
            original_location=original,
            matched_location=matched_location,
            score=round(score, 1),
            match_type="none",
            warning=(
                "No confident location match; using the entered location may be treated "
                "as unknown or unseen by the model."
            ),
            accepted_for_prediction=False,
        )

    if score >= HIGH_CONFIDENCE_LOCATION_SCORE:
        return LocationMatch(
            original_location=original,
            matched_location=matched_location,
            score=round(score, 1),
            match_type="fuzzy",
            warning=None,
            accepted_for_prediction=True,
        )

    return LocationMatch(
        original_location=original,
        matched_location=matched_location,
        score=round(score, 1),
        match_type="fuzzy",
        warning="Possible location match found; review and accept it before prediction.",
        accepted_for_prediction=False,
    )


def normalize_match_text(value: str) -> str:
    """Normalize free text for category and location matching."""
    text = normalize_location(value).lower()
    if text.lower() == UNKNOWN_CATEGORY.lower():
        return ""
    punctuation = string.punctuation.replace("&", "").replace("@", "").replace("/", "")
    text = text.translate(str.maketrans({char: " " for char in punctuation}))
    tokens = [ROAD_WORDS.get(token, token) for token in text.split()]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def _category_values(categories: Mapping[str, list[str]], field_name: str) -> list[str]:
    values = categories.get(field_name) or []
    unique = {str(value).strip() for value in values if str(value).strip()}
    return sorted(unique, key=_option_sort_key)


def _filtered_route_options(values: list[str]) -> tuple[list[str], list[str]]:
    routes = sorted(
        {normalize_route(value) for value in values if is_route_like(value)},
        key=_option_sort_key,
    )
    dropped_count = len({value for value in values if value}) - len(routes)
    if dropped_count <= 0:
        return routes, []
    return [*routes], [f"Excluded {dropped_count} non-route-like Route option value(s)."]


def _ignored_direction_warnings(values: list[str]) -> list[str]:
    ignored = [
        value
        for value in values
        if normalize_direction(value) not in set(DEFAULT_DIRECTIONS)
    ]
    if not ignored:
        return []
    suspicious = [
        value
        for value in ignored
        if audit_value("Direction", value, 1, 0.0).suspicious
    ]
    if suspicious:
        return [
            "Ignored "
            f"{len(ignored)} artifact Direction value(s), including "
            f"{len(suspicious)} suspicious/polluted value(s)."
        ]
    return [
        f"Ignored {len(ignored)} artifact Direction value(s) outside fixed UI directions."
    ]


def _ignored_incident_warnings(values: list[str]) -> list[str]:
    suspicious = [
        value
        for value in values
        if normalize_incident(value) == OTHER_CATEGORY
        and audit_value("Incident", value, 1, 0.0).suspicious
    ]
    if not suspicious:
        return []
    return [
        "Ignored "
        f"{len(suspicious)} suspicious raw Incident category value(s) in favor of the curated list."
    ]


def _normalized_location_options(values: list[str]) -> list[str]:
    return sorted(
        {
            normalize_location(value)
            for value in values
            if normalize_location(value) != UNKNOWN_CATEGORY
        },
        key=_option_sort_key,
    )


def _option_sort_key(value: str) -> tuple[int, Any]:
    if value.isdigit():
        return (0, int(value))
    return (1, value.lower())


def _best_fuzzy_location_match(
    normalized_input: str,
    normalized_locations: list[tuple[str, str]],
) -> tuple[str | None, float]:
    choices = {
        known_location: normalized_location
        for known_location, normalized_location in normalized_locations
    }
    if process is not None and fuzz is not None:
        result = process.extractOne(
            normalized_input,
            choices,
            scorer=fuzz.token_set_ratio,
        )
        if result is not None:
            _, score, matched_location = result
            return str(matched_location), float(score)

    best_location: str | None = None
    best_score = 0.0
    input_tokens = set(normalized_input.split())
    for known_location, normalized_location in normalized_locations:
        candidate_tokens = set(normalized_location.split())
        score = _fallback_similarity(normalized_input, normalized_location, input_tokens, candidate_tokens)
        if score > best_score:
            best_location = known_location
            best_score = score
    return best_location, best_score


def _fallback_similarity(
    normalized_input: str,
    normalized_location: str,
    input_tokens: set[str],
    candidate_tokens: set[str],
) -> float:
    if normalized_input in normalized_location or normalized_location in normalized_input:
        shorter = min(len(normalized_input), len(normalized_location))
        longer = max(len(normalized_input), len(normalized_location))
        return max(80.0, 100.0 * shorter / longer)
    if not input_tokens or not candidate_tokens:
        return 0.0
    significant_input = input_tokens - MATCH_STOPWORDS
    significant_candidate = candidate_tokens - MATCH_STOPWORDS
    if significant_input and significant_input.issubset(significant_candidate):
        return 85.0
    overlap = len(input_tokens & candidate_tokens)
    precision = overlap / len(input_tokens)
    recall = overlap / len(candidate_tokens)
    if precision + recall == 0:
        return 0.0
    return 100.0 * (2 * precision * recall) / (precision + recall)
