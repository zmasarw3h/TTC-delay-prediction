"""GTFS route-stop lookup helpers for the local demo UI."""

from __future__ import annotations

import csv
import math
import os
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.data.categorical_normalization import normalize_location, normalize_route


GTFS_PATH_ENV = "TTC_GTFS_ZIP_PATH"
DEFAULT_GTFS_ZIP_PATH = Path("data/raw/ttc_gtfs.zip")
GTFS_ROUTE_TYPE_BUS = "3"
GTFS_ROUTE_TYPE_STREETCAR = "0"
DIRECTION_LABELS = {
    "N": "North",
    "E": "East",
    "S": "South",
    "W": "West",
    "B": "Both / bidirectional",
    "Unknown": "Unknown",
}


@dataclass(frozen=True)
class RouteOption:
    value: str
    label: str
    mode: str | None


@dataclass(frozen=True)
class RouteLocationOption:
    value: str
    label: str
    normalized_location: str


@dataclass(frozen=True)
class RouteStopIndex:
    source_path: str
    routes: dict[str, RouteOption]
    route_stops: dict[str, list[RouteLocationOption]]
    route_directions: dict[str, list[str]]

    @property
    def is_available(self) -> bool:
        return True


@dataclass(frozen=True)
class MissingRouteStopIndex:
    source_path: str
    warning: str

    @property
    def is_available(self) -> bool:
        return False


RouteStopIndexResult = RouteStopIndex | MissingRouteStopIndex


def configured_gtfs_zip_path() -> Path:
    """Return the configured local GTFS zip path."""
    return Path(os.environ.get(GTFS_PATH_ENV, DEFAULT_GTFS_ZIP_PATH))


@lru_cache(maxsize=4)
def load_route_metadata_index(path_text: str | None = None) -> RouteStopIndexResult:
    """Load lightweight route metadata from a local TTC GTFS zip file."""
    path = Path(path_text) if path_text else configured_gtfs_zip_path()
    if not path.exists():
        return MissingRouteStopIndex(
            source_path=str(path),
            warning=(
                "TTC GTFS route-stop data is not configured. Set "
                f"{GTFS_PATH_ENV} or place a GTFS zip at {DEFAULT_GTFS_ZIP_PATH}."
            ),
        )

    try:
        with zipfile.ZipFile(path) as archive:
            routes_by_id = _read_routes(archive)
    except (KeyError, zipfile.BadZipFile, UnicodeDecodeError, csv.Error) as exc:
        return MissingRouteStopIndex(
            source_path=str(path),
            warning=f"Could not load TTC GTFS route data: {exc}",
        )

    routes = {route_key: route_option for route_key, route_option in routes_by_id.values()}
    return RouteStopIndex(source_path=str(path), routes=routes, route_stops={}, route_directions={})


@lru_cache(maxsize=4)
def load_route_stop_index(path_text: str | None = None) -> RouteStopIndexResult:
    """Load route-stop mappings from a local TTC GTFS zip file."""
    path = Path(path_text) if path_text else configured_gtfs_zip_path()
    if not path.exists():
        return MissingRouteStopIndex(
            source_path=str(path),
            warning=(
                "TTC GTFS route-stop data is not configured. Set "
                f"{GTFS_PATH_ENV} or place a GTFS zip at {DEFAULT_GTFS_ZIP_PATH}."
            ),
        )

    try:
        with zipfile.ZipFile(path) as archive:
            routes_by_id = _read_routes(archive)
            trip_to_route = _read_trips(archive, routes_by_id)
            stop_ids_by_route, trip_endpoints_by_route = _read_stop_times(archive, trip_to_route)
            stop_names_by_id, stop_coordinates_by_id = _read_stops(archive)
    except (KeyError, zipfile.BadZipFile, UnicodeDecodeError, csv.Error) as exc:
        return MissingRouteStopIndex(
            source_path=str(path),
            warning=f"Could not load TTC GTFS route-stop data: {exc}",
        )

    route_stops = _build_route_stops(stop_ids_by_route, stop_names_by_id)
    route_directions = _build_route_directions(
        trip_endpoints_by_route,
        stop_coordinates_by_id,
    )
    routes = {
        route_key: route_option
        for route_key, route_option in routes_by_id.values()
        if route_key in route_stops
    }
    return RouteStopIndex(
        source_path=str(path),
        routes=routes,
        route_stops=route_stops,
        route_directions=route_directions,
    )


def route_locations_for_route(
    route: Any,
    index: RouteStopIndexResult | None = None,
) -> tuple[list[RouteLocationOption], str, str | None]:
    """Return route-scoped stop/location options and any warning."""
    route_key = normalize_route(route)
    if index is None:
        index = load_route_stop_index()
    if not index.is_available:
        return [], route_key, index.warning
    if route_key in index.route_stops:
        return index.route_stops[route_key], route_key, None

    base_route = _base_route(route_key)
    if base_route and base_route in index.route_stops:
        return (
            index.route_stops[base_route],
            base_route,
            f"Using base route {base_route} stop list for branch route {route_key}.",
        )
    return [], route_key, f"Route {route_key} was not found in the GTFS route-stop index."


def mode_for_route(route: Any, index: RouteStopIndexResult | None = None) -> str | None:
    """Return bus/streetcar mode for a route when known from GTFS."""
    route_key = normalize_route(route)
    if index is None:
        index = load_route_stop_index()
    if not index.is_available:
        return None
    route_option = index.routes.get(route_key)
    if route_option is not None:
        return route_option.mode

    base_route = _base_route(route_key)
    if base_route and base_route in index.routes:
        return index.routes[base_route].mode
    return None


def directions_for_route(
    route: Any,
    index: RouteStopIndexResult | None = None,
) -> tuple[list[str], str, str | None]:
    """Return route-scoped normalized direction values."""
    route_key = normalize_route(route)
    if index is None:
        index = load_route_stop_index()
    if not index.is_available:
        return [], route_key, index.warning
    if route_key in index.route_directions:
        return index.route_directions[route_key], route_key, None

    base_route = _base_route(route_key)
    if base_route and base_route in index.route_directions:
        return (
            index.route_directions[base_route],
            base_route,
            f"Using base route {base_route} direction list for branch route {route_key}.",
        )
    return [], route_key, f"Route {route_key} directions were not found in the GTFS route-stop index."


def direction_options_for_route(
    route: Any,
    index: RouteStopIndexResult | None = None,
) -> tuple[list[dict[str, str]], str | None]:
    """Return route-scoped direction option objects for the frontend."""
    directions, _, warning = directions_for_route(route, index)
    if len(directions) >= 2 and "B" not in directions:
        directions = [*directions, "B"]
    return [
        {"value": direction, "label": DIRECTION_LABELS[direction]}
        for direction in directions
        if direction in DIRECTION_LABELS
    ], warning


def validate_route_location(
    route: Any,
    location: Any,
    index: RouteStopIndexResult | None = None,
) -> dict[str, Any]:
    """Validate that a normalized location belongs to the selected route."""
    locations, matched_route, route_warning = route_locations_for_route(route, index)
    normalized_location = normalize_location(location)
    route_location_by_value = {option.value: option for option in locations}
    matched_option = route_location_by_value.get(normalized_location)
    accepted = matched_option is not None

    warning = route_warning
    if locations and not accepted:
        warning = (
            f"Location '{normalized_location}' is not a stop on route {matched_route}. "
            "Choose a location from the selected route stop list."
        )
    elif not locations and not warning:
        warning = f"No GTFS stops are available for route {matched_route}."

    return {
        "route": str(route or "").strip(),
        "normalized_route": matched_route,
        "original_location": str(location or "").strip(),
        "normalized_location": normalized_location,
        "route_location": matched_option.value if matched_option else None,
        "route_location_label": matched_option.label if matched_option else None,
        "accepted_for_prediction": accepted,
        "warning": warning,
    }


def route_options_from_index(index: RouteStopIndexResult | None = None) -> list[dict[str, str | None]]:
    """Return route options from GTFS when available."""
    if index is None:
        index = load_route_metadata_index()
    if not index.is_available:
        return []
    routes = {
        route_key: option
        for route_key, option in index.routes.items()
        if not index.route_stops or route_key in index.route_stops
    }
    return [
        {"value": option.value, "label": option.label, "mode": option.mode}
        for option in sorted(routes.values(), key=lambda option: _route_sort_key(option.value))
    ]


def _iter_csv(archive: zipfile.ZipFile, filename: str):
    with archive.open(filename) as file:
        text = (line.decode("utf-8-sig") for line in file)
        yield from csv.DictReader(text)


def _read_routes(archive: zipfile.ZipFile) -> dict[str, tuple[str, RouteOption]]:
    routes_by_id: dict[str, tuple[str, RouteOption]] = {}
    for row in _iter_csv(archive, "routes.txt"):
        route_id = row.get("route_id", "").strip()
        route_short_name = row.get("route_short_name", "").strip()
        normalized_route = normalize_route(route_short_name or route_id)
        if not route_id or normalized_route == "Unknown":
            continue
        mode = _mode_from_route_type(row.get("route_type", ""))
        route_long_name = row.get("route_long_name", "").strip()
        if mode not in {"bus", "streetcar"} or _is_non_surface_line(route_long_name):
            continue
        label_parts = [normalized_route]
        if route_long_name and route_long_name.upper() != normalized_route:
            label_parts.append(route_long_name)
        routes_by_id[route_id] = (
            normalized_route,
            RouteOption(
                value=normalized_route,
                label=" - ".join(label_parts),
                mode=mode,
            ),
        )
    return routes_by_id


def _read_trips(
    archive: zipfile.ZipFile,
    routes_by_id: dict[str, tuple[str, RouteOption]],
) -> dict[str, str]:
    trip_to_route: dict[str, str] = {}
    for row in _iter_csv(archive, "trips.txt"):
        route_id = row.get("route_id", "").strip()
        trip_id = row.get("trip_id", "").strip()
        if route_id in routes_by_id and trip_id:
            trip_to_route[trip_id] = routes_by_id[route_id][0]
    return trip_to_route


def _read_stop_times(
    archive: zipfile.ZipFile,
    trip_to_route: dict[str, str],
) -> tuple[dict[str, set[str]], dict[str, list[tuple[str, str]]]]:
    stop_ids_by_route: dict[str, set[str]] = {}
    trip_endpoints: dict[str, tuple[str, int, str, int, str]] = {}
    for row in _iter_csv(archive, "stop_times.txt"):
        trip_id = row.get("trip_id", "").strip()
        stop_id = row.get("stop_id", "").strip()
        route_key = trip_to_route.get(trip_id)
        if route_key and stop_id:
            stop_ids_by_route.setdefault(route_key, set()).add(stop_id)
            try:
                stop_sequence = int(row.get("stop_sequence", ""))
            except ValueError:
                continue
            current = trip_endpoints.get(trip_id)
            if current is None:
                trip_endpoints[trip_id] = (
                    route_key,
                    stop_sequence,
                    stop_id,
                    stop_sequence,
                    stop_id,
                )
                continue
            _, first_sequence, first_stop_id, last_sequence, last_stop_id = current
            if stop_sequence < first_sequence:
                first_sequence = stop_sequence
                first_stop_id = stop_id
            if stop_sequence > last_sequence:
                last_sequence = stop_sequence
                last_stop_id = stop_id
            trip_endpoints[trip_id] = (
                route_key,
                first_sequence,
                first_stop_id,
                last_sequence,
                last_stop_id,
            )

    endpoints_by_route: dict[str, list[tuple[str, str]]] = {}
    for route_key, _, first_stop_id, _, last_stop_id in trip_endpoints.values():
        if first_stop_id != last_stop_id:
            endpoints_by_route.setdefault(route_key, []).append((first_stop_id, last_stop_id))
    return stop_ids_by_route, endpoints_by_route


def _read_stops(archive: zipfile.ZipFile) -> tuple[dict[str, str], dict[str, tuple[float, float]]]:
    stop_names_by_id: dict[str, str] = {}
    stop_coordinates_by_id: dict[str, tuple[float, float]] = {}
    for row in _iter_csv(archive, "stops.txt"):
        stop_id = row.get("stop_id", "").strip()
        stop_name = row.get("stop_name", "").strip()
        if stop_id and stop_name:
            stop_names_by_id[stop_id] = stop_name
        try:
            stop_lat = float(row.get("stop_lat", ""))
            stop_lon = float(row.get("stop_lon", ""))
        except ValueError:
            continue
        if stop_id:
            stop_coordinates_by_id[stop_id] = (stop_lat, stop_lon)
    return stop_names_by_id, stop_coordinates_by_id


def _build_route_stops(
    stop_ids_by_route: dict[str, set[str]],
    stop_names_by_id: dict[str, str],
) -> dict[str, list[RouteLocationOption]]:
    route_stops: dict[str, list[RouteLocationOption]] = {}
    for route_key, stop_ids in stop_ids_by_route.items():
        options_by_value: dict[str, RouteLocationOption] = {}
        for stop_id in stop_ids:
            stop_name = stop_names_by_id.get(stop_id)
            if not stop_name:
                continue
            normalized = normalize_location(stop_name)
            if normalized == "Unknown":
                continue
            options_by_value.setdefault(
                normalized,
                RouteLocationOption(
                    value=normalized,
                    label=stop_name,
                    normalized_location=normalized,
                ),
            )
        if options_by_value:
            route_stops[route_key] = sorted(
                options_by_value.values(),
                key=lambda option: option.label.lower(),
            )
    return route_stops


def _build_route_directions(
    trip_endpoints_by_route: dict[str, list[tuple[str, str]]],
    stop_coordinates_by_id: dict[str, tuple[float, float]],
) -> dict[str, list[str]]:
    route_directions: dict[str, list[str]] = {}
    direction_order = ["N", "E", "S", "W"]
    for route_key, endpoints in trip_endpoints_by_route.items():
        directions = {
            _direction_from_endpoints(start_stop_id, end_stop_id, stop_coordinates_by_id)
            for start_stop_id, end_stop_id in endpoints
        }
        directions.discard(None)
        if directions:
            route_directions[route_key] = [
                direction for direction in direction_order if direction in directions
            ]
    return route_directions


def _direction_from_endpoints(
    start_stop_id: str,
    end_stop_id: str,
    stop_coordinates_by_id: dict[str, tuple[float, float]],
) -> str | None:
    start = stop_coordinates_by_id.get(start_stop_id)
    end = stop_coordinates_by_id.get(end_stop_id)
    if start is None or end is None:
        return None
    start_lat, start_lon = start
    end_lat, end_lon = end
    delta_lat = end_lat - start_lat
    delta_lon = end_lon - start_lon
    if delta_lat == 0 and delta_lon == 0:
        return None

    mean_lat_radians = math.radians((start_lat + end_lat) / 2)
    x = delta_lon * math.cos(mean_lat_radians)
    y = delta_lat
    if abs(x) >= abs(y):
        return "E" if x > 0 else "W"
    return "N" if y > 0 else "S"


def _mode_from_route_type(route_type: str) -> str | None:
    if str(route_type).strip() == GTFS_ROUTE_TYPE_BUS:
        return "bus"
    if str(route_type).strip() == GTFS_ROUTE_TYPE_STREETCAR:
        return "streetcar"
    return None


def _is_non_surface_line(route_long_name: str) -> bool:
    return route_long_name.strip().upper().endswith(" LINE")


def _base_route(route: str) -> str | None:
    match = re.fullmatch(r"(\d{1,4})[A-Z]{1,2}", route)
    if match:
        return match.group(1)
    return None


def _route_sort_key(route: str) -> tuple[int, int | str, str]:
    match = re.match(r"^(\d+)([A-Z]*)$", route)
    if match:
        return (0, int(match.group(1)), match.group(2))
    return (1, route.lower(), "")
