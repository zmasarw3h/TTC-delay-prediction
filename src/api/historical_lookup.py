"""Prior-only historical feature lookup for API inference."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.api.feature_derivation import parse_prediction_timestamp
from src.data.categorical_normalization import (
    normalize_direction,
    normalize_incident,
    normalize_location,
    normalize_mode,
    normalize_route,
)
from src.features.build_features import HISTORICAL_FEATURES, TARGET_COLUMN


DEFAULT_HISTORICAL_DATA_PATH = Path("data/processed/modeling/modeling_dataset.csv")
HISTORICAL_DATA_PATH_ENV = "TTC_HISTORICAL_FEATURE_DATA_PATH"
LOW_SUPPORT_THRESHOLD = 3

V1_HISTORICAL_FEATURES = [
    "prior_route_mean_delay",
    "prior_route_hour_mean_delay",
    "prior_incident_mean_delay",
    "prior_mode_mean_delay",
    "prior_global_mean_delay",
    "prior_route_hour_7d_mean_delay",
]
V2_HISTORICAL_FEATURES = [
    feature for feature in HISTORICAL_FEATURES if feature not in V1_HISTORICAL_FEATURES
]


@dataclass(frozen=True)
class HistoricalLookupResult:
    """Computed historical feature values plus lookup diagnostics."""

    features: dict[str, float | int | None]
    warnings: list[str]
    normalized_inputs: dict[str, Any]
    support_counts: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)


class HistoricalFeatureLookup:
    """Compute model-required historical features from prior local incidents."""

    def __init__(self, data_path: str | Path | None = None) -> None:
        self.data_path = Path(
            data_path
            or os.environ.get(HISTORICAL_DATA_PATH_ENV)
            or DEFAULT_HISTORICAL_DATA_PATH
        )
        self._data: pd.DataFrame | None = None

    @property
    def is_loaded(self) -> bool:
        return self._data is not None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = self._load_data()
        return self._data

    def _load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(
                "Historical feature data not found. Set "
                f"{HISTORICAL_DATA_PATH_ENV} or create {self.data_path}."
            )

        frame = pd.read_csv(self.data_path)
        required = {"ts", TARGET_COLUMN, "mode", "Route", "Direction", "Incident", "Location"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Historical feature data is missing required column(s): {missing}.")

        loaded = frame.copy()
        loaded["ts"] = pd.to_datetime(loaded["ts"], errors="coerce")
        loaded[TARGET_COLUMN] = pd.to_numeric(loaded[TARGET_COLUMN], errors="coerce")
        loaded = loaded.dropna(subset=["ts", TARGET_COLUMN]).copy()

        loaded["mode"] = loaded["mode"].map(normalize_mode)
        loaded["Route"] = loaded["Route"].map(normalize_route)
        loaded["Direction"] = loaded["Direction"].map(normalize_direction)
        loaded["Incident"] = loaded["Incident"].map(normalize_incident)
        loaded["Location"] = loaded["Location"].map(normalize_location)
        if "hour" not in loaded.columns:
            loaded["hour"] = loaded["ts"].dt.hour
        else:
            loaded["hour"] = pd.to_numeric(loaded["hour"], errors="coerce").fillna(
                loaded["ts"].dt.hour
            )
        return loaded.sort_values("ts").reset_index(drop=True)

    def info(self) -> dict[str, Any]:
        """Return load status and timestamp coverage without forcing repeated reloads."""
        loadable = self.data_path.exists()
        data = self._data
        warnings: list[str] = []

        row_count: int | None = None
        min_ts = None
        max_ts = None
        if data is not None:
            row_count = int(len(data))
            min_ts = data["ts"].min() if not data.empty else None
            max_ts = data["ts"].max() if not data.empty else None
        elif loadable:
            try:
                summary = _summarize_historical_csv(self.data_path)
                row_count = summary["row_count"]
                min_ts = summary["min_timestamp"]
                max_ts = summary["max_timestamp"]
            except ValueError as exc:
                warnings.append(str(exc))

        return {
            "historical_data_path": str(self.data_path),
            "loaded": self.is_loaded,
            "loadable": loadable and not warnings,
            "row_count": row_count,
            "min_timestamp": min_ts.isoformat() if min_ts is not None and not pd.isna(min_ts) else None,
            "max_timestamp": max_ts.isoformat() if max_ts is not None and not pd.isna(max_ts) else None,
            "available_historical_feature_names": list(HISTORICAL_FEATURES),
            "notes_limitations": [
                "Historical lookup uses local modeling_dataset.csv rows only.",
                "Only rows with ts strictly before the prediction timestamp are used.",
                "The lookup is only as current as the local historical CSV.",
                "This is not a production feature store or live TTC data feed.",
                "Location matching uses normalized text and remains approximate assistance.",
            ],
            "warnings": warnings,
        }

    def compute(self, payload: Mapping[str, Any]) -> HistoricalLookupResult:
        timestamp = parse_prediction_timestamp(payload.get("timestamp"))
        normalized = self.normalize_inputs(payload, timestamp)
        data = self.data
        prior = data[data["ts"] < timestamp].copy()

        features: dict[str, float | int | None] = {feature: None for feature in HISTORICAL_FEATURES}
        support_counts: dict[str, int] = {}
        warnings: list[str] = [
            "Historical features were computed from prior local records with ts before the prediction timestamp."
        ]

        min_ts = data["ts"].min() if not data.empty else None
        max_ts = data["ts"].max() if not data.empty else None
        if max_ts is not None and not pd.isna(max_ts) and timestamp > max_ts:
            warnings.append(
                "Prediction timestamp is beyond the latest local historical record "
                f"({max_ts.isoformat()}); lookup uses all available prior records."
            )
        if min_ts is not None and not pd.isna(min_ts):
            if timestamp <= min_ts:
                warnings.append(
                    "Prediction timestamp is before or at the beginning of local historical data; "
                    "historical support is unavailable."
                )
            elif len(prior) < LOW_SUPPORT_THRESHOLD:
                warnings.append(
                    "Prediction timestamp is near the beginning of local historical data; "
                    f"only {len(prior)} prior record(s) are available."
                )

        route = normalized["Route"]
        hour = normalized["hour"]
        incident = normalized["Incident"]
        mode = normalized["mode"]
        direction = normalized["Direction"]
        location = normalized["Location"]

        route_rows = _matching(prior, Route=route)
        route_hour_rows = _matching(prior, Route=route, hour=hour)
        incident_rows = _matching(prior, Incident=incident)
        mode_rows = _matching(prior, mode=mode)
        global_rows = prior
        route_incident_rows = _matching(prior, Route=route, Incident=incident)
        mode_incident_rows = _matching(prior, mode=mode, Incident=incident)
        route_direction_rows = _matching(prior, Route=route, Direction=direction)
        location_rows = _matching(prior, Location=location)

        seven_day_start = timestamp - pd.Timedelta(days=7)
        thirty_day_start = timestamp - pd.Timedelta(days=30)
        seven_day = prior[(prior["ts"] >= seven_day_start) & (prior["ts"] < timestamp)]
        thirty_day = prior[(prior["ts"] >= thirty_day_start) & (prior["ts"] < timestamp)]

        route_hour_7d_rows = _matching(seven_day, Route=route, hour=hour)
        route_30d_rows = _matching(thirty_day, Route=route)
        incident_30d_rows = _matching(thirty_day, Incident=incident)

        _set_mean(features, support_counts, "prior_route_mean_delay", route_rows)
        _set_mean(features, support_counts, "prior_route_hour_mean_delay", route_hour_rows)
        _set_mean(features, support_counts, "prior_incident_mean_delay", incident_rows)
        _set_mean(features, support_counts, "prior_mode_mean_delay", mode_rows)
        _set_mean(features, support_counts, "prior_global_mean_delay", global_rows)

        route_hour_7d = _mean_delay(route_hour_7d_rows)
        support_counts["prior_route_hour_7d_mean_delay"] = int(len(route_hour_7d_rows))
        features["prior_route_hour_7d_mean_delay"] = _first_present(
            route_hour_7d,
            features["prior_route_mean_delay"],
            features["prior_mode_mean_delay"],
            features["prior_global_mean_delay"],
        )
        if route_hour_7d is None and features["prior_route_hour_7d_mean_delay"] is not None:
            warnings.append(
                "prior_route_hour_7d_mean_delay had no 7-day route-hour support and used established fallback values."
            )

        _set_mean(features, support_counts, "prior_route_incident_mean_delay", route_incident_rows)
        _set_mean(features, support_counts, "prior_mode_incident_mean_delay", mode_incident_rows)
        _set_mean(features, support_counts, "prior_route_direction_mean_delay", route_direction_rows)
        features["prior_route_incident_count"] = int(len(route_incident_rows))
        support_counts["prior_route_incident_count"] = int(len(route_incident_rows))
        _set_mean(features, support_counts, "prior_route_30d_mean_delay", route_30d_rows)
        _set_mean(features, support_counts, "prior_incident_30d_mean_delay", incident_30d_rows)
        _set_rate(features, support_counts, "prior_route_30d_severe_rate_30", route_30d_rows, 30)
        _set_rate(
            features,
            support_counts,
            "prior_incident_30d_severe_rate_30",
            incident_30d_rows,
            30,
        )
        _set_rate(features, support_counts, "prior_route_30d_severe_rate_60", route_30d_rows, 60)
        _set_rate(
            features,
            support_counts,
            "prior_incident_30d_severe_rate_60",
            incident_30d_rows,
            60,
        )
        _set_mean(features, support_counts, "prior_location_mean_delay", location_rows)
        features["prior_location_count"] = int(len(location_rows))
        support_counts["prior_location_count"] = int(len(location_rows))

        warnings.extend(_support_warnings(features, support_counts))
        return HistoricalLookupResult(
            features=features,
            warnings=warnings,
            normalized_inputs=normalized,
            support_counts=support_counts,
            metadata={
                "historical_data_path": str(self.data_path),
                "historical_row_count": int(len(data)),
                "prior_row_count": int(len(prior)),
                "min_historical_timestamp": min_ts.isoformat()
                if min_ts is not None and not pd.isna(min_ts)
                else None,
                "max_historical_timestamp": max_ts.isoformat()
                if max_ts is not None and not pd.isna(max_ts)
                else None,
            },
        )

    @staticmethod
    def normalize_inputs(payload: Mapping[str, Any], timestamp: pd.Timestamp) -> dict[str, Any]:
        return {
            "mode": normalize_mode(payload.get("mode")),
            "Route": normalize_route(payload.get("Route")),
            "Direction": normalize_direction(payload.get("Direction")),
            "Incident": normalize_incident(payload.get("Incident")),
            "Location": normalize_location(payload.get("Location")),
            "timestamp": timestamp.isoformat(),
            "hour": int(timestamp.hour),
        }


def _matching(frame: pd.DataFrame, **criteria: Any) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for column, value in criteria.items():
        mask &= frame[column] == value
    return frame[mask]


def _summarize_historical_csv(path: Path) -> dict[str, Any]:
    try:
        ts = pd.read_csv(path, usecols=["ts"])["ts"]
    except ValueError as exc:
        raise ValueError("Historical feature data is missing required column(s): ['ts'].") from exc
    parsed = pd.to_datetime(ts, errors="coerce").dropna()
    return {
        "row_count": int(len(ts)),
        "min_timestamp": parsed.min() if not parsed.empty else None,
        "max_timestamp": parsed.max() if not parsed.empty else None,
    }


def _mean_delay(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    value = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").mean()
    if pd.isna(value):
        return None
    return float(value)


def _set_mean(
    features: dict[str, float | int | None],
    support_counts: dict[str, int],
    feature_name: str,
    frame: pd.DataFrame,
) -> None:
    features[feature_name] = _mean_delay(frame)
    support_counts[feature_name] = int(len(frame))


def _set_rate(
    features: dict[str, float | int | None],
    support_counts: dict[str, int],
    feature_name: str,
    frame: pd.DataFrame,
    threshold: int,
) -> None:
    support_counts[feature_name] = int(len(frame))
    if frame.empty:
        features[feature_name] = None
        return
    features[feature_name] = float((pd.to_numeric(frame[TARGET_COLUMN], errors="coerce") >= threshold).mean())


def _first_present(*values: float | int | None) -> float | int | None:
    for value in values:
        if value is not None:
            return value
    return None


def _support_warnings(
    features: Mapping[str, float | int | None],
    support_counts: Mapping[str, int],
) -> list[str]:
    warnings: list[str] = []
    missing = [feature for feature, value in features.items() if value is None]
    if missing:
        warnings.append(
            "Historical feature(s) had no prior support and were left missing for model imputation: "
            + ", ".join(missing)
            + "."
        )

    low_support = [
        feature
        for feature, count in support_counts.items()
        if 0 < count < LOW_SUPPORT_THRESHOLD and features.get(feature) is not None
    ]
    if low_support:
        warnings.append(
            "Historical feature(s) were computed with low prior support: "
            + ", ".join(f"{feature}={support_counts[feature]}" for feature in low_support)
            + "."
        )
    return warnings
