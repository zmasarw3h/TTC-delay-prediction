"""Build leakage-safe modeling datasets from cleaned TTC delay data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.categorical_normalization import (
    NORMALIZED_CATEGORICAL_COLUMNS,
    RAW_CATEGORICAL_COLUMNS,
    normalize_categorical_columns,
)


DEFAULT_INPUT = Path("data/processed/ttc_delays_cleaned.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/modeling")
DEFAULT_MAX_DELAY_MINUTES = 240
DEFAULT_TRAIN_END = "2022-12-31"
DEFAULT_VAL_YEAR = 2023
DEFAULT_TEST_YEAR = 2024

CATEGORICAL_COLUMNS = ["mode", "Route", "Direction", "Location", "Incident", "Vehicle"]
MAIN_CATEGORICAL_FEATURES = ["mode", "Route", "Direction", "Incident", "Location"]
TARGET_COLUMN = "Min Delay"
SECONDARY_TARGET_COLUMN = "severe_delay_15"
LEAKAGE_SENSITIVE_COLUMNS = ["Min Gap", "Min Delay"]
EXCLUDED_COLUMNS = [
    "Date",
    "Min Gap",
    "Vehicle",
    "source_file",
    "source_sheet",
    "ts",
    TARGET_COLUMN,
    SECONDARY_TARGET_COLUMN,
]
TIME_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "day_of_year",
    "day_sin",
    "day_cos",
]
HISTORICAL_FEATURES = [
    "prior_route_mean_delay",
    "prior_route_hour_mean_delay",
    "prior_incident_mean_delay",
    "prior_mode_mean_delay",
    "prior_global_mean_delay",
    "prior_route_hour_7d_mean_delay",
    "prior_route_incident_mean_delay",
    "prior_mode_incident_mean_delay",
    "prior_route_direction_mean_delay",
    "prior_route_incident_count",
    "prior_route_30d_mean_delay",
    "prior_incident_30d_mean_delay",
    "prior_route_30d_severe_rate_30",
    "prior_incident_30d_severe_rate_30",
    "prior_route_30d_severe_rate_60",
    "prior_incident_30d_severe_rate_60",
    "prior_location_mean_delay",
    "prior_location_count",
]
NUMERIC_FEATURES = TIME_FEATURES + HISTORICAL_FEATURES
FEATURE_COLUMNS = TIME_FEATURES + MAIN_CATEGORICAL_FEATURES + HISTORICAL_FEATURES


def load_cleaned_delays(input_path: Path) -> pd.DataFrame:
    """Load the cleaned delay audit dataset with stable categorical dtypes."""
    dtype = {column: "string" for column in CATEGORICAL_COLUMNS}
    return pd.read_csv(input_path, dtype=dtype, parse_dates=["ts"])


def create_modeling_dataset(df: pd.DataFrame, max_delay_minutes: int) -> pd.DataFrame:
    """Return a filtered copy for modeling without mutating the audit dataset."""
    modeling = df.copy()
    modeling["ts"] = pd.to_datetime(modeling["ts"], errors="coerce")
    modeling[TARGET_COLUMN] = pd.to_numeric(modeling[TARGET_COLUMN], errors="coerce")

    modeling = modeling.dropna(subset=["ts", TARGET_COLUMN])
    modeling = modeling[
        (modeling[TARGET_COLUMN] >= 0)
        & (modeling[TARGET_COLUMN] <= max_delay_minutes)
    ].copy()
    return modeling.sort_values("ts").reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add incident-time features available when an incident is reported."""
    featured = df.copy()
    ts = pd.to_datetime(featured["ts"], errors="coerce")

    featured["hour"] = ts.dt.hour
    featured["day_of_week"] = ts.dt.dayofweek
    featured["month"] = ts.dt.month
    featured["is_weekend"] = featured["day_of_week"].isin([5, 6]).astype(int)
    if "is_holiday" not in featured.columns:
        featured["is_holiday"] = 0
    featured["is_holiday"] = pd.to_numeric(featured["is_holiday"], errors="coerce").fillna(0).astype(int)

    featured["hour_sin"] = np.sin(2 * np.pi * featured["hour"] / 24)
    featured["hour_cos"] = np.cos(2 * np.pi * featured["hour"] / 24)
    featured["day_of_year"] = ts.dt.dayofyear
    featured["day_sin"] = np.sin(2 * np.pi * featured["day_of_year"] / 366)
    featured["day_cos"] = np.cos(2 * np.pi * featured["day_of_year"] / 366)
    return featured


def _prior_timestamp_mean(df: pd.DataFrame, group_columns: list[str] | None = None) -> pd.Series:
    """Mean target from records with timestamps strictly before each row."""
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if group_columns is None:
        time_stats = (
            pd.DataFrame({"ts": df["ts"], TARGET_COLUMN: target})
            .groupby("ts", dropna=False)[TARGET_COLUMN]
            .agg(["sum", "count"])
            .sort_index()
        )
        prior_sum = time_stats["sum"].cumsum().shift(1)
        prior_count = time_stats["count"].cumsum().shift(1)
        prior_mean_by_ts = prior_sum / prior_count
        return df["ts"].map(prior_mean_by_ts)

    work = df[group_columns + ["ts"]].copy()
    work[TARGET_COLUMN] = target
    grouped_stats = (
        work.groupby(group_columns + ["ts"], dropna=False)[TARGET_COLUMN]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(group_columns + ["ts"])
    )
    grouped = grouped_stats.groupby(group_columns, dropna=False)
    grouped_stats["_cum_sum"] = grouped["sum"].cumsum()
    grouped_stats["_cum_count"] = grouped["count"].cumsum()
    cumulative_grouped = grouped_stats.groupby(group_columns, dropna=False)
    grouped_stats["_prior_sum"] = cumulative_grouped["_cum_sum"].shift(1)
    grouped_stats["_prior_count"] = cumulative_grouped["_cum_count"].shift(1)
    grouped_stats["_prior_mean"] = grouped_stats["_prior_sum"] / grouped_stats["_prior_count"]

    return work.merge(
        grouped_stats[group_columns + ["ts", "_prior_mean"]],
        on=group_columns + ["ts"],
        how="left",
    )["_prior_mean"].set_axis(df.index)


def _prior_timestamp_count(df: pd.DataFrame, group_columns: list[str]) -> pd.Series:
    """Count records with timestamps strictly before each row for a grouping."""
    work = df[group_columns + ["ts"]].copy()
    grouped_stats = (
        work.groupby(group_columns + ["ts"], dropna=False)
        .size()
        .rename("_count")
        .reset_index()
        .sort_values(group_columns + ["ts"])
    )
    grouped_stats["_prior_count"] = (
        grouped_stats.groupby(group_columns, dropna=False)["_count"].cumsum()
        - grouped_stats["_count"]
    )

    return (
        work.merge(
            grouped_stats[group_columns + ["ts", "_prior_count"]],
            on=group_columns + ["ts"],
            how="left",
        )["_prior_count"]
        .fillna(0)
        .astype("int64")
        .set_axis(df.index)
    )


def _prior_rolling_timestamp_mean(
    df: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    window: str,
) -> pd.Series:
    """Rolling prior-only mean over timestamp-aggregated group history."""
    work = df[group_columns + ["ts"]].copy()
    work[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    grouped_stats = (
        work.groupby(group_columns + ["ts"], dropna=False)[value_column]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(group_columns + ["ts"])
    )

    pieces: list[pd.DataFrame] = []
    for _, group in grouped_stats.groupby(group_columns, dropna=False, sort=False):
        ordered = group.sort_values("ts")
        indexed = ordered.set_index(pd.DatetimeIndex(ordered["ts"]))
        prior_sum = indexed["sum"].rolling(window, closed="left").sum()
        prior_count = indexed["count"].rolling(window, closed="left").sum()
        rolled = ordered[group_columns + ["ts"]].copy()
        rolled["_prior_rolling_mean"] = (prior_sum / prior_count).to_numpy()
        pieces.append(rolled)

    if not pieces:
        return pd.Series(dtype="float64", index=df.index)

    rolling_by_ts = pd.concat(pieces, ignore_index=True)
    return work.merge(
        rolling_by_ts[group_columns + ["ts", "_prior_rolling_mean"]],
        on=group_columns + ["ts"],
        how="left",
    )["_prior_rolling_mean"].set_axis(df.index)


def _prior_route_hour_7d_mean(df: pd.DataFrame) -> pd.Series:
    """Mean prior delay for the same route-hour in the previous 7 calendar days."""
    pieces: list[pd.Series] = []
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    work = df[["Route", "hour", "ts"]].copy()
    work[TARGET_COLUMN] = target
    work["_original_index"] = df.index

    for _, group in work.groupby(["Route", "hour"], dropna=False, sort=False):
        ordered = group.sort_values(["ts", "_original_index"])
        indexed = pd.Series(
            ordered[TARGET_COLUMN].to_numpy(),
            index=pd.DatetimeIndex(ordered["ts"]),
        )
        rolled = indexed.rolling("7D", closed="left").mean()
        pieces.append(pd.Series(rolled.to_numpy(), index=ordered["_original_index"]))

    if not pieces:
        return pd.Series(dtype="float64", index=df.index)
    return pd.concat(pieces).reindex(df.index)


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add prior-only target-derived historical features."""
    featured = df.sort_values("ts").reset_index(drop=True).copy()
    featured["prior_global_mean_delay"] = _prior_timestamp_mean(featured)
    featured["prior_route_mean_delay"] = _prior_timestamp_mean(featured, ["Route"])
    featured["prior_route_hour_mean_delay"] = _prior_timestamp_mean(featured, ["Route", "hour"])
    featured["prior_incident_mean_delay"] = _prior_timestamp_mean(featured, ["Incident"])
    featured["prior_mode_mean_delay"] = _prior_timestamp_mean(featured, ["mode"])
    featured["prior_route_incident_mean_delay"] = _prior_timestamp_mean(
        featured, ["Route", "Incident"]
    )
    featured["prior_mode_incident_mean_delay"] = _prior_timestamp_mean(
        featured, ["mode", "Incident"]
    )
    featured["prior_route_direction_mean_delay"] = _prior_timestamp_mean(
        featured, ["Route", "Direction"]
    )
    featured["prior_route_incident_count"] = _prior_timestamp_count(
        featured, ["Route", "Incident"]
    )
    featured["prior_location_mean_delay"] = _prior_timestamp_mean(featured, ["Location"])
    featured["prior_location_count"] = _prior_timestamp_count(featured, ["Location"])

    featured["prior_route_30d_mean_delay"] = _prior_rolling_timestamp_mean(
        featured, ["Route"], TARGET_COLUMN, "30D"
    )
    featured["prior_incident_30d_mean_delay"] = _prior_rolling_timestamp_mean(
        featured, ["Incident"], TARGET_COLUMN, "30D"
    )
    for threshold in [30, 60]:
        severe_column = f"_severe_delay_{threshold}"
        featured[severe_column] = (featured[TARGET_COLUMN] >= threshold).astype(int)
        featured[f"prior_route_30d_severe_rate_{threshold}"] = _prior_rolling_timestamp_mean(
            featured, ["Route"], severe_column, "30D"
        )
        featured[f"prior_incident_30d_severe_rate_{threshold}"] = _prior_rolling_timestamp_mean(
            featured, ["Incident"], severe_column, "30D"
        )
        featured = featured.drop(columns=[severe_column])

    route_hour_7d = _prior_route_hour_7d_mean(featured)
    featured["prior_route_hour_7d_mean_delay"] = (
        route_hour_7d.fillna(featured["prior_route_mean_delay"])
        .fillna(featured["prior_mode_mean_delay"])
        .fillna(featured["prior_global_mean_delay"])
    )
    return featured


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured[SECONDARY_TARGET_COLUMN] = (featured[TARGET_COLUMN] >= 15).astype(int)
    return featured


def build_feature_frame(df: pd.DataFrame, max_delay_minutes: int) -> pd.DataFrame:
    normalized = normalize_categorical_columns(df)
    modeling = create_modeling_dataset(normalized, max_delay_minutes=max_delay_minutes)
    modeled = add_time_features(modeling)
    modeled = add_historical_features(modeled)
    return add_targets(modeled)


def split_modeling_dataset(
    df: pd.DataFrame,
    train_end: str,
    val_year: int,
    test_year: int,
) -> dict[str, pd.DataFrame]:
    ts = pd.to_datetime(df["ts"], errors="coerce")
    train_end_ts = pd.Timestamp(train_end).replace(hour=23, minute=59, second=59, microsecond=999999)
    return {
        "train": df[ts <= train_end_ts].copy(),
        "validation": df[ts.dt.year == val_year].copy(),
        "test": df[ts.dt.year == test_year].copy(),
    }


def _date_range(df: pd.DataFrame) -> dict[str, str | None]:
    if df.empty:
        return {"min": None, "max": None}
    ts = pd.to_datetime(df["ts"], errors="coerce")
    return {
        "min": ts.min().isoformat() if pd.notna(ts.min()) else None,
        "max": ts.max().isoformat() if pd.notna(ts.max()) else None,
    }


def create_feature_metadata(
    modeling_df: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    max_delay_minutes: int,
    train_end: str,
    val_year: int,
    test_year: int,
) -> dict[str, Any]:
    """Create metadata describing the modeling dataset contract."""
    return {
        "target_column": TARGET_COLUMN,
        "optional_secondary_target": SECONDARY_TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "leakage_sensitive_columns": LEAKAGE_SENSITIVE_COLUMNS,
        "max_delay_threshold": {
            "column": TARGET_COLUMN,
            "minimum_inclusive": 0,
            "maximum_inclusive": max_delay_minutes,
        },
        "split_definitions": {
            "train": f"ts <= {train_end}",
            "validation": f"year(ts) == {val_year}",
            "test": f"year(ts) == {test_year}",
        },
        "row_counts_by_split": {
            split_name: int(len(split_df)) for split_name, split_df in splits.items()
        },
        "date_ranges_by_split": {
            split_name: _date_range(split_df) for split_name, split_df in splits.items()
        },
        "modeling_dataset_rows": int(len(modeling_df)),
        "modeling_dataset_date_range": _date_range(modeling_df),
        "categorical_columns": MAIN_CATEGORICAL_FEATURES,
        "categorical_normalization": {
            "applied": True,
            "normalized_columns": NORMALIZED_CATEGORICAL_COLUMNS,
            "raw_columns_preserved": [
                f"{column}_raw"
                for column in RAW_CATEGORICAL_COLUMNS
                if f"{column}_raw" in modeling_df.columns
            ],
            "rules_summary": {
                "mode": "Case-insensitive bus/streetcar normalization; unsupported or missing values become Unknown during feature builds.",
                "Route": "Whitespace stripped, integer-like values converted to route strings, route variants preserved, obvious non-route text set to Unknown.",
                "Direction": "Strictly normalized to N, E, S, W, B, or Unknown.",
                "Incident": "Variants mapped to curated operational categories; unrecognized non-missing labels become Other.",
                "Location": "Safe deterministic uppercase text cleanup, separator normalization, and common road/station abbreviation expansion only.",
            },
        },
        "numeric_columns": NUMERIC_FEATURES,
        "historical_feature_groups": {
            "v1": [
                "prior_route_mean_delay",
                "prior_route_hour_mean_delay",
                "prior_incident_mean_delay",
                "prior_mode_mean_delay",
                "prior_global_mean_delay",
                "prior_route_hour_7d_mean_delay",
            ],
            "v2_prior_expanding": [
                "prior_route_incident_mean_delay",
                "prior_mode_incident_mean_delay",
                "prior_route_direction_mean_delay",
                "prior_route_incident_count",
                "prior_location_mean_delay",
                "prior_location_count",
            ],
            "v2_prior_rolling_30d": [
                "prior_route_30d_mean_delay",
                "prior_incident_30d_mean_delay",
                "prior_route_30d_severe_rate_30",
                "prior_incident_30d_severe_rate_30",
                "prior_route_30d_severe_rate_60",
                "prior_incident_30d_severe_rate_60",
            ],
        },
        "historical_feature_leakage_rules": {
            "prior_only_timestamp_rule": "Historical features use only rows where historical_row.ts < current_row.ts.",
            "same_timestamp_rule": "Rows sharing the exact same timestamp are aggregated at timestamp level and never use one another as history.",
            "fallback_policy": "Existing v1 prior_route_hour_7d_mean_delay keeps its documented fallback chain. V2 means and rates are not filled with full-dataset group means; missing values remain transparent for downstream imputation. Count features expose support.",
            "excluded_target_leakage_columns": LEAKAGE_SENSITIVE_COLUMNS,
        },
        "rolling_window_definitions": {
            "30D": {
                "window": "current ts - 30 calendar days <= historical ts < current ts",
                "features": [
                    "prior_route_30d_mean_delay",
                    "prior_incident_30d_mean_delay",
                    "prior_route_30d_severe_rate_30",
                    "prior_incident_30d_severe_rate_30",
                    "prior_route_30d_severe_rate_60",
                    "prior_incident_30d_severe_rate_60",
                ],
            },
            "7D": {
                "window": "current ts - 7 calendar days <= historical ts < current ts",
                "features": ["prior_route_hour_7d_mean_delay"],
            },
        },
        "severe_rate_feature_definitions": {
            "prior_route_30d_severe_rate_30": "Mean of prior indicator Min Delay >= 30 over the previous 30 calendar days for the same Route.",
            "prior_incident_30d_severe_rate_30": "Mean of prior indicator Min Delay >= 30 over the previous 30 calendar days for the same Incident.",
            "prior_route_30d_severe_rate_60": "Mean of prior indicator Min Delay >= 60 over the previous 30 calendar days for the same Route.",
            "prior_incident_30d_severe_rate_60": "Mean of prior indicator Min Delay >= 60 over the previous 30 calendar days for the same Incident.",
        },
        "high_cardinality_feature_warnings": {
            "Location": {
                "features": ["prior_location_mean_delay", "prior_location_count"],
                "warning": "Location history is high-cardinality and support-sensitive. prior_location_mean_delay is intentionally left missing when no prior location history exists; prior_location_count should be used to learn confidence or guard downstream use.",
            }
        },
        "historical_feature_definitions": {
            "prior_route_mean_delay": "Mean Min Delay for rows with ts strictly before the current row and the same Route.",
            "prior_route_hour_mean_delay": "Mean Min Delay for rows with ts strictly before the current row and the same Route and hour.",
            "prior_incident_mean_delay": "Mean Min Delay for rows with ts strictly before the current row and the same Incident.",
            "prior_mode_mean_delay": "Mean Min Delay for rows with ts strictly before the current row and the same mode.",
            "prior_global_mean_delay": "Mean Min Delay over all rows with ts strictly before the current row.",
            "prior_route_hour_7d_mean_delay": (
                "Mean Min Delay from prior incidents with the same Route and hour whose "
                "timestamps are within the previous 7 calendar days and strictly before the current row. "
                "Missing values fall back to prior route mean, then prior mode mean, then prior global mean."
            ),
            "prior_route_incident_mean_delay": "Mean Min Delay for rows with ts strictly before the current row and the same Route and Incident.",
            "prior_mode_incident_mean_delay": "Mean Min Delay for rows with ts strictly before the current row and the same mode and Incident.",
            "prior_route_direction_mean_delay": "Mean Min Delay for rows with ts strictly before the current row and the same Route and Direction.",
            "prior_route_incident_count": "Count of rows with ts strictly before the current row and the same Route and Incident.",
            "prior_route_30d_mean_delay": "Mean Min Delay for rows with the same Route where current ts - 30 calendar days <= historical ts < current ts.",
            "prior_incident_30d_mean_delay": "Mean Min Delay for rows with the same Incident where current ts - 30 calendar days <= historical ts < current ts.",
            "prior_route_30d_severe_rate_30": "Mean prior severe-delay indicator Min Delay >= 30 for rows with the same Route in the prior 30 calendar days.",
            "prior_incident_30d_severe_rate_30": "Mean prior severe-delay indicator Min Delay >= 30 for rows with the same Incident in the prior 30 calendar days.",
            "prior_route_30d_severe_rate_60": "Mean prior severe-delay indicator Min Delay >= 60 for rows with the same Route in the prior 30 calendar days.",
            "prior_incident_30d_severe_rate_60": "Mean prior severe-delay indicator Min Delay >= 60 for rows with the same Incident in the prior 30 calendar days.",
            "prior_location_mean_delay": "Mean Min Delay for rows with ts strictly before the current row and the same normalized Location. High-cardinality support-sensitive feature; missing values are preserved when no prior location history exists.",
            "prior_location_count": "Count of rows with ts strictly before the current row and the same normalized Location.",
        },
        "generated_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def write_feature_outputs(
    input_path: Path,
    output_dir: Path,
    max_delay_minutes: int,
    train_end: str,
    val_year: int,
    test_year: int,
) -> None:
    cleaned = load_cleaned_delays(input_path)
    modeling_df = build_feature_frame(cleaned, max_delay_minutes=max_delay_minutes)
    splits = split_modeling_dataset(
        modeling_df,
        train_end=train_end,
        val_year=val_year,
        test_year=test_year,
    )
    metadata = create_feature_metadata(
        modeling_df=modeling_df,
        splits=splits,
        max_delay_minutes=max_delay_minutes,
        train_end=train_end,
        val_year=val_year,
        test_year=test_year,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    modeling_df.to_csv(output_dir / "modeling_dataset.csv", index=False)
    splits["train"].to_csv(output_dir / "train.csv", index=False)
    splits["validation"].to_csv(output_dir / "validation.csv", index=False)
    splits["test"].to_csv(output_dir / "test.csv", index=False)
    (output_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe TTC modeling features.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-delay-minutes", type=int, default=DEFAULT_MAX_DELAY_MINUTES)
    parser.add_argument("--train-end", type=str, default=DEFAULT_TRAIN_END)
    parser.add_argument("--val-year", type=int, default=DEFAULT_VAL_YEAR)
    parser.add_argument("--test-year", type=int, default=DEFAULT_TEST_YEAR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_feature_outputs(
        input_path=args.input,
        output_dir=args.output_dir,
        max_delay_minutes=args.max_delay_minutes,
        train_end=args.train_end,
        val_year=args.val_year,
        test_year=args.test_year,
    )


if __name__ == "__main__":
    main()
