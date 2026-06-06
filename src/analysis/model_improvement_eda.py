"""Phase 11A model-improvement EDA for historical feature planning.

This module reads existing normalized modeling splits plus generated prediction
reports. It does not train models or write model artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MODELING_DIR = Path("data/processed/modeling")
DEFAULT_ERROR_ANALYSIS_DIR = Path("reports/error_analysis")
DEFAULT_CALIBRATION_DIR = Path("reports/calibration")
DEFAULT_OUTPUT_DIR = Path("reports/model_improvement_eda")
DEFAULT_MIN_GROUP_SIZE = 100
DEFAULT_TOP_N = 50
TARGET_COLUMN = "Min Delay"
TIMESTAMP_COLUMN = "ts"
EVALUATION_SPLITS = ["validation", "test"]
GROUP_COLUMNS = ["mode", "Route", "Direction", "Incident", "Location", "hour", "month", "delay_bucket"]
SEVERE_GROUP_COLUMNS = ["mode", "Route", "Direction", "Incident", "Location", "hour"]
HIGH_CARDINALITY_COLUMNS = {"Route", "Location"}
SUPPORT_THRESHOLDS = [1, 5, 20, 50]
SEVERE_THRESHOLDS = [30, 60]


def assign_delay_bucket(values: pd.Series) -> pd.Series:
    """Assign delay minutes to stable EDA buckets."""
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, 5, 10, 15, 30, 60, 120, 240, np.inf],
        labels=["0-5", "6-10", "11-15", "16-30", "31-60", "61-120", "121-240", "241+"],
        right=True,
    ).astype("string")


def _rmse(error: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(pd.to_numeric(error, errors="coerce")))))


def grouped_error_metrics(
    frame: pd.DataFrame,
    group_column: str,
    min_group_size: int | None = None,
) -> pd.DataFrame:
    """Compute residual metrics by split and one group column."""
    columns = [
        "split",
        "group_column",
        "group_value",
        "row_count",
        "mae",
        "rmse",
        "mean_error",
        "median_absolute_error",
        "p90_absolute_error",
        "total_error_contribution",
    ]
    if group_column not in frame.columns:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []
    for (split_name, group_value), group in frame.groupby(["split", group_column], dropna=False):
        if min_group_size is not None and len(group) < min_group_size:
            continue
        absolute_error = pd.to_numeric(group["absolute_error"], errors="coerce")
        error = pd.to_numeric(group["error"], errors="coerce")
        mae = float(absolute_error.mean())
        records.append(
            {
                "split": split_name,
                "group_column": group_column,
                "group_value": group_value,
                "row_count": int(len(group)),
                "mae": mae,
                "rmse": _rmse(error),
                "mean_error": float(error.mean()),
                "median_absolute_error": float(absolute_error.median()),
                "p90_absolute_error": float(absolute_error.quantile(0.90)),
                "total_error_contribution": total_error_contribution(len(group), mae),
            }
        )

    return pd.DataFrame.from_records(records, columns=columns).sort_values(
        ["split", "group_column", "mae"],
        ascending=[True, True, False],
        ignore_index=True,
    )


def total_error_contribution(row_count: int, mae: float) -> float:
    """Return a group's absolute-error contribution."""
    return float(row_count) * float(mae)


def _strict_prior_counts_for_group(frame: pd.DataFrame, group_columns: list[str]) -> pd.Series:
    """Count prior rows with the same group and strictly earlier timestamp."""
    required = [*group_columns, TIMESTAMP_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot compute prior counts; missing columns: {missing}")

    work = frame[required].copy()
    work[TIMESTAMP_COLUMN] = pd.to_datetime(work[TIMESTAMP_COLUMN], errors="coerce")
    if work[TIMESTAMP_COLUMN].isna().any():
        raise ValueError("Cannot compute prior counts because some timestamps are unparseable.")
    work["_row_id"] = np.arange(len(work))

    counts_by_ts = (
        work.groupby([*group_columns, TIMESTAMP_COLUMN], dropna=False)
        .size()
        .rename("_rows_at_ts")
        .reset_index()
        .sort_values([*group_columns, TIMESTAMP_COLUMN], kind="mergesort")
    )
    counts_by_ts["_prior_count"] = (
        counts_by_ts.groupby(group_columns, dropna=False)["_rows_at_ts"].cumsum()
        - counts_by_ts["_rows_at_ts"]
    )
    merged = work.merge(
        counts_by_ts[[*group_columns, TIMESTAMP_COLUMN, "_prior_count"]],
        on=[*group_columns, TIMESTAMP_COLUMN],
        how="left",
    ).sort_values("_row_id")
    return merged["_prior_count"].astype("int64").reset_index(drop=True)


def prior_count_support(
    full_frame: pd.DataFrame,
    eval_mask: pd.Series,
    grouping_name: str,
    group_columns: list[str],
) -> dict[str, Any]:
    """Summarize strict prior-count support for validation/test rows."""
    prior_counts = _strict_prior_counts_for_group(full_frame, group_columns)
    eval_counts = prior_counts.loc[eval_mask.reset_index(drop=True)].astype("int64")
    row_count = int(len(eval_counts))
    record: dict[str, Any] = {
        "grouping": grouping_name,
        "group_columns": " + ".join(group_columns),
        "row_count": row_count,
    }
    for threshold in SUPPORT_THRESHOLDS:
        record[f"pct_with_prior_{threshold}"] = (
            float((eval_counts >= threshold).mean() * 100.0) if row_count else np.nan
        )
    record.update(
        {
            "median_prior_count": float(eval_counts.median()) if row_count else np.nan,
            "p25_prior_count": float(eval_counts.quantile(0.25)) if row_count else np.nan,
            "p75_prior_count": float(eval_counts.quantile(0.75)) if row_count else np.nan,
            "max_prior_count": int(eval_counts.max()) if row_count else 0,
        }
    )
    return record


def _strict_prior_mean_for_group(
    frame: pd.DataFrame,
    group_columns: list[str],
    target_column: str = TARGET_COLUMN,
) -> pd.Series:
    """Compute strict prior-only group target means for each row."""
    required = [*group_columns, TIMESTAMP_COLUMN, target_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot compute prior means; missing columns: {missing}")

    work = frame[required].copy()
    work[TIMESTAMP_COLUMN] = pd.to_datetime(work[TIMESTAMP_COLUMN], errors="coerce")
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    if work[TIMESTAMP_COLUMN].isna().any():
        raise ValueError("Cannot compute prior means because some timestamps are unparseable.")
    work["_row_id"] = np.arange(len(work))

    by_ts = (
        work.groupby([*group_columns, TIMESTAMP_COLUMN], dropna=False)[target_column]
        .agg(_sum_at_ts="sum", _count_at_ts="count")
        .reset_index()
        .sort_values([*group_columns, TIMESTAMP_COLUMN], kind="mergesort")
    )
    grouped = by_ts.groupby(group_columns, dropna=False)
    by_ts["_prior_sum"] = grouped["_sum_at_ts"].cumsum() - by_ts["_sum_at_ts"]
    by_ts["_prior_count"] = grouped["_count_at_ts"].cumsum() - by_ts["_count_at_ts"]
    by_ts["_prior_mean"] = by_ts["_prior_sum"] / by_ts["_prior_count"].replace(0, np.nan)

    merged = work.merge(
        by_ts[[*group_columns, TIMESTAMP_COLUMN, "_prior_mean"]],
        on=[*group_columns, TIMESTAMP_COLUMN],
        how="left",
    ).sort_values("_row_id")
    return merged["_prior_mean"].astype("float64").reset_index(drop=True)


def recommendation_table() -> pd.DataFrame:
    """Create the static candidate recommendation schema with initial rankings."""
    records = [
        {
            "feature_name": "prior_route_incident_mean_delay",
            "feature_type": "prior_mean",
            "grouping": "Route + Incident",
            "target": "both",
            "expected_value": "high",
            "reason": "Captures route-specific incident severity while using prior-only history.",
            "risks": "sparse; leakage risk",
            "recommended_for_phase_11b": True,
            "priority_rank": 1,
        },
        {
            "feature_name": "prior_mode_incident_mean_delay",
            "feature_type": "prior_mean",
            "grouping": "mode + Incident",
            "target": "both",
            "expected_value": "high",
            "reason": "Lower-cardinality fallback for incident severity by vehicle mode.",
            "risks": "leakage risk",
            "recommended_for_phase_11b": True,
            "priority_rank": 2,
        },
        {
            "feature_name": "prior_route_direction_mean_delay",
            "feature_type": "prior_mean",
            "grouping": "Route + Direction",
            "target": "regression",
            "expected_value": "medium",
            "reason": "Adds directional route behavior beyond current route-level history.",
            "risks": "sparse; leakage risk",
            "recommended_for_phase_11b": True,
            "priority_rank": 3,
        },
        {
            "feature_name": "prior_route_incident_count",
            "feature_type": "prior_count",
            "grouping": "Route + Incident",
            "target": "both",
            "expected_value": "medium",
            "reason": "Exposes support/confidence for route-incident historical means.",
            "risks": "sparse",
            "recommended_for_phase_11b": True,
            "priority_rank": 4,
        },
        {
            "feature_name": "prior_route_30d_mean_delay",
            "feature_type": "rolling_mean",
            "grouping": "Route",
            "target": "regression",
            "expected_value": "high",
            "reason": "Current recent route behavior may capture service disruptions and seasonal drift.",
            "risks": "slow computation; leakage risk",
            "recommended_for_phase_11b": True,
            "priority_rank": 5,
        },
        {
            "feature_name": "prior_incident_30d_mean_delay",
            "feature_type": "rolling_mean",
            "grouping": "Incident",
            "target": "both",
            "expected_value": "medium",
            "reason": "Recent incident severity can adapt to operational changes with broad support.",
            "risks": "slow computation; leakage risk",
            "recommended_for_phase_11b": True,
            "priority_rank": 6,
        },
        {
            "feature_name": "prior_route_30d_severe_rate_30",
            "feature_type": "rolling_rate",
            "grouping": "Route",
            "target": "severe_30",
            "expected_value": "high",
            "reason": "Directly targets severe-delay propensity using recent prior route outcomes.",
            "risks": "slow computation; leakage risk",
            "recommended_for_phase_11b": True,
            "priority_rank": 7,
        },
        {
            "feature_name": "prior_incident_30d_severe_rate_30",
            "feature_type": "rolling_rate",
            "grouping": "Incident",
            "target": "severe_30",
            "expected_value": "medium",
            "reason": "Direct incident-level severe-rate signal with lower cardinality than route incident.",
            "risks": "slow computation; leakage risk",
            "recommended_for_phase_11b": True,
            "priority_rank": 8,
        },
        {
            "feature_name": "prior_location_mean_delay",
            "feature_type": "prior_mean",
            "grouping": "Location",
            "target": "both",
            "expected_value": "medium",
            "reason": "Potentially captures recurring hotspots if normalized location support is healthy.",
            "risks": "high cardinality; sparse; leakage risk",
            "recommended_for_phase_11b": False,
            "priority_rank": 9,
        },
        {
            "feature_name": "prior_location_incident_mean_delay",
            "feature_type": "prior_mean",
            "grouping": "Location + Incident",
            "target": "both",
            "expected_value": "low",
            "reason": "Useful only if support is unexpectedly strong after location normalization.",
            "risks": "high cardinality; sparse; slow computation; leakage risk",
            "recommended_for_phase_11b": False,
            "priority_rank": 10,
        },
    ]
    return pd.DataFrame.from_records(records)


def _require_file(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}\n{message}")


def load_modeling_splits(modeling_dir: Path) -> dict[str, pd.DataFrame]:
    """Load normalized modeling splits."""
    splits = {}
    for split_name in ["train", *EVALUATION_SPLITS]:
        path = modeling_dir / f"{split_name}.csv"
        _require_file(path, "Rerun `python3 -m src.features.build_features` to regenerate modeling splits.")
        splits[split_name] = pd.read_csv(path, low_memory=False)
    return splits


def load_evaluation_frames(
    modeling_dir: Path,
    error_analysis_dir: Path,
    calibration_dir: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load validation/test context and attach generated calibrated predictions."""
    _require_file(
        error_analysis_dir / "error_summary.json",
        "Rerun `python3 -m src.models.analyze_errors` before Phase 11A EDA.",
    )
    splits = load_modeling_splits(modeling_dir)
    frames = []
    for split_name in EVALUATION_SPLITS:
        prediction_path = calibration_dir / f"calibrated_two_output_predictions_{split_name}.csv"
        _require_file(
            prediction_path,
            "Rerun `python3 -m src.models.calibrate_risk_models` before Phase 11A EDA.",
        )
        split_df = splits[split_name].copy().reset_index(drop=True)
        predictions = pd.read_csv(prediction_path).reset_index(drop=True)
        if len(split_df) != len(predictions):
            raise ValueError(
                f"{prediction_path} has {len(predictions)} rows, but {split_name}.csv has "
                f"{len(split_df)} rows. Rerun the model pipeline so prediction outputs match splits."
            )
        if "split" in predictions.columns and set(predictions["split"].dropna().unique()) != {split_name}:
            raise ValueError(f"{prediction_path} does not contain only split={split_name!r}.")

        frame = split_df.copy()
        frame["split"] = split_name
        frame["actual"] = pd.to_numeric(predictions["actual_delay"], errors="coerce")
        frame["prediction"] = pd.to_numeric(predictions["predicted_delay_minutes"], errors="coerce")
        frame["error"] = frame["prediction"] - frame["actual"]
        frame["absolute_error"] = frame["error"].abs()
        frame["delay_bucket"] = assign_delay_bucket(frame["actual"])
        for threshold in SEVERE_THRESHOLDS:
            probability_column = f"calibrated_severe_delay_probability_{threshold}"
            prediction_column = f"severe_delay_prediction_{threshold}"
            if probability_column in predictions.columns:
                frame[probability_column] = pd.to_numeric(
                    predictions[probability_column], errors="coerce"
                )
            if prediction_column in predictions.columns:
                frame[prediction_column] = pd.to_numeric(
                    predictions[prediction_column], errors="coerce"
                )
        frames.append(frame)
    return splits, pd.concat(frames, ignore_index=True)


def error_tables(eval_frame: pd.DataFrame, min_group_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = []
    for column in GROUP_COLUMNS:
        min_size = min_group_size if column in HIGH_CARDINALITY_COLUMNS else None
        tables.append(grouped_error_metrics(eval_frame, column, min_size))
    by_group = pd.concat(tables, ignore_index=True)
    contribution = by_group.sort_values(
        ["split", "group_column", "total_error_contribution"],
        ascending=[True, True, False],
        ignore_index=True,
    )
    return by_group, contribution


def severe_delay_by_group(eval_frame: pd.DataFrame, min_group_size: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for threshold in SEVERE_THRESHOLDS:
        probability_column = f"calibrated_severe_delay_probability_{threshold}"
        prediction_column = f"severe_delay_prediction_{threshold}"
        for group_column in SEVERE_GROUP_COLUMNS:
            if group_column not in eval_frame.columns:
                continue
            min_size = min_group_size if group_column in HIGH_CARDINALITY_COLUMNS else None
            for (split_name, group_value), group in eval_frame.groupby(
                ["split", group_column], dropna=False
            ):
                if min_size is not None and len(group) < min_size:
                    continue
                actual_flag = pd.to_numeric(group["actual"], errors="coerce") >= threshold
                actual_rate = float(actual_flag.mean()) if len(group) else np.nan
                mean_probability = (
                    float(pd.to_numeric(group[probability_column], errors="coerce").mean())
                    if probability_column in group.columns
                    else np.nan
                )
                if prediction_column in group.columns:
                    predicted_flag = pd.to_numeric(group[prediction_column], errors="coerce") == 1
                    severe_count = int(actual_flag.sum())
                    false_negative_rate = (
                        float(((actual_flag) & (~predicted_flag)).sum() / severe_count)
                        if severe_count
                        else np.nan
                    )
                else:
                    false_negative_rate = np.nan
                records.append(
                    {
                        "split": split_name,
                        "threshold_minutes": threshold,
                        "group_column": group_column,
                        "group_value": group_value,
                        "row_count": int(len(group)),
                        "actual_severe_delay_rate": actual_rate,
                        "mean_predicted_severe_delay_probability": mean_probability,
                        "probability_calibration_gap": mean_probability - actual_rate
                        if pd.notna(mean_probability)
                        else np.nan,
                        "false_negative_rate": false_negative_rate,
                    }
                )
    return pd.DataFrame.from_records(records).sort_values(
        ["threshold_minutes", "split", "group_column", "actual_severe_delay_rate"],
        ascending=[True, True, True, False],
        ignore_index=True,
    )


def candidate_group_support(full_frame: pd.DataFrame) -> pd.DataFrame:
    full = full_frame.copy().reset_index(drop=True)
    eval_mask = full["split"].isin(EVALUATION_SPLITS)
    groupings = {
        "Route": ["Route"],
        "Incident": ["Incident"],
        "mode": ["mode"],
        "Route + Incident": ["Route", "Incident"],
        "mode + Incident": ["mode", "Incident"],
        "Route + Direction": ["Route", "Direction"],
        "Route + hour": ["Route", "hour"],
        "Route + Incident + hour": ["Route", "Incident", "hour"],
        "Location": ["Location"],
        "Location + Incident": ["Location", "Incident"],
        "Location + hour": ["Location", "hour"],
    }
    records = [
        prior_count_support(full, eval_mask, grouping_name, columns)
        for grouping_name, columns in groupings.items()
    ]
    return pd.DataFrame.from_records(records)


def _fallback_prediction(frame: pd.DataFrame, prior_mean: pd.Series) -> pd.Series:
    fallback_columns = [
        "prior_route_mean_delay",
        "prior_mode_mean_delay",
        "prior_global_mean_delay",
    ]
    filled = prior_mean.copy()
    for column in fallback_columns:
        if column in frame.columns:
            filled = filled.fillna(pd.to_numeric(frame[column], errors="coerce"))
    return filled


def candidate_prior_mean_scores(full_frame: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "prior_route_incident_mean_delay": ["Route", "Incident"],
        "prior_mode_incident_mean_delay": ["mode", "Incident"],
        "prior_route_direction_mean_delay": ["Route", "Direction"],
        "prior_location_mean_delay": ["Location"],
        "prior_location_incident_mean_delay": ["Location", "Incident"],
        "prior_route_incident_hour_mean_delay": ["Route", "Incident", "hour"],
    }
    full = full_frame.copy().reset_index(drop=True)
    eval_mask = full["split"].isin(EVALUATION_SPLITS)
    actual = pd.to_numeric(full.loc[eval_mask, TARGET_COLUMN], errors="coerce")
    records: list[dict[str, Any]] = []
    for feature_name, group_columns in candidates.items():
        try:
            prior_mean = _strict_prior_mean_for_group(full, group_columns)
            eval_prior = prior_mean.loc[eval_mask.reset_index(drop=True)]
            available = eval_prior.notna()
            fallback = _fallback_prediction(full.loc[eval_mask].reset_index(drop=True), eval_prior.reset_index(drop=True))
            records.append(
                {
                    "feature_name": feature_name,
                    "grouping": " + ".join(group_columns),
                    "row_count": int(len(actual)),
                    "coverage_percent": float(available.mean() * 100.0),
                    "mae_where_available": float((eval_prior[available] - actual[available]).abs().mean())
                    if available.any()
                    else np.nan,
                    "fallback_mae": float((fallback - actual.reset_index(drop=True)).abs().mean()),
                    "note": "strict prior-only expanding mean",
                }
            )
        except Exception as exc:
            records.append(
                {
                    "feature_name": feature_name,
                    "grouping": " + ".join(group_columns),
                    "row_count": int(len(actual)),
                    "coverage_percent": np.nan,
                    "mae_where_available": np.nan,
                    "fallback_mae": np.nan,
                    "note": f"skipped: {exc}",
                }
            )
    return pd.DataFrame.from_records(records)


def _rolling_values(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    window: str = "30D",
) -> pd.Series:
    """Compute prior-only time-window means for one grouping."""
    required = [*group_columns, TIMESTAMP_COLUMN, value_column]
    work = frame[required].copy()
    work[TIMESTAMP_COLUMN] = pd.to_datetime(work[TIMESTAMP_COLUMN], errors="coerce")
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work["_row_id"] = np.arange(len(work))
    result = pd.Series(np.nan, index=work.index, dtype="float64")
    for _, group in work.sort_values(TIMESTAMP_COLUMN).groupby(group_columns, dropna=False):
        ordered = group.sort_values(TIMESTAMP_COLUMN, kind="mergesort")
        values = ordered.set_index(TIMESTAMP_COLUMN)[value_column]
        rolled = values.rolling(window, closed="left").mean()
        result.loc[ordered["_row_id"].to_numpy()] = rolled.to_numpy()
    return result.sort_index().reset_index(drop=True)


def rolling_window_opportunity(full_frame: pd.DataFrame) -> pd.DataFrame:
    full = full_frame.copy().reset_index(drop=True)
    eval_mask = full["split"].isin(EVALUATION_SPLITS)
    actual = pd.to_numeric(full.loc[eval_mask, TARGET_COLUMN], errors="coerce").reset_index(drop=True)
    candidates = [
        ("prior_route_30d_mean_delay", "rolling_mean", ["Route"], TARGET_COLUMN, "regression"),
        ("prior_incident_30d_mean_delay", "rolling_mean", ["Incident"], TARGET_COLUMN, "regression"),
        ("prior_route_hour_30d_mean_delay", "rolling_mean", ["Route", "hour"], TARGET_COLUMN, "regression"),
    ]
    for threshold in SEVERE_THRESHOLDS:
        full[f"severe_delay_{threshold}"] = (
            pd.to_numeric(full[TARGET_COLUMN], errors="coerce") >= threshold
        ).astype("int64")
        candidates.extend(
            [
                (
                    f"prior_route_30d_severe_rate_{threshold}",
                    "rolling_rate",
                    ["Route"],
                    f"severe_delay_{threshold}",
                    f"severe_{threshold}",
                ),
                (
                    f"prior_incident_30d_severe_rate_{threshold}",
                    "rolling_rate",
                    ["Incident"],
                    f"severe_delay_{threshold}",
                    f"severe_{threshold}",
                ),
            ]
        )

    records = []
    for feature_name, feature_type, group_columns, value_column, target in candidates:
        try:
            values = _rolling_values(full, group_columns, value_column)
            eval_values = values.loc[eval_mask.reset_index(drop=True)].reset_index(drop=True)
            available = eval_values.notna()
            if feature_type == "rolling_mean":
                score = (
                    float((eval_values[available] - actual[available]).abs().mean())
                    if available.any()
                    else np.nan
                )
                score_name = "mae_where_available"
            else:
                threshold = int(target.split("_")[1])
                y = (actual >= threshold).astype("int64")
                score = (
                    float(np.mean(np.square(eval_values[available] - y[available])))
                    if available.any()
                    else np.nan
                )
                score_name = "brier_score_where_available"
            record = {
                "feature_name": feature_name,
                "feature_type": feature_type,
                "grouping": " + ".join(group_columns),
                "target": target,
                "coverage_percent": float(available.mean() * 100.0),
                "missing_fallback_rate": float((~available).mean() * 100.0),
                "note": "30-day strict prior-only rolling estimate",
            }
            record[score_name] = score
            records.append(record)
        except Exception as exc:
            records.append(
                {
                    "feature_name": feature_name,
                    "feature_type": feature_type,
                    "grouping": " + ".join(group_columns),
                    "target": target,
                    "coverage_percent": np.nan,
                    "missing_fallback_rate": np.nan,
                    "mae_where_available": np.nan,
                    "brier_score_where_available": np.nan,
                    "note": f"skipped: {exc}; use optimized implementation in Phase 11B if needed",
                }
            )
    return pd.DataFrame.from_records(records)


def _apply_support_to_recommendations(
    recommendations: pd.DataFrame,
    support: pd.DataFrame,
    prior_scores: pd.DataFrame,
    rolling_scores: pd.DataFrame,
) -> pd.DataFrame:
    recs = recommendations.copy()
    support_lookup = support.set_index("grouping").to_dict("index")
    score_lookup = prior_scores.set_index("feature_name").to_dict("index")
    rolling_lookup = rolling_scores.set_index("feature_name").to_dict("index")
    enriched_reasons = []
    for _, row in recs.iterrows():
        reason = str(row["reason"])
        support_row = support_lookup.get(row["grouping"])
        if support_row:
            reason += (
                f" Support: {support_row.get('pct_with_prior_20', np.nan):.1f}% of eval rows "
                "have at least 20 prior observations."
            )
            if "Location" in str(row["grouping"]) and support_row.get("pct_with_prior_20", 0) >= 75:
                recs.loc[recs["feature_name"] == row["feature_name"], "recommended_for_phase_11b"] = True
        score_row = score_lookup.get(row["feature_name"]) or rolling_lookup.get(row["feature_name"])
        if score_row and pd.notna(score_row.get("coverage_percent", np.nan)):
            reason += f" EDA coverage: {score_row['coverage_percent']:.1f}%."
        enriched_reasons.append(reason)
    recs["reason"] = enriched_reasons
    return recs.sort_values("priority_rank", ignore_index=True)


def _plot_top_contribution(contribution: pd.DataFrame, group_column: str, output_path: Path, top_n: int) -> None:
    subset = contribution[
        (contribution["split"] == "test") & (contribution["group_column"] == group_column)
    ].head(top_n)
    if subset.empty:
        return
    plt.figure(figsize=(10, 6))
    labels = subset["group_value"].astype(str)
    plt.barh(labels[::-1], subset["total_error_contribution"][::-1])
    plt.xlabel("Total absolute error contribution")
    plt.ylabel(group_column)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _write_markdown(
    output_path: Path,
    summary: dict[str, Any],
    recommendations: pd.DataFrame,
    support: pd.DataFrame,
) -> None:
    top_recs = recommendations[recommendations["recommended_for_phase_11b"]].head(8)
    lines = [
        "# Model Improvement EDA",
        "",
        "Phase 11A diagnostic report for planning historical features. This run does not train models or modify model artifacts.",
        "",
        f"Generated: {summary['generated_at']}",
        f"Validation rows: {summary['validation_rows']}",
        f"Test rows: {summary['test_rows']}",
        "",
        "## Highest-Priority Phase 11B Candidates",
        "",
    ]
    for _, row in top_recs.iterrows():
        lines.append(
            f"{int(row['priority_rank'])}. `{row['feature_name']}` ({row['feature_type']}, {row['target']}): {row['reason']}"
        )
    lines.extend(
        [
            "",
            "## Candidate Support Snapshot",
            "",
            support.sort_values("pct_with_prior_20", ascending=False).head(12).to_string(index=False),
            "",
            "## Output Files",
            "",
            "- `error_by_group.csv`",
            "- `error_contribution_by_group.csv`",
            "- `severe_delay_by_group.csv`",
            "- `candidate_group_support.csv`",
            "- `candidate_prior_mean_scores.csv`",
            "- `rolling_window_opportunity.csv`",
            "- `feature_recommendations.csv`",
            "- `model_improvement_eda_summary.json`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_model_improvement_eda(
    modeling_dir: Path = DEFAULT_MODELING_DIR,
    error_analysis_dir: Path = DEFAULT_ERROR_ANALYSIS_DIR,
    calibration_dir: Path = DEFAULT_CALIBRATION_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Any]:
    """Generate Phase 11A model-improvement EDA reports."""
    splits, eval_frame = load_evaluation_frames(modeling_dir, error_analysis_dir, calibration_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    split_frames = []
    for split_name, frame in splits.items():
        work = frame.copy()
        work["split"] = split_name
        split_frames.append(work)
    full_frame = pd.concat(split_frames, ignore_index=True)

    by_group, contribution = error_tables(eval_frame, min_group_size)
    severe = severe_delay_by_group(eval_frame, min_group_size)
    support = candidate_group_support(full_frame)
    prior_scores = candidate_prior_mean_scores(full_frame)
    rolling_scores = rolling_window_opportunity(full_frame)
    recommendations = _apply_support_to_recommendations(
        recommendation_table(), support, prior_scores, rolling_scores
    )

    by_group.to_csv(output_dir / "error_by_group.csv", index=False)
    contribution.to_csv(output_dir / "error_contribution_by_group.csv", index=False)
    severe.to_csv(output_dir / "severe_delay_by_group.csv", index=False)
    support.to_csv(output_dir / "candidate_group_support.csv", index=False)
    prior_scores.to_csv(output_dir / "candidate_prior_mean_scores.csv", index=False)
    rolling_scores.to_csv(output_dir / "rolling_window_opportunity.csv", index=False)
    recommendations.to_csv(output_dir / "feature_recommendations.csv", index=False)

    _plot_top_contribution(contribution, "Route", figures_dir / "top_error_contribution_by_route.png", top_n)
    _plot_top_contribution(
        contribution, "Incident", figures_dir / "top_error_contribution_by_incident.png", top_n
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Phase 11A EDA only; no model training or artifact modification is performed.",
        "modeling_dir": str(modeling_dir),
        "error_analysis_dir": str(error_analysis_dir),
        "calibration_dir": str(calibration_dir),
        "output_dir": str(output_dir),
        "min_group_size": int(min_group_size),
        "top_n": int(top_n),
        "validation_rows": int((eval_frame["split"] == "validation").sum()),
        "test_rows": int((eval_frame["split"] == "test").sum()),
        "recommended_feature_count": int(recommendations["recommended_for_phase_11b"].sum()),
        "top_recommended_features": recommendations[
            recommendations["recommended_for_phase_11b"]
        ]["feature_name"].head(10).tolist(),
        "outputs": [
            "error_by_group.csv",
            "error_contribution_by_group.csv",
            "severe_delay_by_group.csv",
            "candidate_group_support.csv",
            "candidate_prior_mean_scores.csv",
            "rolling_window_opportunity.csv",
            "feature_recommendations.csv",
            "model_improvement_eda_summary.json",
            "model_improvement_eda.md",
        ],
    }
    (output_dir / "model_improvement_eda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_markdown(output_dir / "model_improvement_eda.md", summary, recommendations, support)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-dir", type=Path, default=DEFAULT_MODELING_DIR)
    parser.add_argument("--error-analysis-dir", type=Path, default=DEFAULT_ERROR_ANALYSIS_DIR)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-group-size", type=int, default=DEFAULT_MIN_GROUP_SIZE)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_model_improvement_eda(
        modeling_dir=args.modeling_dir,
        error_analysis_dir=args.error_analysis_dir,
        calibration_dir=args.calibration_dir,
        output_dir=args.output_dir,
        min_group_size=args.min_group_size,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
