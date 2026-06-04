"""Analyze residuals for the fixed trained delay model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.evaluate_baselines import calculate_metrics, fill_predictions_with_fallbacks
from src.models.train_model import (
    TARGET_COLUMN_FALLBACK,
    feature_groups_from_metadata,
    load_feature_metadata,
    load_modeling_splits,
    split_xy,
)


DEFAULT_MODELING_DIR = Path("data/processed/modeling")
DEFAULT_MODEL_PATH = Path("artifacts/models/xgb_delay_model.joblib")
DEFAULT_BASELINE_REPORT = Path("reports/baselines/baseline_metrics.json")
DEFAULT_OUTPUT_DIR = Path("reports/error_analysis")
EVALUATION_SPLITS = ["validation", "test"]
HIGH_DELAY_THRESHOLDS = [15, 30, 60]
DEFAULT_MIN_GROUP_SIZE = 100
WORST_PREDICTION_COUNT = 100
WORST_PREDICTION_COLUMNS = [
    "split",
    "mode",
    "ts",
    "Route",
    "Direction",
    "Incident",
    "Location",
    "actual",
    "prediction",
    "error",
    "absolute_error",
    "delay_bucket",
]


def assign_delay_bucket(values: pd.Series) -> pd.Series:
    """Assign target delay values to stable reporting buckets."""
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, 5, 10, 15, 30, 60, 120, 240, np.inf],
        labels=["0-5", "6-10", "11-15", "16-30", "31-60", "61-120", "121-240", "241+"],
        right=True,
    ).astype("string")


def load_best_baseline_config(path: Path) -> dict[str, Any] | None:
    """Load the selected baseline configuration if the report exists."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    best_baseline = payload.get("best_baseline")
    if not best_baseline or not best_baseline.get("baseline"):
        return None
    return {
        "baseline": str(best_baseline["baseline"]),
        "filled": bool(best_baseline.get("filled", False)),
        "evaluation_name": best_baseline.get(
            "evaluation_name",
            f"{best_baseline['baseline']}_filled"
            if bool(best_baseline.get("filled", False))
            else best_baseline["baseline"],
        ),
    }


def best_baseline_predictions(
    df: pd.DataFrame,
    baseline_config: dict[str, Any] | None,
    train_target_mean: float,
) -> pd.Series | None:
    """Return selected baseline predictions when the required columns are present."""
    if baseline_config is None:
        return None

    baseline_column = baseline_config["baseline"]
    if baseline_column not in df.columns:
        return None

    if baseline_config.get("filled"):
        return fill_predictions_with_fallbacks(
            df=df,
            baseline_column=baseline_column,
            train_target_mean=train_target_mean,
        )
    return pd.to_numeric(df[baseline_column], errors="coerce")


def prediction_error_frame(
    model: Any,
    df: pd.DataFrame,
    split_name: str,
    feature_columns: list[str],
    target_column: str,
    baseline_predictions: pd.Series | None = None,
) -> pd.DataFrame:
    """Create row-level model predictions and residuals for one split."""
    x_split, y_split = split_xy(df, feature_columns, target_column)
    predictions = pd.Series(model.predict(x_split), index=y_split.index, dtype="float64")

    work = df.loc[y_split.index].copy()
    work["split"] = split_name
    work["actual"] = pd.to_numeric(y_split, errors="coerce")
    work["prediction"] = predictions
    work["error"] = work["prediction"] - work["actual"]
    work["absolute_error"] = work["error"].abs()
    work["squared_error"] = np.square(work["error"])
    work["delay_bucket"] = assign_delay_bucket(work["actual"])

    if baseline_predictions is not None:
        baseline = pd.to_numeric(baseline_predictions.reindex(y_split.index), errors="coerce")
        work["baseline_prediction"] = baseline
        work["baseline_error"] = work["baseline_prediction"] - work["actual"]
        work["baseline_absolute_error"] = work["baseline_error"].abs()

    return work


def residual_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Calculate overall residual metrics for one evaluated frame."""
    metrics = calculate_metrics(frame["actual"], frame["prediction"])
    summary = {
        "split": frame["split"].iloc[0] if not frame.empty else pd.NA,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "mean_error": float(frame["error"].mean()) if not frame.empty else np.nan,
        "median_absolute_error": float(frame["absolute_error"].median()) if not frame.empty else np.nan,
        "p90_absolute_error": float(frame["absolute_error"].quantile(0.90)) if not frame.empty else np.nan,
        "p95_absolute_error": float(frame["absolute_error"].quantile(0.95)) if not frame.empty else np.nan,
        "row_count": int(len(frame)),
    }
    if "baseline_prediction" in frame.columns:
        valid_baseline = frame["baseline_prediction"].notna()
        baseline_metrics = calculate_metrics(
            frame.loc[valid_baseline, "actual"],
            frame.loc[valid_baseline, "baseline_prediction"],
        )
        summary.update(
            {
                "baseline_mae": baseline_metrics["mae"],
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_r2": baseline_metrics["r2"],
                "baseline_rows_evaluated": int(valid_baseline.sum()),
            }
        )
    return summary


def grouped_breakdown(
    frame: pd.DataFrame,
    group_column: str,
    min_group_size: int | None = None,
) -> pd.DataFrame:
    """Calculate residual metrics by one grouping column."""
    if group_column not in frame.columns:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    group_columns = ["split", group_column]
    for group_values, group in frame.groupby(group_columns, dropna=False):
        split_name, group_value = group_values
        if min_group_size is not None and len(group) < min_group_size:
            continue
        metrics = calculate_metrics(group["actual"], group["prediction"])
        records.append(
            {
                "split": split_name,
                group_column: group_value,
                "row_count": int(len(group)),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mean_error": float(group["error"].mean()),
                "median_absolute_error": float(group["absolute_error"].median()),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "split",
                group_column,
                "row_count",
                "mae",
                "rmse",
                "mean_error",
                "median_absolute_error",
            ]
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["split", "mae"],
        ascending=[True, False],
        ignore_index=True,
    )


def high_delay_performance(
    frame: pd.DataFrame,
    thresholds: list[int] | None = None,
) -> pd.DataFrame:
    """Summarize model behavior on actual high-delay rows."""
    thresholds = thresholds or HIGH_DELAY_THRESHOLDS
    records: list[dict[str, Any]] = []

    for split_name, split_frame in frame.groupby("split", dropna=False):
        total_rows = len(split_frame)
        for threshold in thresholds:
            high = split_frame[split_frame["actual"] >= threshold]
            underpredicted = high[high["prediction"] < high["actual"]]
            records.append(
                {
                    "split": split_name,
                    "threshold_minutes": threshold,
                    "actual_high_delay_rows": int(len(high)),
                    "actual_high_delay_rate": float(len(high) / total_rows) if total_rows else np.nan,
                    "mae_high_delay": float(high["absolute_error"].mean()) if len(high) else np.nan,
                    "mean_prediction_high_delay": float(high["prediction"].mean()) if len(high) else np.nan,
                    "mean_actual_high_delay": float(high["actual"].mean()) if len(high) else np.nan,
                    "mean_error_high_delay": float(high["error"].mean()) if len(high) else np.nan,
                    "underpredicted_high_delay_percent": (
                        float(len(underpredicted) / len(high) * 100) if len(high) else np.nan
                    ),
                    "average_underprediction_amount": (
                        float((underpredicted["actual"] - underpredicted["prediction"]).mean())
                        if len(underpredicted)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def worst_predictions(frame: pd.DataFrame, split_name: str, n: int = WORST_PREDICTION_COUNT) -> pd.DataFrame:
    """Return the largest absolute-error predictions for one split."""
    split_frame = frame[frame["split"] == split_name].copy()
    for column in WORST_PREDICTION_COLUMNS:
        if column not in split_frame.columns:
            split_frame[column] = pd.NA
    return (
        split_frame.sort_values("absolute_error", ascending=False)
        .head(n)
        .loc[:, WORST_PREDICTION_COLUMNS]
        .reset_index(drop=True)
    )


def write_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    """Write optional matplotlib figures when matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_frame in frame.groupby("split"):
        plt.figure(figsize=(8, 5))
        split_frame["error"].hist(bins=50)
        plt.title(f"Residuals - {split_name}")
        plt.xlabel("Prediction error")
        plt.ylabel("Row count")
        plt.tight_layout()
        plt.savefig(figures_dir / f"residual_histogram_{split_name}.png", dpi=150)
        plt.close()

        sample = split_frame.sample(n=min(5000, len(split_frame)), random_state=42)
        plt.figure(figsize=(6, 6))
        plt.scatter(sample["actual"], sample["prediction"], s=8, alpha=0.35)
        limit = float(max(sample["actual"].max(), sample["prediction"].max())) if len(sample) else 1.0
        plt.plot([0, limit], [0, limit], color="black", linewidth=1)
        plt.title(f"Actual vs Predicted - {split_name}")
        plt.xlabel("Actual delay")
        plt.ylabel("Predicted delay")
        plt.tight_layout()
        plt.savefig(figures_dir / f"actual_vs_predicted_{split_name}.png", dpi=150)
        plt.close()

    for group_column, file_name in [
        ("mode", "mae_by_mode.png"),
        ("delay_bucket", "mae_by_delay_bucket.png"),
    ]:
        breakdown = grouped_breakdown(frame, group_column)
        if breakdown.empty:
            continue
        labels = breakdown["split"].astype(str) + " / " + breakdown[group_column].astype(str)
        plt.figure(figsize=(10, 5))
        plt.bar(labels, breakdown["mae"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("MAE")
        plt.tight_layout()
        plt.savefig(figures_dir / file_name, dpi=150)
        plt.close()


def _json_safe(value: Any) -> Any:
    """Convert pandas and numpy scalars into strict JSON-compatible values."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_outputs(
    frames: dict[str, pd.DataFrame],
    baseline_config: dict[str, Any] | None,
    output_dir: Path,
    min_group_size: int,
) -> dict[str, pd.DataFrame]:
    """Write all error-analysis reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([frames[split_name] for split_name in EVALUATION_SPLITS], ignore_index=True)

    summaries = pd.DataFrame.from_records(
        [residual_summary(frames[split_name]) for split_name in EVALUATION_SPLITS]
    )
    by_mode = grouped_breakdown(combined, "mode")
    by_route = grouped_breakdown(combined, "Route", min_group_size=min_group_size)
    by_incident = grouped_breakdown(combined, "Incident", min_group_size=min_group_size)
    by_hour = grouped_breakdown(combined, "hour")
    by_month = grouped_breakdown(combined, "month")
    by_delay_bucket = grouped_breakdown(combined, "delay_bucket")
    high_delay = high_delay_performance(combined)
    worst_validation = worst_predictions(combined, "validation")
    worst_test = worst_predictions(combined, "test")

    summaries.to_csv(output_dir / "error_summary.csv", index=False)
    by_mode.to_csv(output_dir / "error_by_mode.csv", index=False)
    by_route.to_csv(output_dir / "error_by_route.csv", index=False)
    by_incident.to_csv(output_dir / "error_by_incident.csv", index=False)
    by_hour.to_csv(output_dir / "error_by_hour.csv", index=False)
    by_month.to_csv(output_dir / "error_by_month.csv", index=False)
    by_delay_bucket.to_csv(output_dir / "error_by_delay_bucket.csv", index=False)
    high_delay.to_csv(output_dir / "high_delay_performance.csv", index=False)
    worst_validation.to_csv(output_dir / "worst_predictions_validation.csv", index=False)
    worst_test.to_csv(output_dir / "worst_predictions_test.csv", index=False)

    payload = {
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "Error analysis only; no model training is performed.",
        "baseline_comparison": baseline_config,
        "overall": summaries.to_dict(orient="records"),
        "high_delay_performance": high_delay.to_dict(orient="records"),
        "min_group_size": min_group_size,
    }
    (output_dir / "error_summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    write_plots(combined, output_dir)
    return {
        "summary": summaries,
        "by_mode": by_mode,
        "by_route": by_route,
        "by_incident": by_incident,
        "by_hour": by_hour,
        "by_month": by_month,
        "by_delay_bucket": by_delay_bucket,
        "high_delay": high_delay,
        "worst_validation": worst_validation,
        "worst_test": worst_test,
    }


def run_error_analysis(
    modeling_dir: Path,
    model_path: Path,
    baseline_report: Path,
    output_dir: Path,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> dict[str, Any]:
    """Run fixed-model residual analysis for validation and test splits."""
    metadata = load_feature_metadata(modeling_dir)
    feature_columns, categorical_columns, _ = feature_groups_from_metadata(metadata)
    target_column = metadata.get("target_column", TARGET_COLUMN_FALLBACK)
    splits = load_modeling_splits(modeling_dir, categorical_columns)
    model = joblib.load(model_path)
    baseline_config = load_best_baseline_config(baseline_report)
    train_target_mean = float(pd.to_numeric(splits["train"][target_column], errors="coerce").mean())

    frames = {}
    for split_name in EVALUATION_SPLITS:
        baseline_predictions = best_baseline_predictions(
            df=splits[split_name],
            baseline_config=baseline_config,
            train_target_mean=train_target_mean,
        )
        frames[split_name] = prediction_error_frame(
            model=model,
            df=splits[split_name],
            split_name=split_name,
            feature_columns=feature_columns,
            target_column=target_column,
            baseline_predictions=baseline_predictions,
        )

    reports = write_outputs(
        frames=frames,
        baseline_config=baseline_config,
        output_dir=output_dir,
        min_group_size=min_group_size,
    )
    return {"frames": frames, "reports": reports, "baseline_config": baseline_config}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze validation/test errors for the fixed trained TTC delay model."
    )
    parser.add_argument("--modeling-dir", type=Path, default=DEFAULT_MODELING_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-group-size", type=int, default=DEFAULT_MIN_GROUP_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_error_analysis(
        modeling_dir=args.modeling_dir,
        model_path=args.model_path,
        baseline_report=args.baseline_report,
        output_dir=args.output_dir,
        min_group_size=args.min_group_size,
    )


if __name__ == "__main__":
    main()
