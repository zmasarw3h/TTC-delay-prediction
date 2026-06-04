"""Evaluate leakage-safe historical baseline predictors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.build_features import MAIN_CATEGORICAL_FEATURES, TARGET_COLUMN


DEFAULT_MODELING_DIR = Path("data/processed/modeling")
DEFAULT_OUTPUT_DIR = Path("reports/baselines")

BASELINE_COLUMNS = [
    "prior_global_mean_delay",
    "prior_mode_mean_delay",
    "prior_route_mean_delay",
    "prior_route_hour_mean_delay",
    "prior_incident_mean_delay",
    "prior_route_hour_7d_mean_delay",
]
FALLBACK_COLUMNS = [
    "prior_route_mean_delay",
    "prior_mode_mean_delay",
    "prior_global_mean_delay",
]
SPLIT_FILES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}


def load_split(path: Path) -> pd.DataFrame:
    """Load a modeling split with stable categorical dtypes."""
    dtype = {column: "string" for column in MAIN_CATEGORICAL_FEATURES}
    return pd.read_csv(path, dtype=dtype)


def load_modeling_splits(modeling_dir: Path) -> dict[str, pd.DataFrame]:
    """Load train, validation, and test splits from a modeling directory."""
    return {
        split_name: load_split(modeling_dir / file_name)
        for split_name, file_name in SPLIT_FILES.items()
    }


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calculate regression metrics without external ML dependencies."""
    y_true_numeric = pd.to_numeric(y_true, errors="coerce")
    y_pred_numeric = pd.to_numeric(y_pred, errors="coerce")
    valid_mask = y_true_numeric.notna() & y_pred_numeric.notna()

    if not valid_mask.any():
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}

    errors = y_pred_numeric[valid_mask] - y_true_numeric[valid_mask]
    mae = errors.abs().mean()
    rmse = np.sqrt(np.square(errors).mean())

    centered = y_true_numeric[valid_mask] - y_true_numeric[valid_mask].mean()
    ss_tot = np.square(centered).sum()
    ss_res = np.square(errors).sum()
    r2 = np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)

    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def fill_predictions_with_fallbacks(
    df: pd.DataFrame,
    baseline_column: str,
    train_target_mean: float,
) -> pd.Series:
    """Fill missing baseline predictions using documented historical fallbacks."""
    prediction = pd.to_numeric(df[baseline_column], errors="coerce").copy()
    for fallback_column in FALLBACK_COLUMNS:
        prediction = prediction.fillna(pd.to_numeric(df[fallback_column], errors="coerce"))
    return prediction.fillna(train_target_mean)


def evaluate_prediction(
    df: pd.DataFrame,
    split_name: str,
    baseline_column: str,
    predictions: pd.Series,
    missing_before_fallback: int,
    filled: bool,
) -> dict[str, Any]:
    """Evaluate one baseline prediction series for one split."""
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    prediction = pd.to_numeric(predictions, errors="coerce")
    target_valid = target.notna()
    evaluated_mask = target_valid & prediction.notna()
    row_count = int(target_valid.sum())
    missing_percent = (
        float(missing_before_fallback / row_count * 100) if row_count else 0.0
    )
    metrics = calculate_metrics(target[evaluated_mask], prediction[evaluated_mask])

    return {
        "baseline": baseline_column,
        "evaluation_name": (
            f"{baseline_column}_filled" if filled else baseline_column
        ),
        "split": split_name,
        "filled": bool(filled),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "rows_evaluated": int(evaluated_mask.sum()),
        "missing_predictions_before_fallback": int(missing_before_fallback),
        "missing_predictions_before_fallback_percent": missing_percent,
    }


def evaluate_baselines(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Evaluate configured baseline predictors on validation and test splits."""
    train_target_mean = float(
        pd.to_numeric(splits["train"][TARGET_COLUMN], errors="coerce").mean()
    )
    records: list[dict[str, Any]] = []

    for split_name in ["validation", "test"]:
        df = splits[split_name]
        target_valid = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").notna()

        for baseline_column in BASELINE_COLUMNS:
            raw_prediction = pd.to_numeric(df[baseline_column], errors="coerce")
            missing_before_fallback = int((target_valid & raw_prediction.isna()).sum())
            records.append(
                evaluate_prediction(
                    df=df,
                    split_name=split_name,
                    baseline_column=baseline_column,
                    predictions=raw_prediction,
                    missing_before_fallback=missing_before_fallback,
                    filled=False,
                )
            )

            if missing_before_fallback:
                filled_prediction = fill_predictions_with_fallbacks(
                    df=df,
                    baseline_column=baseline_column,
                    train_target_mean=train_target_mean,
                )
                records.append(
                    evaluate_prediction(
                        df=df,
                        split_name=split_name,
                        baseline_column=baseline_column,
                        predictions=filled_prediction,
                        missing_before_fallback=missing_before_fallback,
                        filled=True,
                    )
                )

    return pd.DataFrame.from_records(records)


def select_best_baseline(metrics: pd.DataFrame) -> dict[str, Any]:
    """Select the best validation baseline by MAE, then RMSE, then R2."""
    validation = metrics[metrics["split"] == "validation"].copy()
    if validation.empty:
        raise ValueError("No validation metrics are available for baseline selection.")

    usable = validation.sort_values(
        ["baseline", "filled"],
        ascending=[True, False],
    ).drop_duplicates(subset=["baseline"], keep="first")
    usable = usable.dropna(subset=["mae", "rmse"])
    if usable.empty:
        raise ValueError("No usable validation baseline metrics are available.")

    best = usable.sort_values(
        ["mae", "rmse", "r2"],
        ascending=[True, True, False],
    ).iloc[0]
    return best.to_dict()


def evaluate_breakdown(
    df: pd.DataFrame,
    split_name: str,
    baseline_column: str,
    filled: bool,
    train_target_mean: float,
    group_column: str,
) -> pd.DataFrame:
    """Evaluate the selected baseline by a categorical group."""
    if filled:
        prediction = fill_predictions_with_fallbacks(
            df=df,
            baseline_column=baseline_column,
            train_target_mean=train_target_mean,
        )
    else:
        prediction = pd.to_numeric(df[baseline_column], errors="coerce")

    work = df[[TARGET_COLUMN, group_column]].copy()
    work["_prediction"] = prediction
    records: list[dict[str, Any]] = []
    for group_value, group in work.groupby(group_column, dropna=False):
        target = pd.to_numeric(group[TARGET_COLUMN], errors="coerce")
        group_prediction = pd.to_numeric(group["_prediction"], errors="coerce")
        valid_mask = target.notna() & group_prediction.notna()
        metrics = calculate_metrics(target[valid_mask], group_prediction[valid_mask])
        records.append(
            {
                "split": split_name,
                group_column: group_value,
                "baseline": baseline_column,
                "filled": bool(filled),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "rows_evaluated": int(valid_mask.sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def best_baseline_by_mode(
    splits: dict[str, pd.DataFrame],
    best_baseline: dict[str, Any],
) -> pd.DataFrame:
    """Compute validation and test mode breakdowns for the selected baseline."""
    train_target_mean = float(
        pd.to_numeric(splits["train"][TARGET_COLUMN], errors="coerce").mean()
    )
    frames = [
        evaluate_breakdown(
            df=splits[split_name],
            split_name=split_name,
            baseline_column=str(best_baseline["baseline"]),
            filled=bool(best_baseline["filled"]),
            train_target_mean=train_target_mean,
            group_column="mode",
        )
        for split_name in ["validation", "test"]
    ]
    return pd.concat(frames, ignore_index=True)


def write_baseline_reports(
    metrics: pd.DataFrame,
    mode_breakdown: pd.DataFrame,
    best_baseline: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write baseline reports as JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "baseline_metrics.csv", index=False)
    mode_breakdown.to_csv(output_dir / "best_baseline_by_mode.csv", index=False)

    payload = {
        "target_column": TARGET_COLUMN,
        "baseline_columns": BASELINE_COLUMNS,
        "fallback_policy": [
            "baseline prediction",
            "prior_route_mean_delay",
            "prior_mode_mean_delay",
            "prior_global_mean_delay",
            "train target mean",
        ],
        "best_baseline": best_baseline,
        "metrics": metrics.to_dict(orient="records"),
    }
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def run_evaluation(modeling_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run the full baseline evaluation workflow."""
    splits = load_modeling_splits(modeling_dir)
    metrics = evaluate_baselines(splits)
    best_baseline = select_best_baseline(metrics)
    mode_breakdown = best_baseline_by_mode(splits, best_baseline)
    write_baseline_reports(
        metrics=metrics,
        mode_breakdown=mode_breakdown,
        best_baseline=best_baseline,
        output_dir=output_dir,
    )
    return {
        "metrics": metrics,
        "best_baseline": best_baseline,
        "mode_breakdown": mode_breakdown,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate historical TTC delay baseline predictors."
    )
    parser.add_argument("--modeling-dir", type=Path, default=DEFAULT_MODELING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(modeling_dir=args.modeling_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
