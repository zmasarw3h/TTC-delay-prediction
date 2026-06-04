"""Train and evaluate the first fixed-configuration XGBoost delay model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.evaluate_baselines import calculate_metrics


DEFAULT_MODELING_DIR = Path("data/processed/modeling")
DEFAULT_REPORTS_DIR = Path("reports/models")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_BASELINE_REPORT = Path("reports/baselines/baseline_metrics.json")
MODEL_NAME = "xgb_delay_regressor_fixed_v1"
MODEL_NOTE = "First fixed-configuration XGBoost model; not Optuna-tuned."
TARGET_COLUMN_FALLBACK = "Min Delay"
LEAKAGE_AND_NON_FEATURE_COLUMNS = {
    "Min Gap",
    "Min Delay",
    "severe_delay_15",
    "ts",
    "Date",
    "Vehicle",
    "source_file",
    "source_sheet",
}
XGB_CONFIG = {
    "objective": "reg:squarederror",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1,
}


def load_feature_metadata(modeling_dir: Path) -> dict[str, Any]:
    """Load modeling feature metadata."""
    metadata_path = modeling_dir / "feature_metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def feature_columns_from_metadata(metadata: dict[str, Any]) -> list[str]:
    """Return approved feature columns after applying explicit exclusions."""
    excluded = set(metadata.get("excluded_columns", [])) | LEAKAGE_AND_NON_FEATURE_COLUMNS
    return [
        column
        for column in metadata["feature_columns"]
        if column not in excluded
    ]


def feature_groups_from_metadata(
    metadata: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """Return all, categorical, and numeric feature columns from metadata."""
    feature_columns = feature_columns_from_metadata(metadata)
    feature_set = set(feature_columns)
    categorical_columns = [
        column
        for column in metadata.get("categorical_columns", [])
        if column in feature_set
    ]
    numeric_columns = [
        column
        for column in metadata.get("numeric_columns", [])
        if column in feature_set
    ]
    assigned = set(categorical_columns) | set(numeric_columns)
    numeric_columns.extend(column for column in feature_columns if column not in assigned)
    return feature_columns, categorical_columns, numeric_columns


def load_split(path: Path, categorical_columns: list[str]) -> pd.DataFrame:
    """Load one split with stable dtypes for categorical features."""
    dtype = {column: "string" for column in categorical_columns}
    return pd.read_csv(path, dtype=dtype)


def load_modeling_splits(
    modeling_dir: Path,
    categorical_columns: list[str],
) -> dict[str, pd.DataFrame]:
    """Load train, validation, and test modeling splits."""
    return {
        "train": load_split(modeling_dir / "train.csv", categorical_columns),
        "validation": load_split(modeling_dir / "validation.csv", categorical_columns),
        "test": load_split(modeling_dir / "test.csv", categorical_columns),
    }


def _one_hot_encoder() -> OneHotEncoder:
    """Create a version-compatible OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> ColumnTransformer:
    """Build preprocessing for metadata-approved feature columns."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    missing_values=pd.NA,
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            ("onehot", _one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_xgb_regressor(config: dict[str, Any] | None = None) -> Any:
    """Create the fixed XGBoost regressor, importing XGBoost only when needed."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "xgboost is required to train the model. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    return XGBRegressor(**(config or XGB_CONFIG))


def build_model_pipeline(
    categorical_columns: list[str],
    numeric_columns: list[str],
    model: Any | None = None,
) -> Pipeline:
    """Build the preprocessing and model pipeline."""
    estimator = model if model is not None else build_xgb_regressor()
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_columns, numeric_columns)),
            ("model", estimator),
        ]
    )


def _valid_target_mask(df: pd.DataFrame, target_column: str) -> pd.Series:
    return pd.to_numeric(df[target_column], errors="coerce").notna()


def split_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return X and y rows with valid target values."""
    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    valid_target = _valid_target_mask(df, target_column)
    return (
        df.loc[valid_target, feature_columns].copy(),
        pd.to_numeric(df.loc[valid_target, target_column], errors="coerce"),
    )


def train_pipeline(
    train_df: pd.DataFrame,
    metadata: dict[str, Any],
    model: Any | None = None,
) -> Pipeline:
    """Fit the model pipeline on training data only."""
    feature_columns, categorical_columns, numeric_columns = feature_groups_from_metadata(metadata)
    target_column = metadata.get("target_column", TARGET_COLUMN_FALLBACK)
    pipeline = build_model_pipeline(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        model=model,
    )
    x_train, y_train = split_xy(train_df, feature_columns, target_column)
    pipeline.fit(x_train, y_train)
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    df: pd.DataFrame,
    split_name: str,
    feature_columns: list[str],
    target_column: str,
) -> dict[str, Any]:
    """Evaluate a fitted model pipeline on one split."""
    x_split, y_split = split_xy(df, feature_columns, target_column)
    predictions = pd.Series(pipeline.predict(x_split), index=y_split.index)
    metrics = calculate_metrics(y_split, predictions)
    return {
        "model": MODEL_NAME,
        "split": split_name,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "rows_evaluated": int(len(y_split)),
    }


def evaluate_model_by_mode(
    pipeline: Pipeline,
    df: pd.DataFrame,
    split_name: str,
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    """Evaluate a fitted pipeline by mode for one split."""
    x_split, y_split = split_xy(df, feature_columns, target_column)
    predictions = pd.Series(pipeline.predict(x_split), index=y_split.index)
    work = df.loc[y_split.index, ["mode"]].copy()
    work[target_column] = y_split
    work["_prediction"] = predictions

    records: list[dict[str, Any]] = []
    for mode, group in work.groupby("mode", dropna=False):
        metrics = calculate_metrics(group[target_column], group["_prediction"])
        records.append(
            {
                "model": MODEL_NAME,
                "split": split_name,
                "mode": mode,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "rows_evaluated": int(len(group)),
            }
        )
    return pd.DataFrame.from_records(records)


def load_best_baseline_metrics(path: Path) -> dict[str, dict[str, float]] | None:
    """Load Phase 6A best baseline MAE by split if the report exists."""
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload.get("best_baseline", {})
    baseline = best.get("baseline")
    filled = bool(best.get("filled", False))
    comparisons: dict[str, dict[str, float]] = {}

    for record in payload.get("metrics", []):
        if record.get("baseline") != baseline:
            continue
        if bool(record.get("filled", False)) != filled:
            continue
        split_name = record.get("split")
        mae = record.get("mae")
        if split_name in {"validation", "test"} and mae is not None:
            comparisons[split_name] = {
                "baseline_name": record.get("evaluation_name", baseline),
                "baseline_mae": float(mae),
            }
    return comparisons or None


def add_baseline_comparison(
    metrics: list[dict[str, Any]],
    baseline_metrics: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float | str]] | None:
    """Calculate MAE difference versus the Phase 6A best baseline."""
    if not baseline_metrics:
        return None

    comparison: dict[str, dict[str, float | str]] = {}
    for record in metrics:
        split_name = record["split"]
        baseline = baseline_metrics.get(split_name)
        if not baseline:
            continue
        baseline_mae = baseline["baseline_mae"]
        model_mae = float(record["mae"])
        comparison[split_name] = {
            "baseline_name": baseline["baseline_name"],
            "baseline_mae": baseline_mae,
            "model_mae": model_mae,
            "mae_improvement": baseline_mae - model_mae,
            "mae_improvement_percent": (
                (baseline_mae - model_mae) / baseline_mae * 100
                if baseline_mae
                else np.nan
            ),
        }
    return comparison or None


def add_baseline_columns_to_metrics(
    metrics: pd.DataFrame,
    baseline_comparison: dict[str, dict[str, float | str]] | None,
) -> pd.DataFrame:
    """Add baseline comparison fields to the tabular metrics report."""
    enriched = metrics.copy()
    enriched["baseline_name"] = pd.NA
    enriched["baseline_mae"] = np.nan
    enriched["mae_improvement"] = np.nan
    enriched["mae_improvement_percent"] = np.nan

    if not baseline_comparison:
        return enriched

    for split_name, comparison in baseline_comparison.items():
        split_mask = enriched["split"] == split_name
        enriched.loc[split_mask, "baseline_name"] = comparison["baseline_name"]
        enriched.loc[split_mask, "baseline_mae"] = comparison["baseline_mae"]
        enriched.loc[split_mask, "mae_improvement"] = comparison["mae_improvement"]
        enriched.loc[split_mask, "mae_improvement_percent"] = comparison[
            "mae_improvement_percent"
        ]
    return enriched


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
    pipeline: Pipeline,
    metrics: pd.DataFrame,
    metrics_by_mode: pd.DataFrame,
    metadata: dict[str, Any],
    model_config: dict[str, Any],
    baseline_comparison: dict[str, dict[str, float | str]] | None,
    reports_dir: Path,
    artifacts_dir: Path,
) -> None:
    """Write model reports and the local trained model artifact."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(reports_dir / "model_metrics.csv", index=False)
    metrics_by_mode.to_csv(reports_dir / "model_metrics_by_mode.csv", index=False)
    joblib.dump(pipeline, artifacts_dir / "xgb_delay_model.joblib")

    payload = {
        "model_name": MODEL_NAME,
        "model_config": model_config,
        "feature_columns": feature_columns_from_metadata(metadata),
        "target_column": metadata.get("target_column", TARGET_COLUMN_FALLBACK),
        "validation_metrics": (
            metrics[metrics["split"] == "validation"].iloc[0].to_dict()
        ),
        "test_metrics": metrics[metrics["split"] == "test"].iloc[0].to_dict(),
        "baseline_comparison": baseline_comparison,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "note": MODEL_NOTE,
    }
    (reports_dir / "model_metrics.json").write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def train_and_evaluate(
    splits: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
    baseline_report_path: Path | None = None,
    model: Any | None = None,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train on train split, evaluate validation and test once, and compare baselines."""
    feature_columns, _, _ = feature_groups_from_metadata(metadata)
    target_column = metadata.get("target_column", TARGET_COLUMN_FALLBACK)
    pipeline = train_pipeline(splits["train"], metadata, model=model)

    metric_records = [
        evaluate_model(pipeline, splits["validation"], "validation", feature_columns, target_column),
        evaluate_model(pipeline, splits["test"], "test", feature_columns, target_column),
    ]
    metrics = pd.DataFrame.from_records(metric_records)
    metrics_by_mode = pd.concat(
        [
            evaluate_model_by_mode(
                pipeline,
                splits[split_name],
                split_name,
                feature_columns,
                target_column,
            )
            for split_name in ["validation", "test"]
        ],
        ignore_index=True,
    )
    baseline_metrics = (
        load_best_baseline_metrics(baseline_report_path)
        if baseline_report_path is not None
        else None
    )
    baseline_comparison = add_baseline_comparison(metric_records, baseline_metrics)
    metrics = add_baseline_columns_to_metrics(metrics, baseline_comparison)
    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "metrics_by_mode": metrics_by_mode,
        "baseline_comparison": baseline_comparison,
        "model_config": model_config or XGB_CONFIG,
    }


def run_training(
    modeling_dir: Path,
    reports_dir: Path,
    artifacts_dir: Path,
    baseline_report_path: Path,
) -> dict[str, Any]:
    """Run the full fixed-configuration model training workflow."""
    metadata = load_feature_metadata(modeling_dir)
    _, categorical_columns, _ = feature_groups_from_metadata(metadata)
    splits = load_modeling_splits(modeling_dir, categorical_columns)
    result = train_and_evaluate(
        splits=splits,
        metadata=metadata,
        baseline_report_path=baseline_report_path,
        model_config=XGB_CONFIG,
    )
    write_outputs(
        pipeline=result["pipeline"],
        metrics=result["metrics"],
        metrics_by_mode=result["metrics_by_mode"],
        metadata=metadata,
        model_config=result["model_config"],
        baseline_comparison=result["baseline_comparison"],
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the first fixed XGBoost TTC delay model."
    )
    parser.add_argument("--modeling-dir", type=Path, default=DEFAULT_MODELING_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        modeling_dir=args.modeling_dir,
        reports_dir=args.reports_dir,
        artifacts_dir=args.artifacts_dir,
        baseline_report_path=args.baseline_report,
    )


if __name__ == "__main__":
    main()
