"""Run fixed Phase 7A model-improvement experiments."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.evaluate_baselines import calculate_metrics
from src.models.train_model import (
    TARGET_COLUMN_FALLBACK,
    XGB_CONFIG,
    build_model_pipeline,
    build_xgb_regressor,
    feature_columns_from_metadata,
    feature_groups_from_metadata,
    load_feature_metadata,
    load_modeling_splits,
    split_xy,
)


DEFAULT_MODELING_DIR = Path("data/processed/modeling")
DEFAULT_BASELINE_REPORT = Path("reports/baselines/baseline_metrics.json")
DEFAULT_FIXED_MODEL_REPORT = Path("reports/models/model_metrics.json")
DEFAULT_REPORTS_DIR = Path("reports/experiments")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/experiments")
DEFAULT_SELECTION_METRIC = "validation_mae"
EVALUATION_SPLITS = ["validation", "test"]
HIGH_DELAY_THRESHOLDS = [15, 30, 60]
EXPERIMENT_ORDER = [
    "combined_xgb_fixed",
    "combined_xgb_weighted",
    "combined_xgb_log_target",
    "mode_specific_xgb",
]
SIMPLE_MODEL_RANK = {name: rank for rank, name in enumerate(EXPERIMENT_ORDER)}
WEIGHT_BUCKETS = [
    (0, 15, 1.0),
    (16, 30, 1.5),
    (31, 60, 2.0),
    (61, 120, 3.0),
    (121, 240, 4.0),
]


@dataclass
class ExperimentDefinition:
    """A fixed experiment configuration."""

    name: str
    strategy: str
    description: str


class LogTargetRegressor:
    """Fit a regression pipeline on log1p(target) and predict on the original scale."""

    def __init__(self, pipeline: Any, prediction_min: float = 0.0, prediction_max: float = 240.0):
        self.pipeline = pipeline
        self.prediction_min = prediction_min
        self.prediction_max = prediction_max

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "LogTargetRegressor":
        self.pipeline.fit(x_train, np.log1p(pd.to_numeric(y_train, errors="coerce")))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        predictions = np.expm1(self.pipeline.predict(x))
        return np.clip(predictions, self.prediction_min, self.prediction_max)


class ModeSpecificRegressor:
    """Route each row to a mode-specific fitted model."""

    def __init__(self, models: dict[str, Any], mode_column: str = "mode"):
        self.models = models
        self.mode_column = mode_column

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.mode_column not in x.columns:
            raise ValueError(f"Mode-specific prediction requires column '{self.mode_column}'.")

        predictions = pd.Series(index=x.index, dtype="float64")
        for mode, model in self.models.items():
            mask = x[self.mode_column].astype("string") == mode
            if mask.any():
                predictions.loc[mask] = model.predict(x.loc[mask])

        missing_modes = sorted(
            set(x.loc[predictions.isna(), self.mode_column].astype("string").dropna())
        )
        if missing_modes:
            raise ValueError(f"No mode-specific model is available for modes: {missing_modes}")
        return predictions.to_numpy()


def delay_sample_weights(delays: pd.Series) -> pd.Series:
    """Assign fixed sample weights by target delay bucket."""
    numeric = pd.to_numeric(delays, errors="coerce")
    weights = pd.Series(1.0, index=delays.index, dtype="float64")
    for lower, upper, weight in WEIGHT_BUCKETS:
        weights.loc[(numeric >= lower) & (numeric <= upper)] = weight
    return weights


def _fresh_xgb_config() -> dict[str, Any]:
    return copy.deepcopy(XGB_CONFIG)


def _build_pipeline(metadata: dict[str, Any]) -> Any:
    _, categorical_columns, numeric_columns = feature_groups_from_metadata(metadata)
    return build_model_pipeline(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        model=build_xgb_regressor(_fresh_xgb_config()),
    )


def train_combined_model(
    train_df: pd.DataFrame,
    metadata: dict[str, Any],
    weighted: bool = False,
) -> Any:
    """Train a single fixed XGBoost pipeline."""
    feature_columns, _, _ = feature_groups_from_metadata(metadata)
    target_column = metadata.get("target_column", TARGET_COLUMN_FALLBACK)
    x_train, y_train = split_xy(train_df, feature_columns, target_column)
    pipeline = _build_pipeline(metadata)
    if weighted:
        pipeline.fit(x_train, y_train, model__sample_weight=delay_sample_weights(y_train))
    else:
        pipeline.fit(x_train, y_train)
    return pipeline


def train_log_target_model(train_df: pd.DataFrame, metadata: dict[str, Any]) -> LogTargetRegressor:
    """Train a single fixed XGBoost pipeline on log1p target values."""
    feature_columns, _, _ = feature_groups_from_metadata(metadata)
    target_column = metadata.get("target_column", TARGET_COLUMN_FALLBACK)
    x_train, y_train = split_xy(train_df, feature_columns, target_column)
    model = LogTargetRegressor(_build_pipeline(metadata))
    return model.fit(x_train, y_train)


def train_mode_specific_model(
    train_df: pd.DataFrame,
    metadata: dict[str, Any],
    required_modes: tuple[str, ...] = ("bus", "streetcar"),
) -> ModeSpecificRegressor:
    """Train one fixed XGBoost pipeline per required mode."""
    if "mode" not in train_df.columns:
        raise ValueError("Mode-specific experiment requires a 'mode' column.")

    feature_columns, _, _ = feature_groups_from_metadata(metadata)
    target_column = metadata.get("target_column", TARGET_COLUMN_FALLBACK)
    models: dict[str, Any] = {}
    train_modes = set(train_df["mode"].astype("string").dropna())
    missing_modes = sorted(set(required_modes) - train_modes)
    if missing_modes:
        raise ValueError(f"Cannot train mode-specific models; missing modes: {missing_modes}")

    for mode in required_modes:
        mode_df = train_df[train_df["mode"].astype("string") == mode]
        x_train, y_train = split_xy(mode_df, feature_columns, target_column)
        models[mode] = _build_pipeline(metadata).fit(x_train, y_train)
    return ModeSpecificRegressor(models=models)


def experiment_definitions() -> list[ExperimentDefinition]:
    """Return the fixed Phase 7A experiment set."""
    return [
        ExperimentDefinition("combined_xgb_fixed", "combined", "Phase 6B fixed XGBoost reference."),
        ExperimentDefinition("combined_xgb_weighted", "weighted", "Fixed XGBoost with target-bucket sample weights."),
        ExperimentDefinition("combined_xgb_log_target", "log_target", "Fixed XGBoost trained on log1p target."),
        ExperimentDefinition("mode_specific_xgb", "mode_specific", "Separate fixed XGBoost pipelines by mode."),
    ]


def train_experiment(
    definition: ExperimentDefinition,
    train_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> Any:
    """Train one fixed experiment."""
    if definition.strategy == "combined":
        return train_combined_model(train_df, metadata, weighted=False)
    if definition.strategy == "weighted":
        return train_combined_model(train_df, metadata, weighted=True)
    if definition.strategy == "log_target":
        return train_log_target_model(train_df, metadata)
    if definition.strategy == "mode_specific":
        return train_mode_specific_model(train_df, metadata)
    raise ValueError(f"Unknown experiment strategy: {definition.strategy}")


def prediction_frame(
    model: Any,
    df: pd.DataFrame,
    split_name: str,
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    """Create row-level predictions and residual fields for one split."""
    x_split, y_split = split_xy(df, feature_columns, target_column)
    predictions = pd.Series(model.predict(x_split), index=y_split.index, dtype="float64")
    frame = df.loc[y_split.index].copy()
    frame["split"] = split_name
    frame["actual"] = pd.to_numeric(y_split, errors="coerce")
    frame["prediction"] = predictions
    frame["error"] = frame["prediction"] - frame["actual"]
    frame["absolute_error"] = frame["error"].abs()
    return frame


def regression_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute the required regression metrics for an evaluated frame."""
    metrics = calculate_metrics(frame["actual"], frame["prediction"])
    return {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "mean_error": float(frame["error"].mean()) if len(frame) else np.nan,
        "median_absolute_error": float(frame["absolute_error"].median()) if len(frame) else np.nan,
        "p90_absolute_error": float(frame["absolute_error"].quantile(0.90)) if len(frame) else np.nan,
        "p95_absolute_error": float(frame["absolute_error"].quantile(0.95)) if len(frame) else np.nan,
        "row_count": int(len(frame)),
    }


def evaluate_frames(experiment_name: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Evaluate overall metrics for validation and test frames."""
    records = []
    for split_name in EVALUATION_SPLITS:
        records.append(
            {
                "experiment": experiment_name,
                "split": split_name,
                **regression_metrics(frames[split_name]),
            }
        )
    return pd.DataFrame.from_records(records)


def evaluate_by_mode(experiment_name: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Evaluate metrics by mode for validation and test frames."""
    records = []
    for split_name, frame in frames.items():
        if "mode" not in frame.columns:
            continue
        for mode, group in frame.groupby("mode", dropna=False):
            records.append(
                {
                    "experiment": experiment_name,
                    "split": split_name,
                    "mode": mode,
                    **regression_metrics(group),
                }
            )
    return pd.DataFrame.from_records(records)


def evaluate_high_delay(
    experiment_name: str,
    frames: dict[str, pd.DataFrame],
    thresholds: list[int] | None = None,
) -> pd.DataFrame:
    """Evaluate high-delay rows and underprediction behavior."""
    thresholds = thresholds or HIGH_DELAY_THRESHOLDS
    records = []
    for split_name, frame in frames.items():
        total_rows = len(frame)
        for threshold in thresholds:
            high = frame[frame["actual"] >= threshold]
            underpredicted = high[high["prediction"] < high["actual"]]
            records.append(
                {
                    "experiment": experiment_name,
                    "split": split_name,
                    "threshold_minutes": threshold,
                    "actual_high_delay_rows": int(len(high)),
                    "actual_high_delay_rate": float(len(high) / total_rows) if total_rows else np.nan,
                    "mae_high_delay": float(high["absolute_error"].mean()) if len(high) else np.nan,
                    "rmse_high_delay": (
                        float(np.sqrt(np.square(high["error"]).mean())) if len(high) else np.nan
                    ),
                    "mean_error_high_delay": float(high["error"].mean()) if len(high) else np.nan,
                    "underprediction_percent": (
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


def _comparison_mae_from_baseline_report(path: Path) -> dict[str, float | str] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload.get("best_baseline", {})
    baseline = best.get("baseline")
    filled = bool(best.get("filled", False))
    result: dict[str, float | str] = {}
    for record in payload.get("metrics", []):
        if record.get("baseline") == baseline and bool(record.get("filled", False)) == filled:
            split_name = record.get("split")
            if split_name in EVALUATION_SPLITS and record.get("mae") is not None:
                result[f"{split_name}_mae"] = float(record["mae"])
                result["name"] = record.get("evaluation_name", baseline)
    return result or None


def _comparison_mae_from_fixed_report(path: Path) -> dict[str, float | str] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, float | str] = {"name": payload.get("model_name", "phase_6b_fixed_model")}
    for split_name in EVALUATION_SPLITS:
        metrics = payload.get(f"{split_name}_metrics", {})
        if metrics.get("mae") is not None:
            result[f"{split_name}_mae"] = float(metrics["mae"])
    return result if len(result) > 1 else None


def add_comparison_columns(
    metrics: pd.DataFrame,
    baseline_report: Path,
    fixed_model_report: Path,
) -> pd.DataFrame:
    """Add optional Phase 6A and Phase 6B comparison columns."""
    enriched = metrics.copy()
    baseline = _comparison_mae_from_baseline_report(baseline_report)
    fixed = _comparison_mae_from_fixed_report(fixed_model_report)
    enriched["phase6a_baseline_name"] = baseline["name"] if baseline else pd.NA
    enriched["phase6a_baseline_mae"] = np.nan
    enriched["phase6b_fixed_model_name"] = fixed["name"] if fixed else pd.NA
    enriched["phase6b_fixed_model_mae"] = np.nan
    for split_name in EVALUATION_SPLITS:
        split_mask = enriched["split"] == split_name
        if baseline and f"{split_name}_mae" in baseline:
            enriched.loc[split_mask, "phase6a_baseline_mae"] = baseline[f"{split_name}_mae"]
        if fixed and f"{split_name}_mae" in fixed:
            enriched.loc[split_mask, "phase6b_fixed_model_mae"] = fixed[f"{split_name}_mae"]
    return enriched


def _metric_name(selection_metric: str) -> str:
    return selection_metric.removeprefix("validation_")


def _metric_ascending(metric_name: str) -> bool:
    return metric_name != "r2"


def select_best_experiment(
    metrics: pd.DataFrame,
    high_delay_metrics: pd.DataFrame,
    by_mode_metrics: pd.DataFrame,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
) -> dict[str, Any]:
    """Select the best experiment using validation metrics only."""
    metric_name = _metric_name(selection_metric)
    validation = metrics[metrics["split"] == "validation"].copy()
    if metric_name not in validation.columns:
        raise ValueError(f"Selection metric is not available: {selection_metric}")

    validation = validation.dropna(subset=[metric_name])
    if validation.empty:
        raise ValueError("No usable validation metrics are available for experiment selection.")

    ascending = _metric_ascending(metric_name)
    best_value = validation[metric_name].min() if ascending else validation[metric_name].max()
    if ascending:
        threshold = best_value * 1.01
        candidates = validation[validation[metric_name] <= threshold].copy()
    else:
        threshold = best_value * 0.99
        candidates = validation[validation[metric_name] >= threshold].copy()

    high30 = high_delay_metrics[
        (high_delay_metrics["split"] == "validation")
        & (high_delay_metrics["threshold_minutes"] == 30)
    ][["experiment", "mae_high_delay"]]
    streetcar = by_mode_metrics[
        (by_mode_metrics["split"] == "validation")
        & (by_mode_metrics["mode"].astype("string") == "streetcar")
    ][["experiment", "mae"]].rename(columns={"mae": "streetcar_mae"})
    candidates = candidates.merge(high30, on="experiment", how="left")
    candidates = candidates.merge(streetcar, on="experiment", how="left")
    candidates["simple_model_rank"] = candidates["experiment"].map(SIMPLE_MODEL_RANK)

    selected = candidates.sort_values(
        ["mae_high_delay", "streetcar_mae", "simple_model_rank", metric_name],
        ascending=[True, True, True, ascending],
        na_position="last",
    ).iloc[0]
    return {
        "selected_experiment": selected["experiment"],
        "selection_metric": selection_metric,
        "validation_best_metric_value": float(best_value),
        "close_candidate_threshold": float(threshold),
        "candidate_experiments_within_1_percent": candidates["experiment"].tolist(),
        "selected_validation_metrics": selected.replace({np.nan: None}).to_dict(),
        "rationale": (
            "Selected using validation data only: first keep experiments within 1% of the "
            f"best {selection_metric}, then prefer lower validation MAE for Min Delay >= 30, "
            "then lower streetcar validation MAE, then the predefined simpler-model order."
        ),
    }


def _json_safe(value: Any) -> Any:
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
    selected_model: Any,
    metrics: pd.DataFrame,
    by_mode_metrics: pd.DataFrame,
    high_delay_metrics: pd.DataFrame,
    selection: dict[str, Any],
    metadata: dict[str, Any],
    reports_dir: Path,
    artifacts_dir: Path,
) -> None:
    """Write reports and the selected experiment artifact only."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(reports_dir / "experiment_metrics.csv", index=False)
    by_mode_metrics.to_csv(reports_dir / "experiment_metrics_by_mode.csv", index=False)
    high_delay_metrics.to_csv(reports_dir / "experiment_high_delay_metrics.csv", index=False)
    (reports_dir / "experiment_selection.json").write_text(
        json.dumps(_json_safe(selection), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "Phase 7A fixed model-improvement experiments; not Optuna tuning.",
        "experiments": [definition.__dict__ for definition in experiment_definitions()],
        "feature_columns": feature_columns_from_metadata(metadata),
        "target_column": metadata.get("target_column", TARGET_COLUMN_FALLBACK),
        "xgb_config": XGB_CONFIG,
        "selection": selection,
    }
    (reports_dir / "experiment_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    joblib.dump(selected_model, artifacts_dir / "selected_experiment.joblib")


def run_experiments(
    modeling_dir: Path = DEFAULT_MODELING_DIR,
    baseline_report: Path = DEFAULT_BASELINE_REPORT,
    fixed_model_report: Path = DEFAULT_FIXED_MODEL_REPORT,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
) -> dict[str, Any]:
    """Run all fixed experiments, select by validation metrics, and write outputs."""
    metadata = load_feature_metadata(modeling_dir)
    feature_columns, categorical_columns, _ = feature_groups_from_metadata(metadata)
    target_column = metadata.get("target_column", TARGET_COLUMN_FALLBACK)
    splits = load_modeling_splits(modeling_dir, categorical_columns)

    trained_models: dict[str, Any] = {}
    metric_frames: list[pd.DataFrame] = []
    by_mode_frames: list[pd.DataFrame] = []
    high_delay_frames: list[pd.DataFrame] = []

    for definition in experiment_definitions():
        model = train_experiment(definition, splits["train"], metadata)
        trained_models[definition.name] = model
        frames = {
            split_name: prediction_frame(
                model=model,
                df=splits[split_name],
                split_name=split_name,
                feature_columns=feature_columns,
                target_column=target_column,
            )
            for split_name in EVALUATION_SPLITS
        }
        metric_frames.append(evaluate_frames(definition.name, frames))
        by_mode_frames.append(evaluate_by_mode(definition.name, frames))
        high_delay_frames.append(evaluate_high_delay(definition.name, frames))

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics = add_comparison_columns(metrics, baseline_report, fixed_model_report)
    by_mode_metrics = pd.concat(by_mode_frames, ignore_index=True)
    high_delay_metrics = pd.concat(high_delay_frames, ignore_index=True)
    selection = select_best_experiment(
        metrics=metrics,
        high_delay_metrics=high_delay_metrics,
        by_mode_metrics=by_mode_metrics,
        selection_metric=selection_metric,
    )
    selected_model = trained_models[selection["selected_experiment"]]
    write_outputs(
        selected_model=selected_model,
        metrics=metrics,
        by_mode_metrics=by_mode_metrics,
        high_delay_metrics=high_delay_metrics,
        selection=selection,
        metadata=metadata,
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
    )
    return {
        "metrics": metrics,
        "metrics_by_mode": by_mode_metrics,
        "high_delay_metrics": high_delay_metrics,
        "selection": selection,
        "selected_model": selected_model,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed Phase 7A model experiments.")
    parser.add_argument("--modeling-dir", type=Path, default=DEFAULT_MODELING_DIR)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--fixed-model-report", type=Path, default=DEFAULT_FIXED_MODEL_REPORT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--selection-metric", default=DEFAULT_SELECTION_METRIC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(
        modeling_dir=args.modeling_dir,
        baseline_report=args.baseline_report,
        fixed_model_report=args.fixed_model_report,
        reports_dir=args.reports_dir,
        artifacts_dir=args.artifacts_dir,
        selection_metric=args.selection_metric,
    )


if __name__ == "__main__":
    main()
