"""Calibrate Phase 7B severe-delay probabilities for Phase 7C."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.models.run_experiments import regression_metrics, train_log_target_model
from src.models.train_model import (
    TARGET_COLUMN_FALLBACK,
    feature_groups_from_metadata,
    load_feature_metadata,
    load_modeling_splits,
    split_xy,
)
from src.models.train_risk_models import (
    XGB_CLASSIFIER_CONFIG,
    build_classifier_pipeline,
    build_xgb_classifier,
    calculate_scale_pos_weight,
    classification_metrics,
    make_binary_target,
    predict_positive_probability,
    select_operating_threshold,
    threshold_table,
)


DEFAULT_MODELING_DIR = Path("data/processed/modeling")
DEFAULT_PHASE_7B_ARTIFACT_PATH = Path("artifacts/risk_models/two_output_model.joblib")
DEFAULT_SELECTED_REGRESSOR_PATH = Path("artifacts/experiments/selected_experiment.joblib")
DEFAULT_REPORTS_DIR = Path("reports/calibration")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/calibration")
DEFAULT_THRESHOLDS = [30, 60]
CALIBRATION_METHODS = ["uncalibrated", "sigmoid", "isotonic"]
METHOD_SIMPLICITY_ORDER = {"sigmoid": 0, "isotonic": 1, "uncalibrated": 2}
CALIBRATION_CUTOFFS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
CALIBRATED_RISK_BAND_DEFINITIONS = {
    "low": {"min_probability": 0.0, "max_probability": 0.10},
    "medium": {"min_probability": 0.10, "max_probability": 0.30},
    "high": {"min_probability": 0.30, "max_probability": 1.0},
}
EVALUATION_SPLITS = ["calibration", "validation", "test"]


def assign_calibrated_risk_band(probability: float) -> str:
    """Assign calibrated low/medium/high risk bands."""
    if probability < 0.10:
        return "low"
    if probability < 0.30:
        return "medium"
    return "high"


def calibrated_risk_bands(probabilities: pd.Series | np.ndarray) -> pd.Series:
    """Vectorized calibrated risk-band assignment."""
    return pd.Series(probabilities).astype("float64").map(assign_calibrated_risk_band)


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error."""
    table = probability_bin_table(y_true, probabilities, n_bins=n_bins)
    total = int(table["row_count"].sum())
    if total == 0:
        return np.nan
    weighted_error = table["row_count"] * table["absolute_calibration_error"].fillna(0.0)
    return float(weighted_error.sum() / total)


def probability_bin_table(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Create equal-width probability calibration bins."""
    y = pd.Series(y_true).astype("int64").reset_index(drop=True)
    proba = pd.Series(probabilities, dtype="float64").clip(0.0, 1.0).reset_index(drop=True)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    records = []
    for index in range(n_bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == n_bins - 1:
            mask = (proba >= lower) & (proba <= upper)
        else:
            mask = (proba >= lower) & (proba < upper)
        group_y = y[mask]
        group_p = proba[mask]
        mean_probability = float(group_p.mean()) if len(group_p) else np.nan
        actual_rate = float(group_y.mean()) if len(group_y) else np.nan
        records.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "row_count": int(mask.sum()),
                "mean_predicted_probability": mean_probability,
                "actual_severe_delay_rate": actual_rate,
                "absolute_calibration_error": (
                    abs(mean_probability - actual_rate) if len(group_p) else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def split_base_and_calibration_training(
    train_df: pd.DataFrame,
    timestamp_column: str = "ts",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training rows into pre-2022 base fitting rows and 2022 calibration rows."""
    if timestamp_column not in train_df.columns:
        raise ValueError(
            "Cannot create Phase 7C calibration split because the training split has no "
            f"`{timestamp_column}` timestamp column."
        )
    timestamps = pd.to_datetime(train_df[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        bad_count = int(timestamps.isna().sum())
        raise ValueError(
            "Cannot create Phase 7C calibration split because "
            f"{bad_count} training rows have unparseable timestamps."
        )
    base_mask = timestamps < pd.Timestamp("2022-01-01")
    calibration_mask = (timestamps >= pd.Timestamp("2022-01-01")) & (
        timestamps < pd.Timestamp("2023-01-01")
    )
    base_df = train_df.loc[base_mask].copy()
    calibration_df = train_df.loc[calibration_mask].copy()
    if base_df.empty or calibration_df.empty:
        raise ValueError(
            "Cannot create Phase 7C calibration split. Expected non-empty base rows with "
            "`ts < 2022-01-01` and non-empty calibration rows from calendar year 2022."
        )
    return base_df, calibration_df


def _prefit_calibrated_classifier(estimator: Any, method: str) -> CalibratedClassifierCV:
    """Create a version-compatible prefit CalibratedClassifierCV."""
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(estimator=FrozenEstimator(estimator), method=method)
    except ImportError:
        pass
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv="prefit")
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv="prefit")


def train_base_risk_classifier(
    base_train_df: pd.DataFrame,
    metadata: dict[str, Any],
    threshold: int,
    model: Any | None = None,
) -> Any:
    """Train one severe-delay classifier on the base training subset."""
    feature_columns, categorical_columns, numeric_columns = feature_groups_from_metadata(metadata)
    target_column = metadata.get("target_column", TARGET_COLUMN_FALLBACK)
    x_train, y_delay = split_xy(base_train_df, feature_columns, target_column)
    y_binary = make_binary_target(y_delay, threshold)
    if y_binary.nunique() < 2:
        raise ValueError(
            f"Cannot train base classifier for Min Delay >= {threshold}; "
            "the pre-2022 base training subset must contain both classes."
        )
    estimator = model or build_xgb_classifier(calculate_scale_pos_weight(y_binary))
    pipeline = build_classifier_pipeline(categorical_columns, numeric_columns, estimator)
    pipeline.fit(x_train, y_binary)
    return pipeline


def fit_calibration_methods(
    base_classifier: Any,
    calibration_df: pd.DataFrame,
    metadata: dict[str, Any],
    threshold: int,
) -> dict[str, Any]:
    """Fit sigmoid and isotonic calibrators on the 2022 calibration subset."""
    feature_columns, _, _ = feature_groups_from_metadata(metadata)
    target_column = metadata.get("target_column", TARGET_COLUMN_FALLBACK)
    x_calibration, y_delay = split_xy(calibration_df, feature_columns, target_column)
    y_binary = make_binary_target(y_delay, threshold)
    if y_binary.nunique() < 2:
        raise ValueError(
            f"Cannot calibrate classifier for Min Delay >= {threshold}; "
            "the 2022 calibration subset must contain both classes."
        )

    models = {"uncalibrated": base_classifier}
    for method in ["sigmoid", "isotonic"]:
        calibrator = _prefit_calibrated_classifier(copy.deepcopy(base_classifier), method=method)
        calibrator.fit(x_calibration, y_binary)
        models[method] = calibrator
    return models


def metrics_with_ece(
    y_true: pd.Series,
    probabilities: pd.Series | np.ndarray,
    probability_cutoff: float,
) -> dict[str, Any]:
    """Compute classification metrics plus 10-bin ECE."""
    return {
        **classification_metrics(y_true, probabilities, probability_cutoff),
        "expected_calibration_error": expected_calibration_error(y_true, probabilities, n_bins=10),
    }


def risk_band_summary(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    split_name: str,
    threshold: int,
    method: str,
) -> pd.DataFrame:
    """Summarize actual severe-delay rate by calibrated probability risk band."""
    frame = pd.DataFrame(
        {
            "actual_severe_delay": pd.Series(y_true).astype("int64").to_numpy(),
            "predicted_probability": np.asarray(probabilities, dtype="float64"),
        }
    )
    frame["risk_band"] = calibrated_risk_bands(frame["predicted_probability"]).to_numpy()

    records = []
    for band in ["low", "medium", "high"]:
        group = frame[frame["risk_band"] == band]
        records.append(
            {
                "split": split_name,
                "threshold_minutes": int(threshold),
                "calibration_method": method,
                "risk_band": band,
                "row_count": int(len(group)),
                "actual_severe_delay_rate": (
                    float(group["actual_severe_delay"].mean()) if len(group) else np.nan
                ),
                "mean_predicted_probability": (
                    float(group["predicted_probability"].mean()) if len(group) else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def select_calibration_method(validation_metrics: pd.DataFrame) -> dict[str, Any]:
    """Select final calibration method using validation metrics only."""
    if validation_metrics.empty:
        raise ValueError("Cannot select a calibration method from an empty validation table.")
    required = {"calibration_method", "brier_score", "expected_calibration_error", "pr_auc"}
    missing = required - set(validation_metrics.columns)
    if missing:
        raise ValueError(f"Validation metrics are missing required columns: {sorted(missing)}")

    work = validation_metrics.copy()
    min_brier = float(work["brier_score"].min())
    if min_brier == 0:
        candidates = work[work["brier_score"] == min_brier].copy()
    else:
        candidates = work[work["brier_score"] <= min_brier * 1.02].copy()

    min_ece = float(candidates["expected_calibration_error"].min())
    if min_ece == 0:
        candidates = candidates[candidates["expected_calibration_error"] == min_ece].copy()
    else:
        candidates = candidates[candidates["expected_calibration_error"] <= min_ece * 1.02].copy()

    max_pr_auc = float(candidates["pr_auc"].max())
    pr_tolerance = max(1e-12, abs(max_pr_auc) * 0.02)
    candidates = candidates[candidates["pr_auc"] >= max_pr_auc - pr_tolerance].copy()
    candidates["simplicity_rank"] = candidates["calibration_method"].map(METHOD_SIMPLICITY_ORDER)
    selected = candidates.sort_values(
        ["simplicity_rank", "brier_score", "expected_calibration_error", "pr_auc"],
        ascending=[True, True, True, False],
    ).iloc[0]

    return {
        "calibration_method": str(selected["calibration_method"]),
        "selection_rule": (
            "Validation-only: prefer lowest Brier score; if Brier scores are within 2%, "
            "prefer lower ECE; if still close, prefer higher PR-AUC; if still close, "
            "prefer sigmoid, isotonic, then uncalibrated."
        ),
        "validation_brier_score": float(selected["brier_score"]),
        "validation_expected_calibration_error": float(selected["expected_calibration_error"]),
        "validation_pr_auc": float(selected["pr_auc"]),
    }


def load_or_train_expected_delay_regressor(
    phase_7b_artifact_path: Path,
    selected_regressor_path: Path,
    train_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[Any, str]:
    """Load the existing expected-delay regressor or train the documented fallback."""
    if phase_7b_artifact_path.exists():
        artifact = joblib.load(phase_7b_artifact_path)
        if isinstance(artifact, dict) and artifact.get("expected_delay_regressor") is not None:
            return artifact["expected_delay_regressor"], str(phase_7b_artifact_path)
        raise ValueError(
            f"Existing Phase 7B artifact at {phase_7b_artifact_path} does not contain "
            "`expected_delay_regressor`."
        )
    if selected_regressor_path.exists():
        return joblib.load(selected_regressor_path), str(selected_regressor_path)
    return train_log_target_model(train_df, metadata), "trained_combined_xgb_log_target"


def prediction_frame_for_split(
    split_df: pd.DataFrame,
    split_name: str,
    feature_columns: list[str],
    target_column: str,
    regressor: Any,
    classifiers: dict[int, Any],
    selected_methods: dict[str, Any],
    selected_cutoffs: dict[str, Any],
) -> pd.DataFrame:
    """Create row-level calibrated two-output predictions for one split."""
    x_split, y_delay = split_xy(split_df, feature_columns, target_column)
    result = pd.DataFrame(index=y_delay.index)
    result["split"] = split_name
    result["actual_delay"] = pd.to_numeric(y_delay, errors="coerce")
    result["predicted_delay_minutes"] = np.asarray(regressor.predict(x_split), dtype="float64")

    passthrough_map = {
        "mode": ["mode"],
        "route": ["route", "Route"],
        "incident": ["incident", "Incident"],
        "location": ["location", "Location"],
        "timestamp": ["timestamp", "ts", "Date"],
    }
    for output_column, candidates in passthrough_map.items():
        source = next((column for column in candidates if column in split_df.columns), None)
        result[output_column] = split_df.loc[y_delay.index, source] if source else pd.NA

    for threshold, classifier in classifiers.items():
        threshold_key = str(threshold)
        probability = predict_positive_probability(classifier, x_split)
        probability_column = f"calibrated_severe_delay_probability_{threshold}"
        result[probability_column] = probability
        result[f"risk_band_{threshold}"] = calibrated_risk_bands(probability).to_numpy()
        cutoff = float(selected_cutoffs[threshold_key]["probability_cutoff"])
        result[f"severe_delay_prediction_{threshold}"] = (probability >= cutoff).astype("int64")
        result[f"selected_probability_cutoff_{threshold}"] = cutoff
        result[f"calibration_method_{threshold}"] = selected_methods[threshold_key][
            "calibration_method"
        ]
    return result.reset_index(drop=True)


def evaluate_regression_predictions(frame: pd.DataFrame, split_name: str) -> dict[str, Any]:
    """Compute expected-delay metrics from a calibrated two-output prediction frame."""
    work = pd.DataFrame(
        {
            "actual": frame["actual_delay"],
            "prediction": frame["predicted_delay_minutes"],
        }
    )
    work["error"] = work["prediction"] - work["actual"]
    work["absolute_error"] = work["error"].abs()
    return {"split": split_name, **regression_metrics(work)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def build_calibrated_artifact(
    expected_delay_regressor: Any,
    calibrated_risk_classifiers: dict[int, Any],
    all_risk_classifiers: dict[int, dict[str, Any]],
    feature_columns: list[str],
    target_column: str,
    selected_methods: dict[str, Any],
    selected_cutoffs: dict[str, Any],
    metadata: dict[str, Any],
    regressor_source: str,
    split_summary: dict[str, Any],
) -> dict[str, Any]:
    """Package the calibrated two-output artifact for later API integration."""
    return {
        "model_name": "calibrated_two_output_delay_and_risk_model",
        "model_phase": "Phase 7C",
        "expected_delay_regressor": expected_delay_regressor,
        "calibrated_risk_classifiers": calibrated_risk_classifiers,
        "risk_classifiers_by_threshold_and_method": all_risk_classifiers,
        "risk_classifier_30": calibrated_risk_classifiers.get(30),
        "risk_classifier_60": calibrated_risk_classifiers.get(60),
        "selected_calibration_methods": selected_methods,
        "selected_probability_thresholds": selected_cutoffs,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "risk_band_definitions": CALIBRATED_RISK_BAND_DEFINITIONS,
        "metadata": {
            "generated_timestamp": datetime.now(timezone.utc).isoformat(),
            "regressor_source": regressor_source,
            "classifier_config": XGB_CLASSIFIER_CONFIG,
            "feature_metadata": metadata,
            "calibration_split": split_summary,
            "notes": [
                "Base classifiers are trained on training rows before 2022-01-01.",
                "Sigmoid and isotonic calibrators are fit on 2022 training rows only.",
                "Calibration methods and operating cutoffs are selected using validation data only.",
                "Test data is evaluated after calibration choices and cutoffs are fixed.",
                "This artifact does not include Optuna, SHAP, FastAPI, or frontend code.",
            ],
        },
    }


def write_calibration_outputs(
    reports_dir: Path,
    artifacts_dir: Path,
    metrics_df: pd.DataFrame,
    threshold_table_df: pd.DataFrame,
    bin_table_df: pd.DataFrame,
    risk_band_summary_df: pd.DataFrame,
    selection: dict[str, Any],
    prediction_frames: dict[str, pd.DataFrame],
    artifact: dict[str, Any],
) -> None:
    """Write all Phase 7C reports and the calibrated artifact."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(reports_dir / "calibration_metrics.csv", index=False)
    threshold_table_df.to_csv(reports_dir / "calibration_threshold_table.csv", index=False)
    bin_table_df.to_csv(reports_dir / "calibration_bin_table.csv", index=False)
    risk_band_summary_df.to_csv(reports_dir / "calibrated_risk_band_summary.csv", index=False)
    (reports_dir / "calibration_selection.json").write_text(
        json.dumps(_json_safe(selection), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    prediction_frames["validation"].to_csv(
        reports_dir / "calibrated_two_output_predictions_validation.csv",
        index=False,
    )
    prediction_frames["test"].to_csv(
        reports_dir / "calibrated_two_output_predictions_test.csv",
        index=False,
    )

    summary = {
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "Phase 7C calibrated severe-delay probability models.",
        "selection": selection,
        "risk_band_definitions": CALIBRATED_RISK_BAND_DEFINITIONS,
        "output_files": [
            "calibration_metrics.csv",
            "calibration_threshold_table.csv",
            "calibration_bin_table.csv",
            "calibrated_risk_band_summary.csv",
            "calibration_selection.json",
            "calibrated_two_output_predictions_validation.csv",
            "calibrated_two_output_predictions_test.csv",
            "calibrated_two_output_summary.json",
        ],
    }
    (reports_dir / "calibrated_two_output_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    joblib.dump(artifact, artifacts_dir / "calibrated_two_output_model.joblib")


def write_calibration_plots(
    reports_dir: Path,
    bin_table_df: pd.DataFrame,
    selected_methods: dict[str, Any],
) -> None:
    """Write simple calibration curve plots for selected methods on validation and test."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for threshold_key, method_info in selected_methods.items():
        threshold = int(threshold_key)
        method = method_info["calibration_method"]
        for split_name in ["validation", "test"]:
            frame = bin_table_df[
                (bin_table_df["threshold_minutes"] == threshold)
                & (bin_table_df["split"] == split_name)
                & (bin_table_df["calibration_method"] == method)
                & (bin_table_df["row_count"] > 0)
            ]
            if frame.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
            ax.plot(
                frame["mean_predicted_probability"],
                frame["actual_severe_delay_rate"],
                marker="o",
                linewidth=1.5,
            )
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Actual severe-delay rate")
            ax.set_title(f"Threshold {threshold} {split_name} calibration")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(
                figures_dir / f"calibration_curve_threshold_{threshold}_{split_name}.png",
                dpi=150,
            )
            plt.close(fig)


def run_calibration(
    modeling_dir: Path = DEFAULT_MODELING_DIR,
    phase_7b_artifact_path: Path = DEFAULT_PHASE_7B_ARTIFACT_PATH,
    selected_regressor_path: Path = DEFAULT_SELECTED_REGRESSOR_PATH,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    thresholds: list[int] | None = None,
) -> dict[str, Any]:
    """Run the full Phase 7C probability-calibration workflow."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    metadata = load_feature_metadata(modeling_dir)
    feature_columns, categorical_columns, _ = feature_groups_from_metadata(metadata)
    target_column = metadata.get("target_column", TARGET_COLUMN_FALLBACK)
    splits = load_modeling_splits(modeling_dir, categorical_columns)
    base_train_df, calibration_df = split_base_and_calibration_training(splits["train"])
    evaluation_splits = {
        "calibration": calibration_df,
        "validation": splits["validation"],
        "test": splits["test"],
    }

    expected_delay_regressor, regressor_source = load_or_train_expected_delay_regressor(
        phase_7b_artifact_path=phase_7b_artifact_path,
        selected_regressor_path=selected_regressor_path,
        train_df=splits["train"],
        metadata=metadata,
    )

    classifiers_by_threshold: dict[int, dict[str, Any]] = {}
    metrics_records = []
    threshold_tables = []
    bin_tables = []
    risk_band_frames = []
    selected_methods: dict[str, Any] = {}
    selected_cutoffs: dict[str, Any] = {}

    for threshold in thresholds:
        base_classifier = train_base_risk_classifier(base_train_df, metadata, threshold)
        method_classifiers = fit_calibration_methods(
            base_classifier=base_classifier,
            calibration_df=calibration_df,
            metadata=metadata,
            threshold=threshold,
        )
        classifiers_by_threshold[threshold] = method_classifiers

        method_validation_metrics = []
        method_selected_cutoffs: dict[str, Any] = {}
        method_probabilities: dict[str, dict[str, np.ndarray]] = {}
        for method, classifier in method_classifiers.items():
            method_probabilities[method] = {}
            for split_name, split_df in evaluation_splits.items():
                x_split, y_delay = split_xy(split_df, feature_columns, target_column)
                y_binary = make_binary_target(y_delay, threshold)
                method_probabilities[method][split_name] = predict_positive_probability(
                    classifier, x_split
                )

            validation_y = make_binary_target(
                split_xy(splits["validation"], feature_columns, target_column)[1],
                threshold,
            )
            validation_table = threshold_table(
                validation_y,
                method_probabilities[method]["validation"],
                cutoffs=CALIBRATION_CUTOFFS,
            )
            validation_table.insert(0, "threshold_minutes", int(threshold))
            validation_table.insert(1, "calibration_method", method)
            validation_table.insert(2, "split", "validation")
            threshold_tables.append(validation_table)

            selected_cutoff = select_operating_threshold(validation_table)
            method_selected_cutoffs[method] = selected_cutoff
            cutoff = float(selected_cutoff["probability_cutoff"])

            for split_name, split_df in evaluation_splits.items():
                y_delay = split_xy(split_df, feature_columns, target_column)[1]
                y_binary = make_binary_target(y_delay, threshold)
                probabilities = method_probabilities[method][split_name]
                metric = {
                    "split": split_name,
                    "threshold_minutes": int(threshold),
                    "calibration_method": method,
                    "selected_probability_cutoff": cutoff,
                    **metrics_with_ece(y_binary, probabilities, cutoff),
                }
                metrics_records.append(metric)
                if split_name == "validation":
                    method_validation_metrics.append(metric)

                bins = probability_bin_table(y_binary, probabilities, n_bins=10)
                bins.insert(0, "threshold_minutes", int(threshold))
                bins.insert(1, "calibration_method", method)
                bins.insert(2, "split", split_name)
                bin_tables.append(bins)
                risk_band_frames.append(
                    risk_band_summary(
                        y_true=y_binary,
                        probabilities=probabilities,
                        split_name=split_name,
                        threshold=threshold,
                        method=method,
                    )
                )

        validation_metrics_df = pd.DataFrame.from_records(method_validation_metrics)
        method_selection = select_calibration_method(validation_metrics_df)
        selected_method = method_selection["calibration_method"]
        threshold_key = str(threshold)
        selected_methods[threshold_key] = method_selection
        selected_cutoffs[threshold_key] = {
            **method_selected_cutoffs[selected_method],
            "calibration_method": selected_method,
        }

    metrics_df = pd.DataFrame.from_records(metrics_records)
    threshold_table_df = pd.concat(threshold_tables, ignore_index=True)
    bin_table_df = pd.concat(bin_tables, ignore_index=True)
    risk_band_summary_df = pd.concat(risk_band_frames, ignore_index=True)
    calibrated_risk_classifiers = {
        threshold: classifiers_by_threshold[threshold][selected_methods[str(threshold)]["calibration_method"]]
        for threshold in thresholds
    }

    prediction_frames = {
        split_name: prediction_frame_for_split(
            split_df=splits[split_name],
            split_name=split_name,
            feature_columns=feature_columns,
            target_column=target_column,
            regressor=expected_delay_regressor,
            classifiers=calibrated_risk_classifiers,
            selected_methods=selected_methods,
            selected_cutoffs=selected_cutoffs,
        )
        for split_name in ["validation", "test"]
    }
    regression_metrics_df = pd.DataFrame.from_records(
        [
            evaluate_regression_predictions(prediction_frames[split_name], split_name)
            for split_name in ["validation", "test"]
        ]
    )

    split_summary = {
        "base_classifier_train": {
            "rule": "training rows with ts < 2022-01-01",
            "row_count": int(len(base_train_df)),
        },
        "calibration": {
            "rule": "training rows from calendar year 2022",
            "row_count": int(len(calibration_df)),
        },
        "validation": {
            "rule": "existing 2023 validation split; used for method and cutoff selection",
            "row_count": int(len(splits["validation"])),
        },
        "test": {
            "rule": "existing 2024 test split; evaluated after selections are fixed",
            "row_count": int(len(splits["test"])),
        },
    }
    selection = {
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            str(threshold): {
                "selected_calibration_method": selected_methods[str(threshold)],
                "selected_operating_cutoff": selected_cutoffs[str(threshold)],
            }
            for threshold in thresholds
        },
        "split_summary": split_summary,
        "regression_metrics": regression_metrics_df.to_dict(orient="records"),
        "risk_band_definitions": CALIBRATED_RISK_BAND_DEFINITIONS,
    }
    artifact = build_calibrated_artifact(
        expected_delay_regressor=expected_delay_regressor,
        calibrated_risk_classifiers=calibrated_risk_classifiers,
        all_risk_classifiers=classifiers_by_threshold,
        feature_columns=feature_columns,
        target_column=target_column,
        selected_methods=selected_methods,
        selected_cutoffs=selected_cutoffs,
        metadata=metadata,
        regressor_source=regressor_source,
        split_summary=split_summary,
    )
    write_calibration_outputs(
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
        metrics_df=metrics_df,
        threshold_table_df=threshold_table_df,
        bin_table_df=bin_table_df,
        risk_band_summary_df=risk_band_summary_df,
        selection=selection,
        prediction_frames=prediction_frames,
        artifact=artifact,
    )
    write_calibration_plots(reports_dir, bin_table_df, selected_methods)

    return {
        "metrics": metrics_df,
        "threshold_table": threshold_table_df,
        "bin_table": bin_table_df,
        "risk_band_summary": risk_band_summary_df,
        "selection": selection,
        "prediction_frames": prediction_frames,
        "artifact": artifact,
    }


def _parse_thresholds(value: str) -> list[int]:
    thresholds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise argparse.ArgumentTypeError("At least one threshold is required.")
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate Phase 7C severe-delay probability models."
    )
    parser.add_argument("--modeling-dir", type=Path, default=DEFAULT_MODELING_DIR)
    parser.add_argument(
        "--phase-7b-artifact-path",
        "--two-output-artifact",
        type=Path,
        default=DEFAULT_PHASE_7B_ARTIFACT_PATH,
        dest="phase_7b_artifact_path",
        help="Path to the Phase 7B two-output artifact.",
    )
    parser.add_argument(
        "--selected-regressor-path",
        type=Path,
        default=DEFAULT_SELECTED_REGRESSOR_PATH,
    )
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--thresholds", type=_parse_thresholds, default=DEFAULT_THRESHOLDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_calibration(
        modeling_dir=args.modeling_dir,
        phase_7b_artifact_path=args.phase_7b_artifact_path,
        selected_regressor_path=args.selected_regressor_path,
        reports_dir=args.reports_dir,
        artifacts_dir=args.artifacts_dir,
        thresholds=args.thresholds,
    )


if __name__ == "__main__":
    main()
