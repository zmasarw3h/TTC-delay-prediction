"""Train Phase 7B two-output delay and severe-delay risk models."""

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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.models.run_experiments import (
    LogTargetRegressor,
    regression_metrics,
    train_log_target_model,
)
from src.models.train_model import (
    TARGET_COLUMN_FALLBACK,
    build_preprocessor,
    feature_columns_from_metadata,
    feature_groups_from_metadata,
    load_feature_metadata,
    load_modeling_splits,
    split_xy,
)


DEFAULT_MODELING_DIR = Path("data/processed/modeling")
DEFAULT_SELECTED_REGRESSOR_PATH = Path("artifacts/experiments/selected_experiment.joblib")
DEFAULT_REPORTS_DIR = Path("reports/risk_models")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/risk_models")
DEFAULT_THRESHOLDS = [30, 60]
PROBABILITY_CUTOFFS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
RISK_BAND_DEFINITIONS = {
    "low": {"min_probability": 0.0, "max_probability": 0.20},
    "medium": {"min_probability": 0.20, "max_probability": 0.50},
    "high": {"min_probability": 0.50, "max_probability": 1.0},
}
XGB_CLASSIFIER_CONFIG = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1,
}
EVALUATION_SPLITS = ["validation", "test"]


def make_binary_target(delays: pd.Series, threshold: int) -> pd.Series:
    """Create a severe-delay binary target for the requested minute threshold."""
    return (pd.to_numeric(delays, errors="coerce") >= threshold).astype("int64")


def calculate_scale_pos_weight(y_binary: pd.Series) -> float:
    """Return negative_count / positive_count for XGBoost class imbalance handling."""
    positives = int((y_binary == 1).sum())
    negatives = int((y_binary == 0).sum())
    if positives == 0:
        raise ValueError("Cannot train risk classifier because the training split has no positives.")
    return float(negatives / positives)


def build_xgb_classifier(scale_pos_weight: float, config: dict[str, Any] | None = None) -> Any:
    """Create a fixed XGBoost classifier, importing XGBoost only when needed."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required to train severe-delay risk models. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    model_config = copy.deepcopy(config or XGB_CLASSIFIER_CONFIG)
    model_config["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**model_config)


def build_classifier_pipeline(
    categorical_columns: list[str],
    numeric_columns: list[str],
    model: Any,
) -> Pipeline:
    """Build a preprocessing and binary-classification pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_columns, numeric_columns)),
            ("model", model),
        ]
    )


def train_risk_classifier(
    train_df: pd.DataFrame,
    metadata: dict[str, Any],
    threshold: int,
    model: Any | None = None,
) -> Pipeline:
    """Train one severe-delay classifier for a fixed target threshold."""
    feature_columns, categorical_columns, numeric_columns = feature_groups_from_metadata(metadata)
    target_column = metadata.get("target_column", TARGET_COLUMN_FALLBACK)
    x_train, y_delay = split_xy(train_df, feature_columns, target_column)
    y_binary = make_binary_target(y_delay, threshold)
    if y_binary.nunique() < 2:
        raise ValueError(
            f"Cannot train risk classifier for Min Delay >= {threshold}; "
            "the training split must contain both classes."
        )
    estimator = model or build_xgb_classifier(calculate_scale_pos_weight(y_binary))
    pipeline = build_classifier_pipeline(categorical_columns, numeric_columns, estimator)
    pipeline.fit(x_train, y_binary)
    return pipeline


def predict_positive_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for a fitted classifier."""
    probabilities = model.predict_proba(x)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Classifier must expose predict_proba with a positive-class column.")
    return np.asarray(probabilities[:, 1], dtype="float64")


def assign_risk_band(probability: float) -> str:
    """Assign the fixed low/medium/high risk band for a probability."""
    if probability < 0.20:
        return "low"
    if probability < 0.50:
        return "medium"
    return "high"


def risk_bands(probabilities: pd.Series | np.ndarray) -> pd.Series:
    """Vectorized risk-band assignment."""
    return pd.Series(probabilities).astype("float64").map(assign_risk_band)


def classification_metrics(
    y_true: pd.Series,
    probabilities: pd.Series | np.ndarray,
    probability_cutoff: float,
) -> dict[str, Any]:
    """Compute binary classification metrics at one operating cutoff."""
    y = pd.Series(y_true).astype("int64")
    proba = pd.Series(probabilities, index=y.index, dtype="float64")
    predicted = (proba >= probability_cutoff).astype("int64")
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()

    try:
        roc_auc = float(roc_auc_score(y, proba)) if y.nunique() == 2 else np.nan
    except ValueError:
        roc_auc = np.nan
    try:
        pr_auc = float(average_precision_score(y, proba)) if y.nunique() == 2 else np.nan
    except ValueError:
        pr_auc = np.nan
    try:
        loss = float(log_loss(y, proba, labels=[0, 1]))
    except ValueError:
        loss = np.nan

    return {
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": loss,
        "brier_score": float(brier_score_loss(y, proba)) if len(y) else np.nan,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "row_count": int(len(y)),
    }


def threshold_table(
    y_true: pd.Series,
    probabilities: pd.Series | np.ndarray,
    cutoffs: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate candidate operating cutoffs for validation-only threshold selection."""
    records = []
    for cutoff in cutoffs or PROBABILITY_CUTOFFS:
        records.append(
            {
                "probability_cutoff": float(cutoff),
                **classification_metrics(y_true, probabilities, cutoff),
            }
        )
    return pd.DataFrame.from_records(records)


def select_operating_threshold(table: pd.DataFrame, min_recall: float = 0.70) -> dict[str, Any]:
    """Select the operating probability threshold using validation metrics only."""
    if table.empty:
        raise ValueError("Cannot select an operating threshold from an empty table.")
    candidates = table[table["recall"] >= min_recall].copy()
    selection_pool = candidates if not candidates.empty else table.copy()
    selected = selection_pool.sort_values(
        ["f1", "recall", "probability_cutoff"],
        ascending=[False, False, False],
    ).iloc[0]
    return {
        "probability_cutoff": float(selected["probability_cutoff"]),
        "selection_rule": (
            f"Validation-only: prefer recall >= {min_recall:.2f}; among qualifying "
            "cutoffs choose highest F1; otherwise choose highest F1."
        ),
        "met_min_recall": bool(selected["recall"] >= min_recall),
        "validation_recall": float(selected["recall"]),
        "validation_f1": float(selected["f1"]),
    }


def risk_band_summary(
    y_true: pd.Series,
    probabilities: pd.Series | np.ndarray,
    split_name: str,
    threshold: int,
) -> pd.DataFrame:
    """Summarize actual severe-delay rate by fixed probability risk band."""
    frame = pd.DataFrame(
        {
            "actual_severe_delay": pd.Series(y_true).astype("int64").to_numpy(),
            "predicted_probability": np.asarray(probabilities, dtype="float64"),
        }
    )
    frame["risk_band"] = risk_bands(frame["predicted_probability"]).to_numpy()

    records = []
    for band in ["low", "medium", "high"]:
        group = frame[frame["risk_band"] == band]
        records.append(
            {
                "split": split_name,
                "threshold_minutes": int(threshold),
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


def prediction_frame_for_split(
    split_df: pd.DataFrame,
    split_name: str,
    feature_columns: list[str],
    target_column: str,
    regressor: Any,
    classifiers: dict[int, Any],
) -> pd.DataFrame:
    """Create the row-level two-output prediction frame for one split."""
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
        probability = predict_positive_probability(classifier, x_split)
        probability_column = f"severe_delay_probability_{threshold}"
        band_column = f"risk_band_{threshold}"
        result[probability_column] = probability
        result[band_column] = risk_bands(probability).to_numpy()
    return result.reset_index(drop=True)


def evaluate_regression_predictions(frame: pd.DataFrame, split_name: str) -> dict[str, Any]:
    """Compute expected-delay metrics from a two-output prediction frame."""
    work = pd.DataFrame(
        {
            "actual": frame["actual_delay"],
            "prediction": frame["predicted_delay_minutes"],
        }
    )
    work["error"] = work["prediction"] - work["actual"]
    work["absolute_error"] = work["error"].abs()
    return {"split": split_name, **regression_metrics(work)}


def load_or_train_expected_delay_regressor(
    selected_regressor_path: Path,
    train_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[Any, str]:
    """Load the Phase 7A selected model if present, otherwise train log-target XGBoost."""
    if selected_regressor_path.exists():
        return joblib.load(selected_regressor_path), str(selected_regressor_path)
    return train_log_target_model(train_df, metadata), "trained_combined_xgb_log_target"


def build_two_output_artifact(
    expected_delay_regressor: Any,
    risk_classifiers: dict[int, Any],
    feature_columns: list[str],
    target_column: str,
    selected_thresholds: dict[str, Any],
    metadata: dict[str, Any],
    regressor_source: str,
) -> dict[str, Any]:
    """Package the final two-output artifact for later service integration."""
    return {
        "model_name": "two_output_delay_and_risk_model",
        "model_phase": "Phase 7B",
        "expected_delay_regressor": expected_delay_regressor,
        "risk_classifiers": risk_classifiers,
        "risk_classifier_30": risk_classifiers.get(30),
        "risk_classifier_60": risk_classifiers.get(60),
        "feature_columns": feature_columns,
        "target_column": target_column,
        "selected_probability_thresholds": selected_thresholds,
        "risk_band_definitions": RISK_BAND_DEFINITIONS,
        "metadata": {
            "generated_timestamp": datetime.now(timezone.utc).isoformat(),
            "regressor_source": regressor_source,
            "classifier_config": XGB_CLASSIFIER_CONFIG,
            "feature_metadata": metadata,
            "notes": [
                "Thresholds were selected using validation data only.",
                "Test data is evaluated after operating thresholds are fixed.",
                "This artifact does not include Optuna, SHAP, FastAPI, or frontend code.",
            ],
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_outputs(
    reports_dir: Path,
    artifacts_dir: Path,
    regression_metrics_df: pd.DataFrame,
    classification_metrics_df: pd.DataFrame,
    threshold_table_df: pd.DataFrame,
    selected_thresholds: dict[str, Any],
    risk_band_summary_df: pd.DataFrame,
    prediction_frames: dict[str, pd.DataFrame],
    artifact: dict[str, Any],
) -> None:
    """Write all Phase 7B reports and the two-output artifact."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    regression_metrics_df.to_csv(reports_dir / "regression_metrics.csv", index=False)
    classification_metrics_df.to_csv(reports_dir / "classification_metrics.csv", index=False)
    threshold_table_df.to_csv(reports_dir / "classification_threshold_table.csv", index=False)
    (reports_dir / "selected_classification_thresholds.json").write_text(
        json.dumps(_json_safe(selected_thresholds), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    risk_band_summary_df.to_csv(reports_dir / "risk_band_summary.csv", index=False)
    prediction_frames["validation"].to_csv(
        reports_dir / "two_output_predictions_validation.csv",
        index=False,
    )
    prediction_frames["test"].to_csv(reports_dir / "two_output_predictions_test.csv", index=False)

    summary = {
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "Phase 7B two-output model; not Optuna tuning and not SHAP explainability.",
        "regression_metrics": regression_metrics_df.to_dict(orient="records"),
        "classification_metrics": classification_metrics_df.to_dict(orient="records"),
        "selected_probability_thresholds": selected_thresholds,
        "risk_band_definitions": RISK_BAND_DEFINITIONS,
        "output_files": [
            "regression_metrics.csv",
            "classification_metrics.csv",
            "classification_threshold_table.csv",
            "selected_classification_thresholds.json",
            "risk_band_summary.csv",
            "two_output_predictions_validation.csv",
            "two_output_predictions_test.csv",
            "two_output_summary.json",
        ],
    }
    (reports_dir / "two_output_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    joblib.dump(artifact, artifacts_dir / "two_output_model.joblib")


def run_risk_model_training(
    modeling_dir: Path = DEFAULT_MODELING_DIR,
    selected_regressor_path: Path = DEFAULT_SELECTED_REGRESSOR_PATH,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    thresholds: list[int] | None = None,
) -> dict[str, Any]:
    """Run the full Phase 7B two-output modeling workflow."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    metadata = load_feature_metadata(modeling_dir)
    feature_columns, categorical_columns, _ = feature_groups_from_metadata(metadata)
    target_column = metadata.get("target_column", TARGET_COLUMN_FALLBACK)
    splits = load_modeling_splits(modeling_dir, categorical_columns)

    expected_delay_regressor, regressor_source = load_or_train_expected_delay_regressor(
        selected_regressor_path=selected_regressor_path,
        train_df=splits["train"],
        metadata=metadata,
    )
    risk_classifiers = {
        threshold: train_risk_classifier(splits["train"], metadata, threshold)
        for threshold in thresholds
    }

    prediction_frames = {
        split_name: prediction_frame_for_split(
            split_df=splits[split_name],
            split_name=split_name,
            feature_columns=feature_columns,
            target_column=target_column,
            regressor=expected_delay_regressor,
            classifiers=risk_classifiers,
        )
        for split_name in EVALUATION_SPLITS
    }

    regression_metrics_df = pd.DataFrame.from_records(
        [
            evaluate_regression_predictions(prediction_frames[split_name], split_name)
            for split_name in EVALUATION_SPLITS
        ]
    )

    threshold_tables = []
    selected_thresholds: dict[str, Any] = {}
    classification_records = []
    risk_band_frames = []

    for threshold in thresholds:
        validation_frame = prediction_frames["validation"]
        validation_y = make_binary_target(validation_frame["actual_delay"], threshold)
        probability_column = f"severe_delay_probability_{threshold}"
        validation_table = threshold_table(validation_y, validation_frame[probability_column])
        validation_table.insert(0, "threshold_minutes", int(threshold))
        validation_table.insert(1, "split", "validation")
        threshold_tables.append(validation_table)

        selected = select_operating_threshold(validation_table)
        selected_thresholds[str(threshold)] = selected
        operating_cutoff = selected["probability_cutoff"]

        for split_name in EVALUATION_SPLITS:
            frame = prediction_frames[split_name]
            y_binary = make_binary_target(frame["actual_delay"], threshold)
            metrics = classification_metrics(y_binary, frame[probability_column], operating_cutoff)
            classification_records.append(
                {
                    "split": split_name,
                    "threshold_minutes": int(threshold),
                    "selected_probability_cutoff": operating_cutoff,
                    **metrics,
                }
            )
            risk_band_frames.append(
                risk_band_summary(
                    y_true=y_binary,
                    probabilities=frame[probability_column],
                    split_name=split_name,
                    threshold=threshold,
                )
            )

    classification_metrics_df = pd.DataFrame.from_records(classification_records)
    threshold_table_df = pd.concat(threshold_tables, ignore_index=True)
    risk_band_summary_df = pd.concat(risk_band_frames, ignore_index=True)
    artifact = build_two_output_artifact(
        expected_delay_regressor=expected_delay_regressor,
        risk_classifiers=risk_classifiers,
        feature_columns=feature_columns,
        target_column=target_column,
        selected_thresholds=selected_thresholds,
        metadata=metadata,
        regressor_source=regressor_source,
    )
    write_outputs(
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
        regression_metrics_df=regression_metrics_df,
        classification_metrics_df=classification_metrics_df,
        threshold_table_df=threshold_table_df,
        selected_thresholds=selected_thresholds,
        risk_band_summary_df=risk_band_summary_df,
        prediction_frames=prediction_frames,
        artifact=artifact,
    )
    return {
        "regression_metrics": regression_metrics_df,
        "classification_metrics": classification_metrics_df,
        "threshold_table": threshold_table_df,
        "selected_thresholds": selected_thresholds,
        "risk_band_summary": risk_band_summary_df,
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
        description="Train Phase 7B two-output delay and severe-delay risk models."
    )
    parser.add_argument("--modeling-dir", type=Path, default=DEFAULT_MODELING_DIR)
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
    run_risk_model_training(
        modeling_dir=args.modeling_dir,
        selected_regressor_path=args.selected_regressor_path,
        reports_dir=args.reports_dir,
        artifacts_dir=args.artifacts_dir,
        thresholds=args.thresholds,
    )


if __name__ == "__main__":
    main()
