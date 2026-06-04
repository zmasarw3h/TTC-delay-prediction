"""Explain the calibrated two-output delay and risk model without retraining."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, roc_auc_score

from src.models.calibrate_risk_models import calibrated_risk_bands
from src.models.train_model import (
    LEAKAGE_AND_NON_FEATURE_COLUMNS,
    TARGET_COLUMN_FALLBACK,
    feature_groups_from_metadata,
    load_feature_metadata,
    load_modeling_splits,
    split_xy,
)
from src.models.train_risk_models import make_binary_target, predict_positive_probability


DEFAULT_MODELING_DIR = Path("data/processed/modeling")
DEFAULT_ARTIFACT_PATH = Path("artifacts/calibration/calibrated_two_output_model.joblib")
DEFAULT_OUTPUT_DIR = Path("reports/explainability")
DEFAULT_SPLIT = "validation"
DEFAULT_MAX_ROWS = 5000
DEFAULT_RANDOM_STATE = 42
DEFAULT_TOP_N_FEATURES = 30
RISK_THRESHOLDS = [30, 60]
LOCAL_EXAMPLE_COUNT = 5
PERMUTATION_REPEATS = 5
LEAKAGE_SENSITIVE_COLUMNS = {
    "Min Gap",
    "Min Delay",
    "severe_delay_15",
    "ts",
    "Date",
    "Vehicle",
    "source_file",
    "source_sheet",
}
PERMUTATION_COLUMNS = [
    "model_output",
    "feature",
    "importance_mean",
    "importance_std",
    "rank",
    "scoring_method",
    "split",
    "rows_used",
]
LOCAL_EXAMPLE_COLUMNS = [
    "split",
    "example_type",
    "row_index",
    "mode",
    "route",
    "incident",
    "location",
    "timestamp",
    "actual_delay",
    "predicted_delay_minutes",
    "calibrated_severe_delay_probability_30",
    "risk_band_30",
    "calibrated_severe_delay_probability_60",
    "risk_band_60",
    "absolute_error",
    "important_feature_values",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def approved_feature_columns(metadata: dict[str, Any], artifact: dict[str, Any] | None = None) -> list[str]:
    """Return metadata/artifact-approved model inputs with leakage fields removed."""
    metadata_features, _, _ = feature_groups_from_metadata(metadata)
    metadata_feature_set = set(metadata_features)
    artifact_features = []
    if isinstance(artifact, dict):
        artifact_features = list(artifact.get("feature_columns") or [])

    candidates = artifact_features or metadata_features
    excluded = (
        set(metadata.get("excluded_columns", []))
        | set(metadata.get("leakage_sensitive_columns", []))
        | LEAKAGE_AND_NON_FEATURE_COLUMNS
        | LEAKAGE_SENSITIVE_COLUMNS
    )
    return [
        column
        for column in candidates
        if column in metadata_feature_set and column not in excluded
    ]


def deterministic_sample(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    """Sample rows reproducibly while preserving all rows when under the limit."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=random_state).sort_index().copy()


def _predict_delay(regressor: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(regressor.predict(x), dtype="float64")


def _regression_scorer(estimator: Any, x: pd.DataFrame, y: pd.Series | np.ndarray) -> float:
    predictions = _predict_delay(estimator, x)
    return -float(mean_absolute_error(y, predictions))


def _classifier_scoring_method(classifier: Any, x: pd.DataFrame, y: pd.Series) -> str:
    probabilities = predict_positive_probability(classifier, x)
    if y.nunique() == 2:
        try:
            average_precision_score(y, probabilities)
            return "average_precision"
        except ValueError:
            pass
        try:
            roc_auc_score(y, probabilities)
            return "roc_auc"
        except ValueError:
            pass
    try:
        log_loss(y, probabilities, labels=[0, 1])
        return "neg_log_loss"
    except ValueError as exc:
        raise ValueError("No supported classifier permutation scoring method is available.") from exc


def _classifier_scorer(scoring_method: str) -> Callable[[Any, pd.DataFrame, pd.Series], float]:
    def score(estimator: Any, x: pd.DataFrame, y: pd.Series) -> float:
        probabilities = predict_positive_probability(estimator, x)
        if scoring_method == "average_precision":
            return float(average_precision_score(y, probabilities))
        if scoring_method == "roc_auc":
            return float(roc_auc_score(y, probabilities))
        if scoring_method == "neg_log_loss":
            return -float(log_loss(y, probabilities, labels=[0, 1]))
        raise ValueError(f"Unsupported scoring method: {scoring_method}")

    return score


def permutation_importance_frame(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    model_output: str,
    scoring_method: str,
    split: str,
    random_state: int,
    scoring: Callable[[Any, pd.DataFrame, pd.Series], float] | None = None,
) -> pd.DataFrame:
    """Compute model-agnostic permutation importance with a stable output schema."""
    scorer = scoring or _regression_scorer
    result = permutation_importance(
        model,
        x,
        y,
        scoring=scorer,
        n_repeats=PERMUTATION_REPEATS,
        random_state=random_state,
        n_jobs=1,
    )
    frame = pd.DataFrame(
        {
            "model_output": model_output,
            "feature": list(x.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "scoring_method": scoring_method,
            "split": split,
            "rows_used": int(len(x)),
        }
    )
    frame = frame.sort_values(
        ["importance_mean", "feature"],
        ascending=[False, True],
        ignore_index=True,
    )
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame.loc[:, PERMUTATION_COLUMNS]


def _artifact_classifier(artifact: dict[str, Any], threshold: int) -> Any:
    classifiers = artifact.get("calibrated_risk_classifiers") or {}
    return (
        classifiers.get(threshold)
        or classifiers.get(str(threshold))
        or artifact.get(f"risk_classifier_{threshold}")
    )


def prediction_context_frame(
    df: pd.DataFrame,
    split: str,
    feature_columns: list[str],
    target_column: str,
    artifact: dict[str, Any],
) -> pd.DataFrame:
    """Create row-level predictions used for representative examples."""
    x, y_delay = split_xy(df, feature_columns, target_column)
    regressor = artifact["expected_delay_regressor"]
    frame = pd.DataFrame(index=y_delay.index)
    frame["split"] = split
    frame["row_index"] = y_delay.index
    frame["actual_delay"] = pd.to_numeric(y_delay, errors="coerce")
    frame["predicted_delay_minutes"] = _predict_delay(regressor, x)
    frame["absolute_error"] = (frame["predicted_delay_minutes"] - frame["actual_delay"]).abs()

    passthrough = {
        "mode": ["mode"],
        "route": ["route", "Route"],
        "incident": ["incident", "Incident"],
        "location": ["location", "Location"],
        "timestamp": ["timestamp", "ts", "Date"],
    }
    for output_column, candidates in passthrough.items():
        source = next((column for column in candidates if column in df.columns), None)
        frame[output_column] = df.loc[y_delay.index, source] if source else pd.NA

    for threshold in RISK_THRESHOLDS:
        classifier = _artifact_classifier(artifact, threshold)
        probability_column = f"calibrated_severe_delay_probability_{threshold}"
        band_column = f"risk_band_{threshold}"
        if classifier is None:
            frame[probability_column] = np.nan
            frame[band_column] = pd.NA
            continue
        probabilities = predict_positive_probability(classifier, x)
        frame[probability_column] = probabilities
        frame[band_column] = calibrated_risk_bands(probabilities).to_numpy()
    return frame


def _important_feature_values(
    source_row: pd.Series,
    important_features: list[str],
    max_features: int = 10,
) -> str:
    values = {}
    for feature in important_features[:max_features]:
        if feature in source_row.index:
            values[feature] = _json_safe(source_row[feature])
    return json.dumps(values, sort_keys=True)


def representative_prediction_examples(
    df: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    important_features: list[str],
) -> pd.DataFrame:
    """Select representative local prediction examples without implying causality."""
    selections: list[pd.DataFrame] = []
    selection_specs = [
        ("low_predicted_delay", "predicted_delay_minutes", True),
        ("high_predicted_delay", "predicted_delay_minutes", False),
        ("high_risk_threshold_30", "calibrated_severe_delay_probability_30", False),
        ("high_risk_threshold_60", "calibrated_severe_delay_probability_60", False),
        ("large_regression_error", "absolute_error", False),
    ]
    for example_type, sort_column, ascending in selection_specs:
        if sort_column not in prediction_frame.columns:
            continue
        ranked = (
            prediction_frame[prediction_frame[sort_column].notna()]
            .sort_values(sort_column, ascending=ascending)
            .head(LOCAL_EXAMPLE_COUNT)
            .copy()
        )
        if ranked.empty:
            continue
        ranked["example_type"] = example_type
        selections.append(ranked)

    if not selections:
        return pd.DataFrame(columns=LOCAL_EXAMPLE_COLUMNS)

    examples = pd.concat(selections, ignore_index=True)
    examples["important_feature_values"] = [
        _important_feature_values(df.loc[row_index], important_features)
        for row_index in examples["row_index"]
    ]
    for column in LOCAL_EXAMPLE_COLUMNS:
        if column not in examples.columns:
            examples[column] = pd.NA
    return examples.loc[:, LOCAL_EXAMPLE_COLUMNS]


def write_top_feature_plot(frame: pd.DataFrame, output_path: Path, title: str, top_n: int) -> None:
    """Write a horizontal bar chart for the top permutation-importance features."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if frame.empty:
        return

    plot_frame = frame.head(top_n).sort_values("importance_mean", ascending=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = max(4.0, 0.25 * len(plot_frame) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(plot_frame["feature"], plot_frame["importance_mean"], xerr=plot_frame["importance_std"])
    ax.set_xlabel("Permutation importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def optional_shap_status(
    include_shap: bool,
    regressor: Any,
    x: pd.DataFrame,
    random_state: int,
) -> dict[str, Any]:
    """Attempt a small optional SHAP run where the installed environment supports it."""
    if not include_shap:
        return {"requested": False, "status": "not_requested", "details": "SHAP was not requested."}
    try:
        import shap  # type: ignore
    except ImportError:
        return {
            "requested": True,
            "status": "unavailable",
            "details": "SHAP is not installed; permutation importance reports were still generated.",
        }

    try:
        background = deterministic_sample(x, min(100, len(x)), random_state)
        explain_rows = deterministic_sample(x, min(25, len(x)), random_state + 1)
        explainer = shap.Explainer(regressor.predict, background)
        _ = explainer(explain_rows)
        return {
            "requested": True,
            "status": "available_regression_smoke_test",
            "details": (
                "SHAP successfully ran for a small expected-delay regression sample. "
                "No SHAP report files are emitted in this phase; permutation importance "
                "remains the documented output."
            ),
        }
    except Exception as exc:  # pragma: no cover - depends on optional third-party behavior
        return {
            "requested": True,
            "status": "skipped_after_error",
            "details": f"SHAP could not explain the wrapped model in this environment: {exc}",
        }


def _top_features(frame: pd.DataFrame, n: int = 10) -> list[dict[str, Any]]:
    columns = ["feature", "importance_mean", "importance_std", "rank", "scoring_method"]
    return frame.head(n).loc[:, columns].to_dict(orient="records")


def run_explainability(
    modeling_dir: Path = DEFAULT_MODELING_DIR,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    split: str = DEFAULT_SPLIT,
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = DEFAULT_RANDOM_STATE,
    top_n_features: int = DEFAULT_TOP_N_FEATURES,
    include_shap: bool = False,
) -> dict[str, Any]:
    """Generate global and representative local explainability reports."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be either 'validation' or 'test'.")

    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("Expected the calibrated artifact to be a dictionary.")
    if artifact.get("expected_delay_regressor") is None:
        raise ValueError("Artifact is missing `expected_delay_regressor`.")

    metadata = load_feature_metadata(modeling_dir)
    feature_columns = approved_feature_columns(metadata, artifact)
    if not feature_columns:
        raise ValueError("No approved feature columns are available for explanation.")

    _, categorical_columns, _ = feature_groups_from_metadata(metadata)
    splits = load_modeling_splits(modeling_dir, categorical_columns)
    sampled_df = deterministic_sample(splits[split], max_rows=max_rows, random_state=random_state)
    target_column = artifact.get("target_column") or metadata.get("target_column", TARGET_COLUMN_FALLBACK)
    x_sample, y_delay = split_xy(sampled_df, feature_columns, target_column)
    sampled_df = sampled_df.loc[y_delay.index].copy()

    regressor = artifact["expected_delay_regressor"]
    regression_importance = permutation_importance_frame(
        model=regressor,
        x=x_sample,
        y=y_delay,
        model_output="expected_delay_regression",
        scoring_method="negative_mean_absolute_error",
        split=split,
        random_state=random_state,
    )

    risk_importance: dict[int, pd.DataFrame] = {}
    for threshold in RISK_THRESHOLDS:
        classifier = _artifact_classifier(artifact, threshold)
        if classifier is None:
            raise ValueError(f"Artifact is missing calibrated risk classifier for threshold {threshold}.")
        y_binary = make_binary_target(y_delay, threshold)
        scoring_method = _classifier_scoring_method(classifier, x_sample, y_binary)
        risk_importance[threshold] = permutation_importance_frame(
            model=classifier,
            x=x_sample,
            y=y_binary,
            model_output=f"calibrated_severe_delay_risk_{threshold}",
            scoring_method=scoring_method,
            split=split,
            random_state=random_state,
            scoring=_classifier_scorer(scoring_method),
        )

    global_importance = pd.concat(
        [regression_importance, risk_importance[30], risk_importance[60]],
        ignore_index=True,
    )
    important_features = (
        global_importance.sort_values(["importance_mean", "feature"], ascending=[False, True])[
            "feature"
        ]
        .drop_duplicates()
        .head(top_n_features)
        .tolist()
    )
    prediction_frame = prediction_context_frame(
        df=sampled_df,
        split=split,
        feature_columns=feature_columns,
        target_column=target_column,
        artifact=artifact,
    )
    examples = representative_prediction_examples(sampled_df, prediction_frame, important_features)

    shap_status = optional_shap_status(include_shap, regressor, x_sample, random_state)
    excluded_columns = sorted(
        set(metadata.get("excluded_columns", []))
        | set(metadata.get("leakage_sensitive_columns", []))
        | LEAKAGE_AND_NON_FEATURE_COLUMNS
        | LEAKAGE_SENSITIVE_COLUMNS
    )
    summary = {
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(artifact_path),
        "split_used": split,
        "rows_sampled": int(len(x_sample)),
        "feature_columns": feature_columns,
        "excluded_leakage_sensitive_columns": excluded_columns,
        "model_outputs_explained": [
            "expected_delay_regression",
            "calibrated_severe_delay_risk_30",
            "calibrated_severe_delay_risk_60",
        ],
        "explanation_methods_used": ["permutation_importance"]
        + (["optional_shap_smoke_test"] if include_shap else []),
        "shap_availability_status": shap_status,
        "top_10_features_regression": _top_features(regression_importance),
        "top_10_features_risk_30": _top_features(risk_importance[30]),
        "top_10_features_risk_60": _top_features(risk_importance[60]),
        "limitations": [
            "Permutation importance is associative and should not be interpreted as causal.",
            "Importance values can vary by split, sample, random seed, and correlated features.",
            "Categorical variables are permuted at the original feature-column level before preprocessing.",
            "Calibrated risk classifier importances explain probability/ranking outputs, not causal mechanisms.",
            "Representative prediction examples provide local feature context only unless SHAP is available.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    regression_importance.to_csv(output_dir / "permutation_importance_regression.csv", index=False)
    risk_importance[30].to_csv(output_dir / "permutation_importance_risk_30.csv", index=False)
    risk_importance[60].to_csv(output_dir / "permutation_importance_risk_60.csv", index=False)
    global_importance.to_csv(output_dir / "global_feature_importance.csv", index=False)
    examples.to_csv(output_dir / "representative_prediction_examples.csv", index=False)
    (output_dir / "explainability_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_top_feature_plot(
        regression_importance,
        output_dir / "figures" / "top_features_regression.png",
        "Top expected-delay features",
        top_n_features,
    )
    write_top_feature_plot(
        risk_importance[30],
        output_dir / "figures" / "top_features_risk_30.png",
        "Top severe-delay risk features: 30 minutes",
        top_n_features,
    )
    write_top_feature_plot(
        risk_importance[60],
        output_dir / "figures" / "top_features_risk_60.png",
        "Top severe-delay risk features: 60 minutes",
        top_n_features,
    )
    return {
        "regression_importance": regression_importance,
        "risk_importance_30": risk_importance[30],
        "risk_importance_60": risk_importance[60],
        "global_importance": global_importance,
        "representative_examples": examples,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate explainability reports for the calibrated two-output model."
    )
    parser.add_argument("--modeling-dir", type=Path, default=DEFAULT_MODELING_DIR)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=["validation", "test"], default=DEFAULT_SPLIT)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--top-n-features", type=int, default=DEFAULT_TOP_N_FEATURES)
    parser.add_argument("--include-shap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_explainability(
        modeling_dir=args.modeling_dir,
        artifact_path=args.artifact_path,
        output_dir=args.output_dir,
        split=args.split,
        max_rows=args.max_rows,
        random_state=args.random_state,
        top_n_features=args.top_n_features,
        include_shap=args.include_shap,
    )


if __name__ == "__main__":
    main()
