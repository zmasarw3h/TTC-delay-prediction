import importlib
import json

import joblib
import numpy as np
import pandas as pd

from src.models import explain_models
from src.models.explain_models import (
    PERMUTATION_COLUMNS,
    approved_feature_columns,
    deterministic_sample,
    permutation_importance_frame,
    representative_prediction_examples,
    run_explainability,
)


class FakeRegressor:
    def fit(self, x, y):
        return self

    def predict(self, x):
        return pd.to_numeric(x["feature"], errors="coerce").to_numpy(dtype="float64") * 10.0


class FakeClassifier:
    def __init__(self, feature_scale=1.0):
        self.feature_scale = feature_scale

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        feature = pd.to_numeric(x["feature"], errors="coerce").to_numpy(dtype="float64")
        probabilities = np.clip((feature * self.feature_scale) / 10.0, 0.01, 0.99)
        return np.column_stack([1.0 - probabilities, probabilities])


def _metadata():
    return {
        "target_column": "Min Delay",
        "feature_columns": [
            "feature",
            "mode",
            "Route",
            "Incident",
            "Location",
            "Min Delay",
            "Min Gap",
            "severe_delay_15",
            "ts",
            "source_file",
        ],
        "categorical_columns": ["mode", "Route", "Incident", "Location"],
        "numeric_columns": ["feature", "Min Delay", "Min Gap"],
        "excluded_columns": ["Min Delay", "Min Gap", "severe_delay_15", "ts", "source_file"],
        "leakage_sensitive_columns": ["Min Delay", "Min Gap"],
    }


def _frame():
    return pd.DataFrame(
        {
            "feature": [0, 1, 2, 3, 4, 5, 6, 7],
            "mode": ["bus", "bus", "bus", "streetcar", "streetcar", "bus", "streetcar", "bus"],
            "Route": ["1", "1", "2", "501", "501", "3", "504", "4"],
            "Incident": ["Delay", "Mechanical", "Delay", "Collision"] * 2,
            "Location": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "Min Delay": [2, 8, 15, 32, 45, 61, 75, 90],
            "Min Gap": [999] * 8,
            "severe_delay_15": [0, 0, 1, 1, 1, 1, 1, 1],
            "ts": [f"2023-01-0{i + 1} 08:00:00" for i in range(8)],
            "source_file": ["raw.xlsx"] * 8,
        }
    )


def _artifact():
    return {
        "expected_delay_regressor": FakeRegressor(),
        "calibrated_risk_classifiers": {
            30: FakeClassifier(feature_scale=1.3),
            60: FakeClassifier(feature_scale=1.0),
        },
        "risk_classifier_30": FakeClassifier(feature_scale=1.3),
        "risk_classifier_60": FakeClassifier(feature_scale=1.0),
        "feature_columns": [
            "feature",
            "mode",
            "Route",
            "Incident",
            "Location",
            "Min Delay",
            "Min Gap",
            "ts",
        ],
        "target_column": "Min Delay",
    }


def test_approved_feature_selection_excludes_leakage_sensitive_columns():
    features = approved_feature_columns(_metadata(), _artifact())

    assert features == ["feature", "mode", "Route", "Incident", "Location"]
    assert "Min Delay" not in features
    assert "Min Gap" not in features
    assert "ts" not in features


def test_sampling_is_deterministic_with_random_state():
    frame = _frame()

    first = deterministic_sample(frame, max_rows=4, random_state=42)
    second = deterministic_sample(frame, max_rows=4, random_state=42)
    different = deterministic_sample(frame, max_rows=4, random_state=7)

    assert first.index.tolist() == second.index.tolist()
    assert first.index.tolist() != different.index.tolist()


def test_permutation_importance_output_schema_is_correct():
    frame = _frame()
    x = frame[["feature", "mode"]]
    y = frame["Min Delay"]

    importance = permutation_importance_frame(
        model=FakeRegressor(),
        x=x,
        y=y,
        model_output="expected_delay_regression",
        scoring_method="negative_mean_absolute_error",
        split="validation",
        random_state=42,
    )

    assert list(importance.columns) == PERMUTATION_COLUMNS
    assert set(importance["feature"]) == {"feature", "mode"}
    assert importance["rows_used"].tolist() == [8, 8]


def test_representative_example_selection_includes_probability_and_risk_fields():
    source = _frame()
    predictions = pd.DataFrame(
        {
            "split": ["validation"] * len(source),
            "row_index": source.index,
            "mode": source["mode"],
            "route": source["Route"],
            "incident": source["Incident"],
            "location": source["Location"],
            "timestamp": source["ts"],
            "actual_delay": source["Min Delay"],
            "predicted_delay_minutes": [1, 5, 10, 20, 40, 60, 80, 100],
            "calibrated_severe_delay_probability_30": [0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 0.8, 0.9],
            "risk_band_30": ["low", "low", "medium", "medium", "high", "high", "high", "high"],
            "calibrated_severe_delay_probability_60": [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9],
            "risk_band_60": ["low", "low", "low", "medium", "medium", "high", "high", "high"],
            "absolute_error": [1, 3, 5, 12, 5, 1, 5, 10],
        }
    )

    examples = representative_prediction_examples(source, predictions, ["feature", "mode"])

    assert "calibrated_severe_delay_probability_30" in examples.columns
    assert "risk_band_60" in examples.columns
    assert "important_feature_values" in examples.columns
    assert {"high_risk_threshold_30", "large_regression_error"}.issubset(
        set(examples["example_type"])
    )


def test_run_explainability_writes_expected_reports_with_fake_models(tmp_path):
    modeling_dir = tmp_path / "modeling"
    output_dir = tmp_path / "reports" / "explainability"
    artifact_path = tmp_path / "artifacts" / "calibrated_two_output_model.joblib"
    modeling_dir.mkdir(parents=True)
    artifact_path.parent.mkdir(parents=True)

    (modeling_dir / "feature_metadata.json").write_text(json.dumps(_metadata()), encoding="utf-8")
    frame = _frame()
    frame.to_csv(modeling_dir / "train.csv", index=False)
    frame.to_csv(modeling_dir / "validation.csv", index=False)
    frame.to_csv(modeling_dir / "test.csv", index=False)
    joblib.dump(_artifact(), artifact_path)

    result = run_explainability(
        modeling_dir=modeling_dir,
        artifact_path=artifact_path,
        output_dir=output_dir,
        split="validation",
        max_rows=6,
        random_state=42,
        top_n_features=3,
    )

    expected_files = [
        "permutation_importance_regression.csv",
        "permutation_importance_risk_30.csv",
        "permutation_importance_risk_60.csv",
        "global_feature_importance.csv",
        "representative_prediction_examples.csv",
        "explainability_summary.json",
    ]
    for file_name in expected_files:
        assert (output_dir / file_name).exists()

    summary = json.loads((output_dir / "explainability_summary.json").read_text())
    assert summary["split_used"] == "validation"
    assert summary["rows_sampled"] == 6
    assert "Min Delay" not in summary["feature_columns"]
    assert result["representative_examples"]["risk_band_30"].notna().any()


def test_module_import_is_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    module = importlib.reload(explain_models)

    assert hasattr(module, "run_explainability")
    assert not (tmp_path / "reports").exists()
