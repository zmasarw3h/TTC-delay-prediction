import importlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.models import train_risk_models
from src.models.train_model import feature_columns_from_metadata
from src.models.train_risk_models import (
    assign_risk_band,
    build_two_output_artifact,
    calculate_scale_pos_weight,
    make_binary_target,
    risk_bands,
    run_risk_model_training,
    select_operating_threshold,
    threshold_table,
)


class ConstantRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value, dtype="float64")


def _metadata():
    return {
        "target_column": "Min Delay",
        "feature_columns": [
            "mode",
            "Route",
            "Incident",
            "Location",
            "hour",
            "prior_route_mean_delay",
            "Min Gap",
            "Min Delay",
            "ts",
            "severe_delay_15",
        ],
        "categorical_columns": ["mode", "Route", "Incident", "Location"],
        "numeric_columns": [
            "hour",
            "prior_route_mean_delay",
            "Min Gap",
            "Min Delay",
            "severe_delay_15",
        ],
        "excluded_columns": ["Min Gap", "Min Delay", "ts", "severe_delay_15"],
    }


def _frame(delays):
    rows = []
    for index, delay in enumerate(delays):
        rows.append(
            {
                "mode": "bus" if index % 2 == 0 else "streetcar",
                "Route": "1" if index % 2 == 0 else "501",
                "Incident": "Mechanical" if index % 3 else "Collision",
                "Location": "Station" if index % 2 == 0 else "Street",
                "hour": 7 + index,
                "prior_route_mean_delay": 8 + index,
                "Min Gap": 999,
                "Min Delay": delay,
                "ts": f"2023-01-{index + 1:02d} 08:00:00",
                "severe_delay_15": int(delay >= 15),
            }
        )
    return pd.DataFrame(rows)


def test_binary_targets_for_30_and_60():
    delays = pd.Series([0, 29, 30, 59, 60])

    assert make_binary_target(delays, 30).tolist() == [0, 0, 1, 1, 1]
    assert make_binary_target(delays, 60).tolist() == [0, 0, 0, 0, 1]


def test_scale_pos_weight_calculation():
    y = pd.Series([0, 0, 0, 1])

    assert calculate_scale_pos_weight(y) == 3.0


def test_threshold_table_and_validation_only_selection():
    y = pd.Series([1, 1, 1, 0, 0])
    probabilities = pd.Series([0.95, 0.60, 0.25, 0.40, 0.10])
    table = threshold_table(y, probabilities, cutoffs=[0.2, 0.5, 0.8])

    selected = select_operating_threshold(table)

    assert table["probability_cutoff"].tolist() == [0.2, 0.5, 0.8]
    assert selected["probability_cutoff"] == 0.2
    assert selected["met_min_recall"] is True


def test_risk_band_assignment():
    assert assign_risk_band(0.19) == "low"
    assert assign_risk_band(0.20) == "medium"
    assert assign_risk_band(0.49) == "medium"
    assert assign_risk_band(0.50) == "high"
    assert risk_bands(pd.Series([0.1, 0.3, 0.8])).tolist() == ["low", "medium", "high"]


def test_two_output_artifact_structure():
    artifact = build_two_output_artifact(
        expected_delay_regressor=ConstantRegressor(10),
        risk_classifiers={30: "classifier30", 60: "classifier60"},
        feature_columns=["mode", "Route", "hour"],
        target_column="Min Delay",
        selected_thresholds={"30": {"probability_cutoff": 0.3}},
        metadata=_metadata(),
        regressor_source="test",
    )

    assert artifact["expected_delay_regressor"].value == 10
    assert artifact["risk_classifier_30"] == "classifier30"
    assert artifact["risk_classifier_60"] == "classifier60"
    assert artifact["selected_probability_thresholds"]["30"]["probability_cutoff"] == 0.3
    assert artifact["risk_band_definitions"]["medium"]["min_probability"] == 0.20


def test_min_gap_is_not_used_as_feature():
    features = feature_columns_from_metadata(_metadata())

    assert "Min Gap" not in features
    assert "Min Delay" not in features


def test_run_risk_model_training_writes_outputs_with_fake_models(tmp_path, monkeypatch):
    modeling_dir = tmp_path / "modeling"
    reports_dir = tmp_path / "reports"
    artifacts_dir = tmp_path / "artifacts"
    selected_regressor_path = tmp_path / "selected_experiment.joblib"
    modeling_dir.mkdir()
    (modeling_dir / "feature_metadata.json").write_text(json.dumps(_metadata()), encoding="utf-8")
    _frame([5, 12, 30, 45, 60, 90, 8, 75]).to_csv(modeling_dir / "train.csv", index=False)
    _frame([10, 35, 65, 80]).to_csv(modeling_dir / "validation.csv", index=False)
    _frame([15, 40, 70, 95]).to_csv(modeling_dir / "test.csv", index=False)
    joblib.dump(ConstantRegressor(7.9), selected_regressor_path)

    monkeypatch.setattr(
        train_risk_models,
        "build_xgb_classifier",
        lambda scale_pos_weight, config=None: DecisionTreeClassifier(max_depth=2, random_state=42),
    )

    result = run_risk_model_training(
        modeling_dir=modeling_dir,
        selected_regressor_path=selected_regressor_path,
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
        thresholds=[30, 60],
    )

    assert set(result["classification_metrics"]["threshold_minutes"]) == {30, 60}
    assert set(result["classification_metrics"]["split"]) == {"validation", "test"}
    assert (reports_dir / "regression_metrics.csv").exists()
    assert (reports_dir / "classification_metrics.csv").exists()
    assert (reports_dir / "classification_threshold_table.csv").exists()
    assert (reports_dir / "selected_classification_thresholds.json").exists()
    assert (reports_dir / "risk_band_summary.csv").exists()
    assert (reports_dir / "two_output_predictions_validation.csv").exists()
    assert (reports_dir / "two_output_predictions_test.csv").exists()
    assert (reports_dir / "two_output_summary.json").exists()
    assert (artifacts_dir / "two_output_model.joblib").exists()
    assert "Min Gap" not in result["artifact"]["feature_columns"]


def test_module_import_is_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    module = importlib.reload(train_risk_models)

    assert hasattr(module, "run_risk_model_training")
    assert hasattr(module, "LogTargetRegressor")
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "artifacts").exists()
