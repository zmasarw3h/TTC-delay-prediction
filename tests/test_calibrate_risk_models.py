import importlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.models import calibrate_risk_models
from src.models.calibrate_risk_models import (
    assign_calibrated_risk_band,
    build_calibrated_artifact,
    calibrated_risk_bands,
    expected_calibration_error,
    probability_bin_table,
    run_calibration,
    select_calibration_method,
    split_base_and_calibration_training,
)
from src.models.train_risk_models import select_operating_threshold, threshold_table


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
        ],
        "categorical_columns": ["mode", "Route", "Incident", "Location"],
        "numeric_columns": ["hour", "prior_route_mean_delay", "Min Gap", "Min Delay"],
        "excluded_columns": ["Min Gap", "Min Delay", "ts"],
    }


def _row(index, timestamp, delay):
    return {
        "mode": "bus" if index % 2 == 0 else "streetcar",
        "Route": "1" if index % 2 == 0 else "501",
        "Incident": "Mechanical" if index % 3 else "Collision",
        "Location": "Station" if index % 2 == 0 else "Street",
        "hour": 6 + (index % 12),
        "prior_route_mean_delay": 5 + index,
        "Min Gap": 999,
        "Min Delay": delay,
        "ts": timestamp,
    }


def _frame(year, delays):
    return pd.DataFrame(
        [
            _row(index, f"{year}-{(index % 12) + 1:02d}-01 08:00:00", delay)
            for index, delay in enumerate(delays)
        ]
    )


def test_expected_calibration_error_uses_weighted_bin_errors():
    y = pd.Series([0, 1, 1, 0])
    probabilities = pd.Series([0.05, 0.15, 0.85, 0.95])

    ece = expected_calibration_error(y, probabilities, n_bins=2)

    assert round(ece, 6) == 0.4


def test_probability_bin_table_creation():
    y = pd.Series([0, 1, 1])
    probabilities = pd.Series([0.05, 0.25, 0.95])

    table = probability_bin_table(y, probabilities, n_bins=2)

    assert table["bin_lower"].tolist() == [0.0, 0.5]
    assert table["bin_upper"].tolist() == [0.5, 1.0]
    assert table["row_count"].tolist() == [2, 1]
    assert table.loc[0, "actual_severe_delay_rate"] == 0.5


def test_operating_threshold_selection_prefers_recall_then_f1():
    y = pd.Series([1, 1, 1, 0, 0])
    probabilities = pd.Series([0.95, 0.60, 0.25, 0.40, 0.10])
    table = threshold_table(y, probabilities, cutoffs=[0.2, 0.5, 0.8])

    selected = select_operating_threshold(table)

    assert selected["probability_cutoff"] == 0.2
    assert selected["met_min_recall"] is True


def test_calibration_method_selection_uses_validation_rule():
    validation_metrics = pd.DataFrame(
        [
            {
                "calibration_method": "uncalibrated",
                "brier_score": 0.204,
                "expected_calibration_error": 0.040,
                "pr_auc": 0.55,
            },
            {
                "calibration_method": "sigmoid",
                "brier_score": 0.200,
                "expected_calibration_error": 0.050,
                "pr_auc": 0.54,
            },
            {
                "calibration_method": "isotonic",
                "brier_score": 0.201,
                "expected_calibration_error": 0.030,
                "pr_auc": 0.53,
            },
        ]
    )

    selected = select_calibration_method(validation_metrics)

    assert selected["calibration_method"] == "isotonic"


def test_calibrated_risk_band_assignment():
    assert assign_calibrated_risk_band(0.09) == "low"
    assert assign_calibrated_risk_band(0.10) == "medium"
    assert assign_calibrated_risk_band(0.29) == "medium"
    assert assign_calibrated_risk_band(0.30) == "high"
    assert calibrated_risk_bands(pd.Series([0.05, 0.20, 0.70])).tolist() == [
        "low",
        "medium",
        "high",
    ]


def test_training_split_logic_uses_pre_2022_and_calendar_2022():
    train = pd.concat(
        [
            _frame(2020, [5, 30]),
            _frame(2021, [10, 60]),
            _frame(2022, [15, 90]),
        ],
        ignore_index=True,
    )

    base, calibration = split_base_and_calibration_training(train)

    assert len(base) == 4
    assert len(calibration) == 2
    assert pd.to_datetime(base["ts"]).max() < pd.Timestamp("2022-01-01")
    assert set(pd.to_datetime(calibration["ts"]).dt.year) == {2022}


def test_calibrated_artifact_structure():
    artifact = build_calibrated_artifact(
        expected_delay_regressor=ConstantRegressor(8),
        calibrated_risk_classifiers={30: "sigmoid30", 60: "isotonic60"},
        all_risk_classifiers={30: {"sigmoid": "sigmoid30"}, 60: {"isotonic": "isotonic60"}},
        feature_columns=["mode", "Route", "hour"],
        target_column="Min Delay",
        selected_methods={"30": {"calibration_method": "sigmoid"}},
        selected_cutoffs={"30": {"probability_cutoff": 0.2}},
        metadata=_metadata(),
        regressor_source="test",
        split_summary={"calibration": {"row_count": 2}},
    )

    assert artifact["model_phase"] == "Phase 7C"
    assert artifact["expected_delay_regressor"].value == 8
    assert artifact["risk_classifier_30"] == "sigmoid30"
    assert artifact["selected_calibration_methods"]["30"]["calibration_method"] == "sigmoid"
    assert artifact["risk_band_definitions"]["medium"]["min_probability"] == 0.10


def test_run_calibration_writes_new_outputs_with_fake_models(tmp_path, monkeypatch):
    modeling_dir = tmp_path / "modeling"
    reports_dir = tmp_path / "reports" / "calibration"
    artifacts_dir = tmp_path / "artifacts" / "calibration"
    phase_7b_artifact_path = tmp_path / "artifacts" / "risk_models" / "two_output_model.joblib"
    modeling_dir.mkdir(parents=True)
    phase_7b_artifact_path.parent.mkdir(parents=True)
    (modeling_dir / "feature_metadata.json").write_text(json.dumps(_metadata()), encoding="utf-8")

    train = pd.concat(
        [
            _frame(2020, [5, 12, 30, 45, 60, 90, 8, 75]),
            _frame(2021, [4, 18, 35, 55, 65, 95, 9, 70]),
            _frame(2022, [6, 20, 40, 58, 68, 100, 7, 80]),
        ],
        ignore_index=True,
    )
    train.to_csv(modeling_dir / "train.csv", index=False)
    _frame(2023, [10, 35, 65, 80, 5, 70]).to_csv(modeling_dir / "validation.csv", index=False)
    _frame(2024, [15, 40, 70, 95, 3, 62]).to_csv(modeling_dir / "test.csv", index=False)
    joblib.dump({"expected_delay_regressor": ConstantRegressor(7.9)}, phase_7b_artifact_path)

    monkeypatch.setattr(
        calibrate_risk_models,
        "build_xgb_classifier",
        lambda scale_pos_weight, config=None: DecisionTreeClassifier(max_depth=2, random_state=42),
    )

    result = run_calibration(
        modeling_dir=modeling_dir,
        phase_7b_artifact_path=phase_7b_artifact_path,
        selected_regressor_path=tmp_path / "missing_selected.joblib",
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
        thresholds=[30, 60],
    )

    assert set(result["metrics"]["calibration_method"]) == {
        "uncalibrated",
        "sigmoid",
        "isotonic",
    }
    assert (reports_dir / "calibration_metrics.csv").exists()
    assert (reports_dir / "calibration_threshold_table.csv").exists()
    assert (reports_dir / "calibration_bin_table.csv").exists()
    assert (reports_dir / "calibrated_risk_band_summary.csv").exists()
    assert (reports_dir / "calibration_selection.json").exists()
    assert (reports_dir / "calibrated_two_output_predictions_validation.csv").exists()
    assert (reports_dir / "calibrated_two_output_predictions_test.csv").exists()
    assert (reports_dir / "calibrated_two_output_summary.json").exists()
    assert (artifacts_dir / "calibrated_two_output_model.joblib").exists()
    assert not (tmp_path / "reports" / "risk_models").exists()
    assert not (tmp_path / "artifacts" / "risk_models" / "calibrated_two_output_model.joblib").exists()


def test_module_import_is_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    module = importlib.reload(calibrate_risk_models)

    assert hasattr(module, "run_calibration")
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "artifacts").exists()
