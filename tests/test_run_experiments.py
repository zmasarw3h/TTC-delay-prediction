import importlib
import json

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src.models import run_experiments
from src.models.run_experiments import (
    LogTargetRegressor,
    ModeSpecificRegressor,
    delay_sample_weights,
    evaluate_frames,
    evaluate_high_delay,
    feature_columns_from_metadata,
    prediction_frame,
    run_experiments as run_phase7a_experiments,
    select_best_experiment,
)
from src.models.train_model import build_model_pipeline


class ConstantModel:
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
            "hour",
            "prior_route_mean_delay",
            "Min Gap",
            "Min Delay",
            "ts",
            "severe_delay_15",
        ],
        "categorical_columns": ["mode", "Route"],
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
                "hour": 8 + index,
                "prior_route_mean_delay": 10 + index,
                "Min Gap": 999,
                "Min Delay": delay,
                "ts": f"2023-01-{index + 1:02d} 08:00:00",
                "severe_delay_15": int(delay >= 15),
            }
        )
    return pd.DataFrame(rows)


def test_sample_weights_are_assigned_correctly():
    weights = delay_sample_weights(pd.Series([0, 15, 16, 30, 31, 60, 61, 120, 121, 240]))

    assert weights.tolist() == [1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]


def test_log_target_model_predicts_on_original_delay_scale():
    pipeline = build_model_pipeline(
        categorical_columns=[],
        numeric_columns=["x"],
        model=DecisionTreeRegressor(random_state=42),
    )
    model = LogTargetRegressor(pipeline)
    x_train = pd.DataFrame({"x": [1, 2, 3, 4]})
    y_train = pd.Series([5, 10, 20, 40])

    model.fit(x_train, y_train)
    predictions = model.predict(pd.DataFrame({"x": [1, 4]}))

    assert predictions[0] > 4
    assert predictions[1] > 30
    assert predictions.max() <= 240


def test_mode_specific_prediction_routes_rows_to_matching_model():
    model = ModeSpecificRegressor(
        models={"bus": ConstantModel(11), "streetcar": ConstantModel(22)}
    )
    predictions = model.predict(
        pd.DataFrame({"mode": ["streetcar", "bus", "streetcar"], "x": [1, 2, 3]})
    )

    assert predictions.tolist() == [22.0, 11.0, 22.0]


def test_experiment_metrics_are_computed():
    frame = prediction_frame(
        model=ConstantModel(10),
        df=_frame([8, 12, 40]),
        split_name="validation",
        feature_columns=["mode", "Route", "hour", "prior_route_mean_delay"],
        target_column="Min Delay",
    )
    metrics = evaluate_frames("constant", {"validation": frame, "test": frame})
    high_delay = evaluate_high_delay("constant", {"validation": frame, "test": frame})

    assert set(metrics["split"]) == {"validation", "test"}
    assert metrics.loc[metrics["split"] == "validation", "mae"].iloc[0] > 0
    assert set(high_delay["threshold_minutes"]) == {15, 30, 60}
    assert "underprediction_percent" in high_delay.columns


def test_selection_rule_uses_validation_only():
    metrics = pd.DataFrame(
        [
            {"experiment": "combined_xgb_fixed", "split": "validation", "mae": 10.0},
            {"experiment": "combined_xgb_weighted", "split": "validation", "mae": 10.05},
            {"experiment": "mode_specific_xgb", "split": "validation", "mae": 50.0},
            {"experiment": "combined_xgb_fixed", "split": "test", "mae": 10.0},
            {"experiment": "combined_xgb_weighted", "split": "test", "mae": 20.0},
            {"experiment": "mode_specific_xgb", "split": "test", "mae": 1.0},
        ]
    )
    high_delay = pd.DataFrame(
        [
            {
                "experiment": "combined_xgb_fixed",
                "split": "validation",
                "threshold_minutes": 30,
                "mae_high_delay": 8.0,
            },
            {
                "experiment": "combined_xgb_weighted",
                "split": "validation",
                "threshold_minutes": 30,
                "mae_high_delay": 4.0,
            },
            {
                "experiment": "mode_specific_xgb",
                "split": "validation",
                "threshold_minutes": 30,
                "mae_high_delay": 1.0,
            },
        ]
    )
    by_mode = pd.DataFrame(
        [
            {
                "experiment": name,
                "split": "validation",
                "mode": "streetcar",
                "mae": value,
            }
            for name, value in [
                ("combined_xgb_fixed", 5.0),
                ("combined_xgb_weighted", 6.0),
                ("mode_specific_xgb", 1.0),
            ]
        ]
    )

    selection = select_best_experiment(metrics, high_delay, by_mode)

    assert selection["selected_experiment"] == "combined_xgb_weighted"
    assert "mode_specific_xgb" not in selection["candidate_experiments_within_1_percent"]


def test_min_gap_is_not_used():
    features = feature_columns_from_metadata(_metadata())

    assert "Min Gap" not in features
    assert "Min Delay" not in features


def test_run_experiments_writes_reports_and_selected_artifact_only(tmp_path, monkeypatch):
    modeling_dir = tmp_path / "modeling"
    reports_dir = tmp_path / "reports"
    artifacts_dir = tmp_path / "artifacts"
    modeling_dir.mkdir()
    (modeling_dir / "feature_metadata.json").write_text(json.dumps(_metadata()), encoding="utf-8")
    _frame([8, 12, 20, 30, 45, 70]).to_csv(modeling_dir / "train.csv", index=False)
    _frame([9, 18, 35, 65]).to_csv(modeling_dir / "validation.csv", index=False)
    _frame([10, 22, 40, 80]).to_csv(modeling_dir / "test.csv", index=False)

    monkeypatch.setattr(
        run_experiments,
        "build_xgb_regressor",
        lambda config=None: DecisionTreeRegressor(max_depth=2, random_state=42),
    )
    result = run_phase7a_experiments(
        modeling_dir=modeling_dir,
        baseline_report=tmp_path / "missing_baseline.json",
        fixed_model_report=tmp_path / "missing_fixed.json",
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
    )

    assert set(result["metrics"]["experiment"]) == {
        "combined_xgb_fixed",
        "combined_xgb_weighted",
        "combined_xgb_log_target",
        "mode_specific_xgb",
    }
    assert (reports_dir / "experiment_metrics.csv").exists()
    assert (reports_dir / "experiment_metrics_by_mode.csv").exists()
    assert (reports_dir / "experiment_high_delay_metrics.csv").exists()
    assert (reports_dir / "experiment_selection.json").exists()
    assert (reports_dir / "experiment_summary.json").exists()
    assert (artifacts_dir / "selected_experiment.joblib").exists()
    assert len(list(artifacts_dir.glob("*.joblib"))) == 1


def test_module_import_is_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    module = importlib.reload(run_experiments)

    assert hasattr(module, "run_experiments")
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "artifacts").exists()
