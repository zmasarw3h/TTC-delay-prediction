import importlib
import json

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src.models import train_model
from src.models.train_model import (
    build_preprocessor,
    calculate_metrics,
    feature_columns_from_metadata,
    run_training,
    train_and_evaluate,
)


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
                "ts": f"2023-01-0{index + 1} 08:00:00",
                "severe_delay_15": int(delay >= 15),
            }
        )
    return pd.DataFrame(rows)


def test_preprocessing_pipeline_builds():
    preprocessor = build_preprocessor(
        categorical_columns=["mode", "Route"],
        numeric_columns=["hour", "prior_route_mean_delay"],
    )
    transformed = preprocessor.fit_transform(
        pd.DataFrame(
            {
                "mode": ["bus", pd.NA],
                "Route": ["1", "501"],
                "hour": [8, None],
                "prior_route_mean_delay": [10.0, 20.0],
            }
        )
    )

    assert transformed.shape[0] == 2


def test_metrics_calculation_works():
    metrics = calculate_metrics(
        y_true=pd.Series([10, 20]),
        y_pred=pd.Series([11, 18]),
    )

    assert metrics["mae"] == 1.5
    assert round(metrics["rmse"], 3) == 1.581


def test_feature_metadata_is_respected_and_min_gap_is_not_used():
    features = feature_columns_from_metadata(_metadata())

    assert features == ["mode", "Route", "hour", "prior_route_mean_delay"]
    assert "Min Gap" not in features
    assert "Min Delay" not in features
    assert "ts" not in features


def test_train_evaluate_function_runs_on_small_splits():
    splits = {
        "train": _frame([10, 12, 20, 22]),
        "validation": _frame([11, 21]),
        "test": _frame([13, 23]),
    }

    result = train_and_evaluate(
        splits=splits,
        metadata=_metadata(),
        model=DecisionTreeRegressor(max_depth=2, random_state=42),
        model_config={"model": "decision_tree_test_double"},
    )

    assert set(result["metrics"]["split"]) == {"validation", "test"}
    assert set(result["metrics_by_mode"]["split"]) == {"validation", "test"}
    assert "Min Gap" not in result["pipeline"].named_steps["preprocessor"].feature_names_in_


def test_run_training_writes_expected_files(tmp_path):
    modeling_dir = tmp_path / "modeling"
    reports_dir = tmp_path / "reports"
    artifacts_dir = tmp_path / "artifacts"
    baseline_report = tmp_path / "baseline_metrics.json"
    modeling_dir.mkdir()

    metadata = _metadata()
    (modeling_dir / "feature_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    _frame([10, 12, 20, 22]).to_csv(modeling_dir / "train.csv", index=False)
    _frame([11, 21]).to_csv(modeling_dir / "validation.csv", index=False)
    _frame([13, 23]).to_csv(modeling_dir / "test.csv", index=False)
    baseline_report.write_text(
        json.dumps(
            {
                "best_baseline": {"baseline": "prior_route_mean_delay", "filled": True},
                "metrics": [
                    {
                        "baseline": "prior_route_mean_delay",
                        "evaluation_name": "prior_route_mean_delay_filled",
                        "filled": True,
                        "split": "validation",
                        "mae": 8.0,
                    },
                    {
                        "baseline": "prior_route_mean_delay",
                        "evaluation_name": "prior_route_mean_delay_filled",
                        "filled": True,
                        "split": "test",
                        "mae": 9.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    original_build_xgb_regressor = train_model.build_xgb_regressor
    train_model.build_xgb_regressor = lambda config=None: DecisionTreeRegressor(
        max_depth=2,
        random_state=42,
    )
    try:
        run_training(
            modeling_dir=modeling_dir,
            reports_dir=reports_dir,
            artifacts_dir=artifacts_dir,
            baseline_report_path=baseline_report,
        )
    finally:
        train_model.build_xgb_regressor = original_build_xgb_regressor

    assert (reports_dir / "model_metrics.json").exists()
    assert (reports_dir / "model_metrics.csv").exists()
    assert (reports_dir / "model_metrics_by_mode.csv").exists()
    assert (artifacts_dir / "xgb_delay_model.joblib").exists()

    payload = json.loads((reports_dir / "model_metrics.json").read_text())
    assert payload["target_column"] == "Min Delay"
    assert "Min Gap" not in payload["feature_columns"]
    assert payload["baseline_comparison"]["validation"]["baseline_mae"] == 8.0


def test_module_import_is_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    module = importlib.reload(train_model)

    assert hasattr(module, "run_training")
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "artifacts").exists()
