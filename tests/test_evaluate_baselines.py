import importlib

import pandas as pd

from src.models import evaluate_baselines
from src.models.evaluate_baselines import (
    BASELINE_COLUMNS,
    calculate_metrics,
    fill_predictions_with_fallbacks,
    run_evaluation,
    select_best_baseline,
)


def _split_frame(delays, predictions=None):
    predictions = predictions or {}
    frame = pd.DataFrame(
        {
            "Min Delay": delays,
            "mode": ["bus", "streetcar"][: len(delays)],
            "Route": ["1", "501"][: len(delays)],
            "Direction": ["N/B", "E/B"][: len(delays)],
            "Incident": ["Delay", "Mechanical"][: len(delays)],
            "Location": ["A", "B"][: len(delays)],
        }
    )
    for column in BASELINE_COLUMNS:
        frame[column] = predictions.get(column, [10.0] * len(delays))
    return frame


def test_metric_calculation_works():
    metrics = calculate_metrics(
        y_true=pd.Series([10, 20]),
        y_pred=pd.Series([12, 18]),
    )

    assert metrics["mae"] == 2.0
    assert metrics["rmse"] == 2.0
    assert round(metrics["r2"], 2) == 0.84


def test_fallback_filling_works():
    frame = pd.DataFrame(
        {
            "baseline": [pd.NA, pd.NA, 7.0],
            "prior_route_mean_delay": [pd.NA, pd.NA, 1.0],
            "prior_mode_mean_delay": [8.0, pd.NA, 2.0],
            "prior_global_mean_delay": [9.0, pd.NA, 3.0],
        }
    )

    filled = fill_predictions_with_fallbacks(
        df=frame,
        baseline_column="baseline",
        train_target_mean=15.0,
    )

    assert filled.tolist() == [8.0, 15.0, 7.0]


def test_best_baseline_selection_prefers_filled_validation_candidate():
    metrics = pd.DataFrame(
        [
            {
                "baseline": "a",
                "evaluation_name": "a",
                "split": "validation",
                "filled": False,
                "mae": 1.0,
                "rmse": 1.0,
                "r2": 0.0,
            },
            {
                "baseline": "b",
                "evaluation_name": "b",
                "split": "validation",
                "filled": False,
                "mae": 2.0,
                "rmse": 2.0,
                "r2": 0.0,
            },
            {
                "baseline": "b",
                "evaluation_name": "b_filled",
                "split": "validation",
                "filled": True,
                "mae": 0.5,
                "rmse": 0.5,
                "r2": 0.5,
            },
            {
                "baseline": "c",
                "evaluation_name": "c",
                "split": "test",
                "filled": False,
                "mae": 0.1,
                "rmse": 0.1,
                "r2": 0.9,
            },
        ]
    )

    best = select_best_baseline(metrics)

    assert best["baseline"] == "b"
    assert best["filled"] is True


def test_run_evaluation_writes_expected_files(tmp_path):
    modeling_dir = tmp_path / "modeling"
    output_dir = tmp_path / "baselines"
    modeling_dir.mkdir()

    train = _split_frame([10, 20])
    validation = _split_frame(
        [10, 30],
        predictions={
            "prior_mode_mean_delay": [10.0, 29.0],
            "prior_route_mean_delay": [pd.NA, 32.0],
            "prior_route_hour_7d_mean_delay": [pd.NA, 28.0],
        },
    )
    test = _split_frame(
        [12, 24],
        predictions={
            "prior_mode_mean_delay": [11.0, 25.0],
            "prior_route_mean_delay": [13.0, pd.NA],
            "prior_route_hour_7d_mean_delay": [pd.NA, 23.0],
        },
    )

    train.to_csv(modeling_dir / "train.csv", index=False)
    validation.to_csv(modeling_dir / "validation.csv", index=False)
    test.to_csv(modeling_dir / "test.csv", index=False)

    result = run_evaluation(modeling_dir=modeling_dir, output_dir=output_dir)

    assert (output_dir / "baseline_metrics.json").exists()
    assert (output_dir / "baseline_metrics.csv").exists()
    assert (output_dir / "best_baseline_by_mode.csv").exists()
    assert not result["metrics"].empty
    assert {"validation", "test"} == set(result["mode_breakdown"]["split"])


def test_module_import_is_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    module = importlib.reload(evaluate_baselines)

    assert hasattr(module, "run_evaluation")
    assert not (tmp_path / "reports").exists()
