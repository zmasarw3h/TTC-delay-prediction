import importlib
import json

import joblib
import pandas as pd

from src.models import analyze_errors
from src.models.analyze_errors import (
    assign_delay_bucket,
    grouped_breakdown,
    high_delay_performance,
    prediction_error_frame,
    run_error_analysis,
    worst_predictions,
)


class DummyModel:
    def predict(self, x):
        return pd.to_numeric(x["feature"], errors="coerce") + 1


def _metadata():
    return {
        "target_column": "Min Delay",
        "feature_columns": ["feature", "mode", "Route", "Min Delay", "Min Gap", "ts"],
        "categorical_columns": ["mode", "Route"],
        "numeric_columns": ["feature", "Min Delay", "Min Gap"],
        "excluded_columns": ["Min Delay", "Min Gap", "ts"],
    }


def _frame(delays, features=None):
    features = features or list(range(len(delays)))
    return pd.DataFrame(
        {
            "feature": features,
            "Min Delay": delays,
            "mode": ["bus", "bus", "streetcar", "streetcar"][: len(delays)],
            "Route": ["1", "1", "501", "501"][: len(delays)],
            "Direction": ["N/B", "S/B", "E/B", "W/B"][: len(delays)],
            "Incident": ["Delay", "Delay", "Mechanical", "Mechanical"][: len(delays)],
            "Location": ["A", "B", "C", "D"][: len(delays)],
            "hour": [8, 9, 10, 11][: len(delays)],
            "month": [1, 1, 2, 2][: len(delays)],
            "ts": [f"2023-01-0{i + 1} 08:00:00" for i in range(len(delays))],
            "Min Gap": [999] * len(delays),
            "prior_route_mean_delay": [10.0] * len(delays),
            "prior_mode_mean_delay": [9.0] * len(delays),
            "prior_global_mean_delay": [8.0] * len(delays),
        }
    )


def test_delay_bucket_assignment():
    buckets = assign_delay_bucket(pd.Series([0, 5, 6, 10, 11, 30, 61, 240, 241]))

    assert buckets.tolist() == [
        "0-5",
        "0-5",
        "6-10",
        "6-10",
        "11-15",
        "16-30",
        "61-120",
        "121-240",
        "241+",
    ]


def test_prediction_error_frame_creation():
    frame = prediction_error_frame(
        model=DummyModel(),
        df=_frame([2, 5], features=[1, 3]),
        split_name="validation",
        feature_columns=["feature", "mode", "Route"],
        target_column="Min Delay",
        baseline_predictions=pd.Series([2.5, 4.0]),
    )

    assert frame["prediction"].tolist() == [2.0, 4.0]
    assert frame["actual"].tolist() == [2, 5]
    assert frame["error"].tolist() == [0.0, -1.0]
    assert frame["absolute_error"].tolist() == [0.0, 1.0]
    assert frame["baseline_error"].tolist() == [0.5, -1.0]


def test_grouped_breakdown_metrics_and_min_group_size():
    frame = prediction_error_frame(
        model=DummyModel(),
        df=_frame([2, 5, 20, 25], features=[1, 3, 10, 20]),
        split_name="test",
        feature_columns=["feature", "mode", "Route"],
        target_column="Min Delay",
    )

    breakdown = grouped_breakdown(frame, "mode", min_group_size=2)

    assert set(breakdown["mode"]) == {"bus", "streetcar"}
    bus = breakdown[breakdown["mode"] == "bus"].iloc[0]
    assert bus["row_count"] == 2
    assert bus["mae"] == 0.5
    assert grouped_breakdown(frame, "Route", min_group_size=3).empty


def test_high_delay_metrics():
    frame = prediction_error_frame(
        model=DummyModel(),
        df=_frame([10, 20, 40, 80], features=[9, 10, 35, 50]),
        split_name="validation",
        feature_columns=["feature", "mode", "Route"],
        target_column="Min Delay",
    )

    metrics = high_delay_performance(frame, thresholds=[30])
    record = metrics.iloc[0]

    assert record["actual_high_delay_rows"] == 2
    assert record["actual_high_delay_rate"] == 0.5
    assert record["underpredicted_high_delay_percent"] == 100.0
    assert record["average_underprediction_amount"] == 16.5


def test_worst_prediction_extraction():
    frame = prediction_error_frame(
        model=DummyModel(),
        df=_frame([1, 10, 100], features=[1, 8, 50]),
        split_name="test",
        feature_columns=["feature", "mode", "Route"],
        target_column="Min Delay",
    )

    worst = worst_predictions(frame, "test", n=2)

    assert len(worst) == 2
    assert worst.iloc[0]["actual"] == 100
    assert list(worst.columns) == analyze_errors.WORST_PREDICTION_COLUMNS


def test_run_error_analysis_writes_expected_files(tmp_path):
    modeling_dir = tmp_path / "modeling"
    output_dir = tmp_path / "error_analysis"
    model_path = tmp_path / "model.joblib"
    baseline_report = tmp_path / "baseline_metrics.json"
    modeling_dir.mkdir()

    (modeling_dir / "feature_metadata.json").write_text(
        json.dumps(_metadata()),
        encoding="utf-8",
    )
    _frame([1, 2, 3, 4], features=[0, 1, 2, 3]).to_csv(modeling_dir / "train.csv", index=False)
    _frame([2, 5, 20, 25], features=[1, 3, 10, 20]).to_csv(
        modeling_dir / "validation.csv",
        index=False,
    )
    _frame([10, 20, 40, 80], features=[9, 10, 35, 50]).to_csv(
        modeling_dir / "test.csv",
        index=False,
    )
    baseline_report.write_text(
        json.dumps(
            {
                "best_baseline": {
                    "baseline": "prior_route_mean_delay",
                    "filled": True,
                    "evaluation_name": "prior_route_mean_delay_filled",
                }
            }
        ),
        encoding="utf-8",
    )
    joblib.dump(DummyModel(), model_path)

    result = run_error_analysis(
        modeling_dir=modeling_dir,
        model_path=model_path,
        baseline_report=baseline_report,
        output_dir=output_dir,
        min_group_size=2,
    )

    expected_files = [
        "error_summary.json",
        "error_summary.csv",
        "error_by_mode.csv",
        "error_by_route.csv",
        "error_by_incident.csv",
        "error_by_hour.csv",
        "error_by_month.csv",
        "error_by_delay_bucket.csv",
        "high_delay_performance.csv",
        "worst_predictions_validation.csv",
        "worst_predictions_test.csv",
    ]
    for file_name in expected_files:
        assert (output_dir / file_name).exists()
    assert set(result["frames"]) == {"validation", "test"}
    summary_payload = json.loads((output_dir / "error_summary.json").read_text())
    assert summary_payload["note"] == "Error analysis only; no model training is performed."


def test_module_import_is_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    module = importlib.reload(analyze_errors)

    assert hasattr(module, "run_error_analysis")
    assert not (tmp_path / "reports").exists()
