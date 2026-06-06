import importlib

import pandas as pd

from src.analysis import model_improvement_eda
from src.analysis.model_improvement_eda import (
    assign_delay_bucket,
    grouped_error_metrics,
    prior_count_support,
    recommendation_table,
    total_error_contribution,
)


def test_delay_bucket_assignment():
    buckets = assign_delay_bucket(pd.Series([0, 5, 6, 10, 11, 15, 16, 30, 31, 60, 61, 120, 121, 240]))

    assert buckets.tolist() == [
        "0-5",
        "0-5",
        "6-10",
        "6-10",
        "11-15",
        "11-15",
        "16-30",
        "16-30",
        "31-60",
        "31-60",
        "61-120",
        "61-120",
        "121-240",
        "121-240",
    ]


def test_grouped_error_metrics_and_contribution():
    frame = pd.DataFrame(
        {
            "split": ["validation", "validation", "validation", "test"],
            "mode": ["bus", "bus", "streetcar", "bus"],
            "actual": [10.0, 20.0, 30.0, 20.0],
            "prediction": [8.0, 25.0, 20.0, 22.0],
        }
    )
    frame["error"] = frame["prediction"] - frame["actual"]
    frame["absolute_error"] = frame["error"].abs()

    grouped = grouped_error_metrics(frame, "mode", min_group_size=2)

    assert len(grouped) == 1
    row = grouped.iloc[0]
    assert row["split"] == "validation"
    assert row["group_value"] == "bus"
    assert row["row_count"] == 2
    assert row["mae"] == 3.5
    assert row["total_error_contribution"] == 7.0
    assert total_error_contribution(4, 2.5) == 10.0


def test_prior_count_support_uses_only_strict_prior_timestamps():
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "validation", "validation", "test"],
            "ts": [
                "2022-01-01 08:00:00",
                "2022-01-01 08:00:00",
                "2023-01-01 08:00:00",
                "2023-01-01 08:00:00",
                "2024-01-01 08:00:00",
            ],
            "Route": ["1", "1", "1", "1", "1"],
            "Min Delay": [5, 10, 15, 20, 25],
        }
    )
    eval_mask = frame["split"].isin(["validation", "test"])

    support = prior_count_support(frame, eval_mask, "Route", ["Route"])

    assert support["row_count"] == 3
    assert support["pct_with_prior_1"] == 100.0
    assert support["pct_with_prior_5"] == 0.0
    assert support["median_prior_count"] == 2.0
    assert support["max_prior_count"] == 4


def test_recommendation_table_schema():
    table = recommendation_table()
    expected_columns = [
        "feature_name",
        "feature_type",
        "grouping",
        "target",
        "expected_value",
        "reason",
        "risks",
        "recommended_for_phase_11b",
        "priority_rank",
    ]

    assert list(table.columns) == expected_columns
    assert table["priority_rank"].is_monotonic_increasing
    assert table["recommended_for_phase_11b"].dtype == bool


def test_module_import_is_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    module = importlib.reload(model_improvement_eda)

    assert hasattr(module, "run_model_improvement_eda")
    assert not (tmp_path / "reports").exists()
