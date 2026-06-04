import pandas as pd

from src.features.diagnose_target import compute_threshold_counts, load_cleaned_delays


def test_load_cleaned_delays_uses_explicit_categorical_dtypes(tmp_path):
    input_path = tmp_path / "cleaned.csv"
    pd.DataFrame(
        {
            "mode": ["bus", "streetcar"],
            "ts": ["2024-01-01 08:00:00", "2024-01-01 09:00:00"],
            "Route": [501, "29A"],
            "Direction": ["E/B", "W/B"],
            "Location": ["Queen", "King"],
            "Incident": ["Mechanical", "Delay"],
            "Min Delay": [10, 20],
            "Vehicle": [1234, "abcd"],
        }
    ).to_csv(input_path, index=False)

    loaded = load_cleaned_delays(input_path)

    assert str(loaded["Route"].dtype) == "string"
    assert str(loaded["Vehicle"].dtype) == "string"
    assert pd.api.types.is_datetime64_any_dtype(loaded["ts"])


def test_compute_threshold_counts_overall_and_by_mode():
    frame = pd.DataFrame(
        {
            "mode": ["bus", "bus", "streetcar", "streetcar"],
            "Min Delay": [0, 61, 241, 1500],
        }
    )

    counts = compute_threshold_counts(frame)

    overall_zero = counts[
        (counts["scope"] == "overall") & (counts["condition"] == "Min Delay == 0")
    ].iloc[0]
    assert overall_zero["count"] == 1
    assert overall_zero["percentage"] == 25.0

    overall_gt_240 = counts[
        (counts["scope"] == "overall") & (counts["condition"] == "Min Delay > 240")
    ].iloc[0]
    assert overall_gt_240["count"] == 2
    assert overall_gt_240["percentage"] == 50.0

    bus_gt_60 = counts[
        (counts["mode"] == "bus") & (counts["condition"] == "Min Delay > 60")
    ].iloc[0]
    assert bus_gt_60["count"] == 1
    assert bus_gt_60["denominator"] == 2

    streetcar_gt_1440 = counts[
        (counts["mode"] == "streetcar") & (counts["condition"] == "Min Delay > 1440")
    ].iloc[0]
    assert streetcar_gt_1440["count"] == 1
    assert streetcar_gt_1440["percentage"] == 50.0
