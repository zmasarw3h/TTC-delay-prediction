import pandas as pd

from src.features.build_features import (
    FEATURE_COLUMNS,
    create_feature_metadata,
    create_modeling_dataset,
    build_feature_frame,
    split_modeling_dataset,
)


def _sample_frame():
    return pd.DataFrame(
        {
            "mode": ["bus", "bus", "bus", "streetcar", "streetcar"],
            "ts": [
                "2022-12-31 23:00:00",
                "2023-01-01 08:00:00",
                "2023-01-02 08:00:00",
                "2024-01-01 09:00:00",
                "2025-01-01 09:00:00",
            ],
            "Date": [
                "2022-12-31",
                "2023-01-01",
                "2023-01-02",
                "2024-01-01",
                "2025-01-01",
            ],
            "Route": ["1", "1", "1", "501", "501"],
            "Direction": ["N/B", "N/B", "N/B", "E/B", "E/B"],
            "Location": ["A", "A", "A", "B", "B"],
            "Incident": ["Delay", "Delay", "Delay", "Mechanical", "Mechanical"],
            "Min Delay": [10, 20, 30, 40, 300],
            "Min Gap": [99, 99, 99, 99, 99],
            "Vehicle": ["100", "101", "102", "200", "201"],
            "is_holiday": [0, 1, 0, 1, 0],
            "source_file": ["x.xlsx"] * 5,
            "source_sheet": [1] * 5,
        }
    )


def test_create_modeling_dataset_filters_target_threshold_without_mutating_source():
    source = _sample_frame()

    modeled = create_modeling_dataset(source, max_delay_minutes=240)

    assert len(modeled) == 4
    assert modeled["Min Delay"].max() == 40
    assert len(source) == 5


def test_split_modeling_dataset_uses_chronological_definitions():
    modeled = create_modeling_dataset(_sample_frame(), max_delay_minutes=240)

    splits = split_modeling_dataset(
        modeled,
        train_end="2022-12-31",
        val_year=2023,
        test_year=2024,
    )

    assert len(splits["train"]) == 1
    assert splits["train"]["ts"].dt.year.tolist() == [2022]
    assert splits["validation"]["ts"].dt.year.tolist() == [2023, 2023]
    assert splits["test"]["ts"].dt.year.tolist() == [2024]


def test_historical_features_do_not_use_current_row():
    featured = build_feature_frame(_sample_frame(), max_delay_minutes=240)

    first_route_hour = featured[
        (featured["Route"] == "1") & (featured["hour"] == 8)
    ].iloc[0]
    second_route_hour = featured[
        (featured["Route"] == "1") & (featured["hour"] == 8)
    ].iloc[1]

    assert pd.isna(first_route_hour["prior_route_hour_mean_delay"])
    assert second_route_hour["prior_route_hour_mean_delay"] == 20
    assert second_route_hour["prior_route_hour_7d_mean_delay"] == 20
    assert second_route_hour["prior_route_mean_delay"] == 15
    assert second_route_hour["prior_global_mean_delay"] == 15


def test_build_feature_frame_applies_categorical_normalization_before_history():
    frame = _sample_frame()
    frame.loc[0, "Route"] = 1.0
    frame.loc[1, "Route"] = "1"
    frame.loc[0, "Direction"] = "north"
    frame.loc[1, "Direction"] = "N/B"
    frame.loc[0, "Incident"] = "Delay"
    frame.loc[1, "Incident"] = "General Delay"
    frame.loc[0, "Location"] = "Kennedy Stn"

    featured = build_feature_frame(frame, max_delay_minutes=240)
    second_route = featured[featured["ts"] == pd.Timestamp("2023-01-01 08:00:00")].iloc[0]

    assert featured.loc[0, "Route"] == "1"
    assert featured.loc[0, "Direction"] == "N"
    assert featured.loc[0, "Incident"] == "General Delay"
    assert featured.loc[0, "Location"] == "KENNEDY STATION"
    assert featured.loc[0, "Route_raw"] == 1.0
    assert second_route["prior_route_mean_delay"] == 10
    assert second_route["prior_incident_mean_delay"] == 10


def test_historical_features_do_not_use_same_timestamp_rows():
    frame = pd.DataFrame(
        {
            "mode": ["bus", "bus", "bus", "bus"],
            "ts": [
                "2023-01-01 08:00:00",
                "2023-01-02 08:00:00",
                "2023-01-02 08:00:00",
                "2023-01-03 08:00:00",
            ],
            "Route": ["1", "1", "1", "1"],
            "Direction": ["N/B", "N/B", "N/B", "N/B"],
            "Location": ["A", "A", "A", "A"],
            "Incident": ["Delay", "Delay", "Delay", "Delay"],
            "Min Delay": [10, 20, 100, 40],
            "Min Gap": [99, 99, 99, 99],
            "Vehicle": ["100", "101", "102", "103"],
            "is_holiday": [0, 0, 0, 0],
        }
    )

    featured = build_feature_frame(frame, max_delay_minutes=240)
    same_timestamp_rows = featured[featured["ts"] == pd.Timestamp("2023-01-02 08:00:00")]
    next_timestamp_row = featured[featured["ts"] == pd.Timestamp("2023-01-03 08:00:00")].iloc[0]

    for column in [
        "prior_route_mean_delay",
        "prior_route_hour_mean_delay",
        "prior_incident_mean_delay",
        "prior_mode_mean_delay",
        "prior_global_mean_delay",
        "prior_route_hour_7d_mean_delay",
    ]:
        assert same_timestamp_rows[column].tolist() == [10.0, 10.0]

    assert next_timestamp_row["prior_route_mean_delay"] == 130 / 3
    assert next_timestamp_row["prior_route_hour_mean_delay"] == 130 / 3
    assert next_timestamp_row["prior_incident_mean_delay"] == 130 / 3
    assert next_timestamp_row["prior_mode_mean_delay"] == 130 / 3
    assert next_timestamp_row["prior_global_mean_delay"] == 130 / 3


def test_min_gap_is_excluded_from_main_feature_list():
    assert "Min Gap" not in FEATURE_COLUMNS


def test_create_feature_metadata_contains_expected_contract():
    featured = build_feature_frame(_sample_frame(), max_delay_minutes=240)
    splits = split_modeling_dataset(
        featured,
        train_end="2022-12-31",
        val_year=2023,
        test_year=2024,
    )

    metadata = create_feature_metadata(
        modeling_df=featured,
        splits=splits,
        max_delay_minutes=240,
        train_end="2022-12-31",
        val_year=2023,
        test_year=2024,
    )

    assert metadata["target_column"] == "Min Delay"
    assert metadata["max_delay_threshold"]["maximum_inclusive"] == 240
    assert "Min Gap" not in metadata["feature_columns"]
    assert "Min Gap" in metadata["leakage_sensitive_columns"]
    assert metadata["row_counts_by_split"] == {
        "train": 1,
        "validation": 2,
        "test": 1,
    }
    assert metadata["categorical_normalization"]["applied"] is True
    assert "Direction" in metadata["categorical_normalization"]["normalized_columns"]
    assert "Direction_raw" in metadata["categorical_normalization"]["raw_columns_preserved"]
    assert "prior_route_hour_7d_mean_delay" in metadata["historical_feature_definitions"]
