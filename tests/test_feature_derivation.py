from __future__ import annotations

import pytest

from src.api.feature_derivation import derive_is_holiday, derive_time_features


def test_timestamp_derives_all_time_fields():
    features = derive_time_features("2024-02-03T08:30:00")

    assert features["hour"] == 8
    assert features["day_of_week"] == 5
    assert features["month"] == 2
    assert features["is_weekend"] == 1
    assert features["day_of_year"] == 34
    assert "hour_sin" in features
    assert "hour_cos" in features
    assert "day_sin" in features
    assert "day_cos" in features
    assert features["is_holiday"] == 0


def test_known_ontario_holiday_returns_one():
    assert derive_is_holiday("2024-12-25T08:30:00") == 1
    assert derive_is_holiday("2024-07-01T08:30:00") == 1


def test_normal_weekday_returns_zero():
    assert derive_is_holiday("2024-02-06T08:30:00") == 0


def test_invalid_timestamp_raises_clear_error():
    with pytest.raises(ValueError, match="Invalid timestamp"):
        derive_time_features("not-a-date")
