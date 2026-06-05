"""Timestamp-based feature derivation utilities for API payloads."""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import pandas as pd


TIME_FEATURE_FIELDS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]


def parse_prediction_timestamp(value: Any) -> pd.Timestamp:
    """Parse a prediction timestamp or raise a clear validation error."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timestamp '{value}'. Provide an ISO-like datetime string.") from exc
    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp '{value}'. Provide an ISO-like datetime string.")
    return timestamp


def derive_time_features(timestamp: Any) -> dict[str, int | float]:
    """Derive model time features, including the Ontario holiday flag."""
    ts = parse_prediction_timestamp(timestamp)
    return {
        "hour": int(ts.hour),
        "day_of_week": int(ts.dayofweek),
        "month": int(ts.month),
        "is_weekend": int(ts.dayofweek in {5, 6}),
        "day_of_year": int(ts.dayofyear),
        "hour_sin": math.sin(2 * math.pi * ts.hour / 24),
        "hour_cos": math.cos(2 * math.pi * ts.hour / 24),
        "day_sin": math.sin(2 * math.pi * ts.dayofyear / 366),
        "day_cos": math.cos(2 * math.pi * ts.dayofyear / 366),
        "is_holiday": derive_is_holiday(ts),
    }


def derive_is_holiday(timestamp: Any) -> int:
    """Return 1 when the timestamp date is an Ontario-relevant holiday."""
    ts = parse_prediction_timestamp(timestamp)
    day = ts.date()
    try:
        import holidays

        ontario_holidays = holidays.country_holidays("CA", subdiv="ON", years=[day.year])
        return int(day in ontario_holidays)
    except Exception:
        return int(day in _fallback_ontario_holidays(day.year))


def _fallback_ontario_holidays(year: int) -> set[date]:
    easter = _easter_sunday(year)
    return {
        date(year, 1, 1),
        _nth_weekday(year, 2, weekday=0, n=3),
        easter - timedelta(days=2),
        _last_monday_on_or_before(year, 5, 24),
        date(year, 7, 1),
        _nth_weekday(year, 8, weekday=0, n=1),
        _nth_weekday(year, 9, weekday=0, n=1),
        _nth_weekday(year, 10, weekday=0, n=2),
        date(year, 12, 25),
        date(year, 12, 26),
    }


def _nth_weekday(year: int, month: int, *, weekday: int, n: int) -> date:
    current = date(year, month, 1)
    days_until_weekday = (weekday - current.weekday()) % 7
    return current + timedelta(days=days_until_weekday + (n - 1) * 7)


def _last_monday_on_or_before(year: int, month: int, day: int) -> date:
    current = date(year, month, day)
    return current - timedelta(days=(current.weekday() - 0) % 7)


def _easter_sunday(year: int) -> date:
    """Compute Gregorian Easter Sunday using the Meeus/Jones/Butcher algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)
