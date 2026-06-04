import pandas as pd

from src.data.clean_data import clean_delay_frame
from src.data.load_data import normalize_columns


def test_normalize_columns_handles_ttc_aliases():
    frame = pd.DataFrame(
        {
            "Report Date": ["2024-01-01"],
            "Line": ["501"],
            "Bound": ["E/B"],
            "Delay": [12],
            "Gap": [20],
        }
    )

    normalized = normalize_columns(frame, mode="streetcar")

    assert {"Date", "Route", "Direction", "Min Delay", "Min Gap"}.issubset(
        normalized.columns
    )


def test_clean_delay_frame_drops_invalid_rows_and_adds_time_fields():
    frame = pd.DataFrame(
        {
            "Date": ["2024-01-01", "not a date", "2024-01-03", "2024-01-04"],
            "Time": ["08:30", "09:00", "10:00", "11:00"],
            "Route": ["501.0", "29", "7", "63"],
            "Direction": [" E/B ", "N/B", "S/B", "W/B"],
            "Location": [" Queen  and  Bay ", "x", "y", "z"],
            "Incident": ["Mechanical", "Delay", "Delay", "Delay"],
            "Min Delay": ["15", "20", "-1", "bad"],
            "Min Gap": ["30", "40", "50", "60"],
            "Vehicle": [1234.0, None, None, None],
        }
    )

    cleaned = clean_delay_frame(frame, mode="streetcar")

    assert len(cleaned) == 1
    assert cleaned.loc[0, "mode"] == "streetcar"
    assert cleaned.loc[0, "Route"] == "501"
    assert cleaned.loc[0, "Location"] == "Queen and Bay"
    assert cleaned.loc[0, "hour"] == 8
    assert cleaned.loc[0, "is_holiday"] == 1

