"""Command-line TTC delay data cleaner."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
from pathlib import Path

import pandas as pd

from src.data.load_data import load_ttc_delay_files


LOGGER = logging.getLogger(__name__)

TEXT_COLUMNS = ["Route", "Direction", "Location", "Incident", "Vehicle"]
OUTPUT_COLUMNS = [
    "mode",
    "ts",
    "Date",
    "Route",
    "Direction",
    "Location",
    "Incident",
    "Min Delay",
    "Min Gap",
    "Vehicle",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "source_file",
    "source_sheet",
]


def _clean_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text or text.lower() in {"nan", "none", "nat"}:
        return pd.NA
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def _parse_time(value: object) -> dt.time | None:
    if pd.isna(value):
        return None
    if isinstance(value, dt.datetime):
        return value.time().replace(microsecond=0)
    if isinstance(value, dt.time):
        return value.replace(microsecond=0)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if 0 <= float(value) < 1:
            seconds = int(round(float(value) * 24 * 60 * 60))
            seconds %= 24 * 60 * 60
            return (dt.datetime.min + dt.timedelta(seconds=seconds)).time()

    text = str(value).strip()
    if not text or not re.search(r"\d", text):
        return None

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return parsed.time().replace(microsecond=0)
    return None


def build_timestamp(df: pd.DataFrame) -> pd.Series:
    """Combine TTC report date and time columns into a timestamp."""
    if "Date" not in df.columns:
        raise ValueError("Required date column not found after normalization.")

    date_values = pd.to_datetime(df["Date"], errors="coerce")
    if "Time" not in df.columns:
        return date_values

    times = df["Time"].map(_parse_time)
    midnight = dt.time(0, 0)
    combined = [
        pd.NaT
        if pd.isna(date_value)
        else dt.datetime.combine(date_value.date(), time_value or midnight)
        for date_value, time_value in zip(date_values, times, strict=False)
    ]
    return pd.to_datetime(pd.Series(combined, index=df.index), errors="coerce")


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday_before(year: int, month: int, day: int, weekday: int) -> dt.date:
    current = dt.date(year, month, day)
    while current.weekday() != weekday:
        current -= dt.timedelta(days=1)
    return current


def _easter_date(year: int) -> dt.date:
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
    return dt.date(year, month, day)


def ontario_holidays(years: list[int]) -> set[dt.date]:
    """Return Ontario holidays, using the holidays package when available."""
    try:
        import holidays

        return set(holidays.country_holidays("CA", subdiv="ON", years=years).keys())
    except ImportError:
        LOGGER.warning("Package `holidays` is not installed; using built-in Ontario fallback.")

    dates: set[dt.date] = set()
    for year in years:
        dates.update(
            {
                dt.date(year, 1, 1),
                _nth_weekday(year, 2, weekday=0, n=3),
                _easter_date(year) - dt.timedelta(days=2),
                _last_weekday_before(year, 5, 25, weekday=0),
                dt.date(year, 7, 1),
                _nth_weekday(year, 9, weekday=0, n=1),
                _nth_weekday(year, 10, weekday=0, n=2),
                dt.date(year, 12, 25),
                dt.date(year, 12, 26),
            }
        )
    return dates


def clean_delay_frame(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Standardize and clean a loaded TTC delay DataFrame for one mode."""
    cleaned = df.copy()
    cleaned["mode"] = mode
    cleaned["ts"] = build_timestamp(cleaned)
    cleaned["Date"] = cleaned["ts"].dt.date

    if "Min Delay" not in cleaned.columns:
        raise ValueError("Required target column `Min Delay` not found after normalization.")

    cleaned["Min Delay"] = pd.to_numeric(cleaned["Min Delay"], errors="coerce")
    if "Min Gap" in cleaned.columns:
        cleaned["Min Gap"] = pd.to_numeric(cleaned["Min Gap"], errors="coerce")
    else:
        cleaned["Min Gap"] = pd.NA

    for column in TEXT_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = pd.NA
        cleaned[column] = cleaned[column].map(_clean_text)

    before_rows = len(cleaned)
    cleaned = cleaned.dropna(subset=["ts", "Min Delay"])
    cleaned = cleaned[cleaned["Min Delay"] >= 0].copy()
    dropped_rows = before_rows - len(cleaned)

    cleaned["hour"] = cleaned["ts"].dt.hour
    cleaned["day_of_week"] = cleaned["ts"].dt.dayofweek
    cleaned["month"] = cleaned["ts"].dt.month
    cleaned["is_weekend"] = cleaned["day_of_week"].isin([5, 6]).astype(int)

    years = sorted(cleaned["ts"].dt.year.dropna().astype(int).unique())
    cleaned["is_holiday"] = cleaned["ts"].dt.date.isin(ontario_holidays(years)).astype(int)

    for column in OUTPUT_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = pd.NA

    LOGGER.info("%s: dropped %s invalid rows", mode, dropped_rows)
    return cleaned[OUTPUT_COLUMNS].sort_values("ts").reset_index(drop=True)


def process_mode(raw_dir: Path | None, processed_dir: Path, mode: str) -> pd.DataFrame | None:
    """Load, clean, and write one mode if its raw directory is available."""
    if raw_dir is None:
        LOGGER.info("%s: no raw directory provided; skipping", mode)
        return None
    if not raw_dir.exists():
        LOGGER.warning("%s: raw directory does not exist; skipping: %s", mode, raw_dir)
        return None

    raw = load_ttc_delay_files(raw_dir, mode=mode)
    LOGGER.info("%s: loaded %s rows from %s", mode, len(raw), raw_dir)

    cleaned = clean_delay_frame(raw, mode=mode)
    output_path = processed_dir / f"{mode}_delays_cleaned.csv"
    cleaned.to_csv(output_path, index=False)
    LOGGER.info("%s: saved %s rows to %s", mode, len(cleaned), output_path)
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw TTC bus/streetcar delay files.")
    parser.add_argument("--bus-raw-dir", type=Path, default=None)
    parser.add_argument("--streetcar-raw-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for mode, raw_dir in (("bus", args.bus_raw_dir), ("streetcar", args.streetcar_raw_dir)):
        result = process_mode(raw_dir=raw_dir, processed_dir=args.processed_dir, mode=mode)
        if result is not None:
            outputs.append(result)

    if not outputs:
        raise SystemExit("No data processed. Provide at least one existing raw data directory with TTC files.")

    combined = pd.concat(outputs, ignore_index=True, sort=False).sort_values(["ts", "mode"])
    combined_path = args.processed_dir / "ttc_delays_cleaned.csv"
    combined.to_csv(combined_path, index=False)
    LOGGER.info("combined: saved %s rows to %s", len(combined), combined_path)


if __name__ == "__main__":
    main()
