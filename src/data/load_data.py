"""Utilities for loading raw TTC delay files."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from zipfile import BadZipFile
import re
import warnings

import pandas as pd


SUPPORTED_EXTENSIONS = {".xlsx", ".xlsm", ".xls", ".xlsb", ".csv"}


def _column_key(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


COLUMN_ALIASES = {
    "reportdate": "Date",
    "date": "Date",
    "day": "Day",
    "time": "Time",
    "reporttime": "Time",
    "route": "Route",
    "line": "Route",
    "rte": "Route",
    "direction": "Direction",
    "bound": "Direction",
    "dir": "Direction",
    "location": "Location",
    "loc": "Location",
    "incident": "Incident",
    "incidenttype": "Incident",
    "delay": "Min Delay",
    "mindelay": "Min Delay",
    "minutessdelay": "Min Delay",
    "minutesdelay": "Min Delay",
    "gap": "Min Gap",
    "mingap": "Min Gap",
    "vehiclenumber": "Vehicle",
    "vehicle": "Vehicle",
    "vehicleid": "Vehicle",
}


def discover_delay_files(raw_dir: Path) -> list[Path]:
    """Return supported TTC delay files below ``raw_dir``."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw data path is not a directory: {raw_dir}")

    files = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.name.startswith("~$")
    ]
    return sorted(files)


def read_excel_any(path: Path) -> Iterator[pd.DataFrame]:
    """Yield DataFrames from a CSV or any supported Excel-like file."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".csv":
        yield pd.read_csv(path)
        return

    engine_by_ext = {
        ".xlsx": "openpyxl",
        ".xlsm": "openpyxl",
        ".xls": "xlrd",
        ".xlsb": "pyxlsb",
    }
    engine = engine_by_ext.get(ext)
    if engine is None:
        raise ValueError(f"Unsupported file extension for {path}")

    try:
        excel_file = pd.ExcelFile(path, engine=engine)
    except BadZipFile:
        warnings.warn(f"{path.name}: invalid modern Excel container; trying pandas fallback")
        excel_file = pd.ExcelFile(path)

    for sheet_name in excel_file.sheet_names:
        frame = pd.read_excel(excel_file, sheet_name=sheet_name)
        if not frame.empty:
            yield frame


def normalize_columns(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Normalize drifting TTC bus/streetcar column names."""
    normalized = df.copy()
    rename_map: dict[object, str] = {}
    seen: set[str] = set()

    for column in normalized.columns:
        canonical = COLUMN_ALIASES.get(_column_key(column), str(column).strip())
        if canonical in seen:
            continue
        rename_map[column] = canonical
        seen.add(canonical)

    normalized = normalized.rename(columns=rename_map)
    normalized = normalized.loc[:, ~normalized.columns.duplicated()]
    normalized["source_mode"] = mode
    return normalized


def load_ttc_delay_files(raw_dir: Path, mode: str) -> pd.DataFrame:
    """Load and combine all supported raw TTC delay files for one mode."""
    files = discover_delay_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No supported TTC delay files found in {raw_dir}")

    frames: list[pd.DataFrame] = []
    for path in files:
        for sheet_index, frame in enumerate(read_excel_any(path), start=1):
            normalized = normalize_columns(frame, mode=mode)
            normalized["source_file"] = path.name
            normalized["source_sheet"] = sheet_index
            frames.append(normalized)

    if not frames:
        raise ValueError(f"No readable rows found in supported files under {raw_dir}")

    return pd.concat(frames, ignore_index=True, sort=False)

