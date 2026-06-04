"""Target diagnostics for the cleaned TTC delay audit dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("data/processed/ttc_delays_cleaned.csv")
DEFAULT_OUTPUT_DIR = Path("reports/target_diagnostics")
CATEGORICAL_COLUMNS = ["Route", "Direction", "Location", "Incident", "Vehicle", "mode"]
TARGET_COLUMN = "Min Delay"
THRESHOLDS = [0, 60, 120, 240, 480, 1440]
QUANTILES = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]
TOP_OUTLIER_COLUMNS = [
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
    "source_file",
    "source_sheet",
]


def load_cleaned_delays(input_path: Path) -> pd.DataFrame:
    """Load the cleaned delay audit dataset with stable dtypes."""
    dtype = {column: "string" for column in CATEGORICAL_COLUMNS}
    return pd.read_csv(input_path, dtype=dtype, parse_dates=["ts"])


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def summarize_target(df: pd.DataFrame) -> dict[str, Any]:
    """Build JSON-serializable target diagnostics."""
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    summary_stats = target.describe().to_dict()
    quantiles = target.quantile(QUANTILES).to_dict()
    ts = pd.to_datetime(df["ts"], errors="coerce")

    return {
        "total_rows": int(len(df)),
        "rows_by_mode": {
            str(mode): int(count)
            for mode, count in df["mode"].value_counts(dropna=False).sort_index().items()
        },
        "date_range": {
            "min": _json_safe(ts.min()),
            "max": _json_safe(ts.max()),
        },
        "min_delay_summary": {
            str(key): _json_safe(value) for key, value in summary_stats.items()
        },
        "min_delay_quantiles": {
            str(key): _json_safe(value) for key, value in quantiles.items()
        },
    }


def compute_threshold_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Compute target threshold counts overall and by mode."""
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    rows: list[dict[str, Any]] = []

    def add_rows(scope: str, mode: str, mask: pd.Series, denominator: int) -> None:
        scoped_target = target[mask]
        for threshold in THRESHOLDS:
            if threshold == 0:
                count = int((scoped_target == 0).sum())
                condition = "Min Delay == 0"
            else:
                count = int((scoped_target > threshold).sum())
                condition = f"Min Delay > {threshold}"
            percentage = (count / denominator * 100) if denominator else 0.0
            rows.append(
                {
                    "scope": scope,
                    "mode": mode,
                    "threshold_minutes": threshold,
                    "condition": condition,
                    "count": count,
                    "percentage": percentage,
                    "denominator": denominator,
                }
            )

    add_rows(scope="overall", mode="all", mask=pd.Series(True, index=df.index), denominator=len(df))
    for mode, mode_index in df.groupby("mode", dropna=False).groups.items():
        mode_mask = df.index.isin(mode_index)
        add_rows(
            scope="mode",
            mode=str(mode),
            mask=pd.Series(mode_mask, index=df.index),
            denominator=int(mode_mask.sum()),
        )

    return pd.DataFrame(rows)


def top_delay_outliers(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Return the largest target records with audit columns."""
    output = df.copy()
    output[TARGET_COLUMN] = pd.to_numeric(output[TARGET_COLUMN], errors="coerce")
    for column in TOP_OUTLIER_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    return output.sort_values(TARGET_COLUMN, ascending=False).head(n)[TOP_OUTLIER_COLUMNS]


def write_diagnostics(input_path: Path, output_dir: Path) -> None:
    """Write target diagnostics files for a cleaned TTC delay dataset."""
    df = load_cleaned_delays(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_target(df)
    (output_dir / "target_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    compute_threshold_counts(df).to_csv(
        output_dir / "target_threshold_counts.csv",
        index=False,
    )
    top_delay_outliers(df).to_csv(
        output_dir / "top_delay_outliers.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Min Delay target diagnostics.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_diagnostics(input_path=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
