"""Audit categorical value quality in the TTC modeling dataset.

This module is import-safe: report generation only runs through ``main()``.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.categorical_normalization import (
    INCIDENT_CATEGORIES,
    UNKNOWN_CATEGORY,
    normalize_route,
)


DEFAULT_INPUT = Path("data/processed/modeling/modeling_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("reports/category_audit")
AUDIT_COLUMNS = ("mode", "Route", "Direction", "Incident", "Location")
VALUE_AUDIT_COLUMNS = ("Direction", "Route", "Incident", "Location")
HEALTHY_DIRECTIONS = {"N", "S", "E", "W", "B", "UNKNOWN"}
ROUTE_PATTERN = re.compile(r"^(?:\d{1,4}[A-Z]{0,2}|RAD)$")
NULL_LIKE_TEXT = {"", "nan", "none", "null", "unknown", "n/a", "na"}
LOCATION_WORDS = {
    "station",
    "street",
    "st",
    "avenue",
    "ave",
    "road",
    "rd",
    "loop",
    "garage",
    "terminal",
    "yard",
    "division",
    "and",
    "&",
    "at",
    "north",
    "south",
    "east",
    "west",
}
INCIDENT_WORDS = {
    "mechanical",
    "mech",
    "diversion",
    "delay",
    "held",
    "late",
    "operations",
    "operator",
    "investigation",
    "security",
    "collision",
    "emergency",
    "cleaning",
    "utilized",
    "off route",
    "general",
}
DIRECTION_POLLUTION_WORDS = {
    "mechanical",
    "diversion",
    "delay",
    "held",
    "street",
    "station",
    "loop",
    "garage",
    "mins",
    "late",
}
INCIDENT_FRAGMENT_GROUPS = {
    "mechanical": {"mech", "mechanical", "mechanical delay"},
    "general_delay": {"general", "general delay", "delay"},
    "operations": {"operations", "operation", "operator", "operations - operator"},
}
COMMON_INCIDENT_LABELS = {
    category.lower() for category in INCIDENT_CATEGORIES if category != UNKNOWN_CATEGORY
}


@dataclass(frozen=True)
class ValueAudit:
    column: str
    value: str
    count: int
    percent: float
    suspicious: bool
    reasons: list[str]
    looks_like_fields: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit category health in the TTC modeling dataset."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Modeling CSV to audit.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated audit reports.",
    )
    parser.add_argument("--top-n", type=int, default=25, help="Top values to report per column.")
    return parser.parse_args()


def is_route_like(value: object) -> bool:
    """Return True for route identifiers suitable for route option lists."""
    return normalize_route(value) != UNKNOWN_CATEGORY


def audit_dataframe(df: pd.DataFrame, *, top_n: int = 25) -> dict[str, object]:
    missing_columns = [column for column in AUDIT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Input data is missing required column(s): {missing_columns}")

    total_rows = int(len(df))
    value_sets = {
        column: {
            _clean_value(value).lower()
            for value in df[column].dropna().unique()
            if _clean_value(value)
        }
        for column in AUDIT_COLUMNS
    }
    incident_fragment_groups = _active_incident_fragment_groups(value_sets["Incident"])

    summary: dict[str, object] = {}
    top_rows: list[dict[str, object]] = []
    suspicious_rows: list[dict[str, object]] = []
    column_audits: dict[str, list[dict[str, object]]] = {}

    for column in AUDIT_COLUMNS:
        series = df[column]
        missing_count = int(series.isna().sum() + series.dropna().map(_is_null_like).sum())
        counts = _value_counts(series)
        audits: list[ValueAudit] = []
        for value, count in counts.items():
            percent = _percent(int(count), total_rows)
            audits.append(
                audit_value(
                    column,
                    value,
                    int(count),
                    percent,
                    value_sets=value_sets,
                    incident_fragment_groups=incident_fragment_groups,
                )
            )

        top_values = [
            {
                "column": column,
                "value": value,
                "count": int(count),
                "percent": _percent(int(count), total_rows),
            }
            for value, count in list(counts.items())[:top_n]
        ]
        top_rows.extend(top_values)

        suspicious = [audit for audit in audits if audit.suspicious]
        suspicious_rows.extend(_audit_to_row(audit) for audit in suspicious)
        column_audits[column] = [_audit_to_row(audit) for audit in audits]
        summary[column] = {
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_percent": _percent(missing_count, total_rows),
            "unique_count": int(len(counts)),
            "top_values": top_values,
            "suspicious_values_count": int(len(suspicious)),
            "suspicious_rows_count": int(sum(audit.count for audit in suspicious)),
            "suspicious_examples": [_audit_to_row(audit) for audit in suspicious[:10]],
        }

    return {
        "summary": summary,
        "top_rows": top_rows,
        "suspicious_rows": suspicious_rows,
        "column_audits": column_audits,
    }


def audit_value(
    column: str,
    value: object,
    count: int,
    percent: float,
    *,
    value_sets: dict[str, set[str]] | None = None,
    incident_fragment_groups: dict[str, set[str]] | None = None,
) -> ValueAudit:
    text = _clean_value(value)
    lower = text.lower()
    reasons: list[str] = []
    looks_like_fields: list[str] = []

    if column == "Direction":
        reasons.extend(_direction_reasons(text))
    elif column == "Route":
        reasons.extend(_route_reasons(text))
    elif column == "Incident":
        reasons.extend(_incident_reasons(text, count, incident_fragment_groups or {}))
    elif column == "Location":
        reasons.extend(_location_reasons(text))
    elif column == "mode":
        if lower not in {"bus", "streetcar"}:
            reasons.append("unexpected mode value")

    looks_like_fields = _looks_like_other_fields(column, text, value_sets or {})
    suspicious = bool(reasons or looks_like_fields)
    return ValueAudit(
        column=column,
        value=text,
        count=count,
        percent=percent,
        suspicious=suspicious,
        reasons=reasons,
        looks_like_fields=looks_like_fields,
    )


def write_reports(
    audit: dict[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "category_summary.json").write_text(
        json.dumps(audit["summary"], indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(audit["top_rows"]).to_csv(
        output_dir / "category_top_values.csv",
        index=False,
    )
    pd.DataFrame(audit["suspicious_rows"]).to_csv(
        output_dir / "category_suspicious_values.csv",
        index=False,
    )
    column_audits = audit["column_audits"]
    if not isinstance(column_audits, dict):
        raise TypeError("column_audits must be a dictionary.")
    for column in VALUE_AUDIT_COLUMNS:
        rows = column_audits.get(column, [])
        pd.DataFrame(rows).to_csv(
            output_dir / f"{column.lower()}_value_audit.csv",
            index=False,
        )


def run_audit(input_path: Path, output_dir: Path, *, top_n: int = 25) -> dict[str, object]:
    df = pd.read_csv(
        input_path,
        usecols=list(AUDIT_COLUMNS),
        dtype={column: "string" for column in AUDIT_COLUMNS},
        keep_default_na=True,
        low_memory=False,
    )
    audit = audit_dataframe(df, top_n=top_n)
    write_reports(audit, output_dir)
    return audit


def _direction_reasons(value: str) -> list[str]:
    reasons: list[str] = []
    upper = value.upper()
    lower = value.lower()
    if _is_null_like(value):
        return reasons
    if value != upper and upper in HEALTHY_DIRECTIONS:
        reasons.append("lowercase direction variant should be normalized")
    if len(value) > 3:
        reasons.append("longer than 3 characters")
    if any(char.isdigit() for char in value):
        reasons.append("contains digits")
    if any(word in lower for word in DIRECTION_POLLUTION_WORDS):
        reasons.append("contains incident or location word")
    if upper not in HEALTHY_DIRECTIONS and len(value) <= 3 and value != upper:
        reasons.append("lowercase or mixed-case uncommon direction")
    return reasons


def _route_reasons(value: str) -> list[str]:
    lower = value.lower()
    reasons: list[str] = []
    if _is_null_like(value):
        reasons.append("null-like route text")
    if is_route_like(value):
        return reasons
    if len(value) > 8:
        reasons.append("long route value")
    if len(value.split()) >= 2:
        reasons.append("many-word route value")
    if _contains_any(lower, LOCATION_WORDS):
        reasons.append("looks like a location or intersection")
    if _contains_any(lower, INCIDENT_WORDS):
        reasons.append("looks like an incident label")
    if not reasons:
        reasons.append("not route-like")
    return reasons


def _incident_reasons(
    value: str,
    count: int,
    incident_fragment_groups: dict[str, set[str]],
) -> list[str]:
    lower = value.lower()
    reasons: list[str] = []
    if _is_null_like(value) or lower in COMMON_INCIDENT_LABELS:
        return reasons
    if is_route_like(value):
        reasons.append("route-like incident value")
    if _contains_any(lower, LOCATION_WORDS):
        reasons.append("location-like incident value")
    if count <= 2 and len(value) > 2 and lower not in COMMON_INCIDENT_LABELS:
        reasons.append("extremely rare incident label")
    for group_name, variants in incident_fragment_groups.items():
        if lower in variants:
            reasons.append(f"possible fragmented incident label: {group_name}")
            break
    return reasons


def _location_reasons(value: str) -> list[str]:
    lower = value.lower()
    reasons: list[str] = []
    if _is_null_like(value):
        return reasons
    if len(value) <= 2:
        reasons.append("too short for a useful location")
    if is_route_like(value):
        reasons.append("route-like location value")
    if _contains_any(lower, INCIDENT_WORDS):
        reasons.append("incident-like location value")
    return reasons


def _looks_like_other_fields(
    column: str,
    value: str,
    value_sets: dict[str, set[str]],
) -> list[str]:
    if column == "mode":
        return []
    lower = value.lower()
    fields: list[str] = []
    if not lower or _is_null_like(lower):
        return fields
    for other_column, values in value_sets.items():
        if other_column == column:
            continue
        if lower in values:
            fields.append(other_column)
    if column != "Route" and is_route_like(value):
        fields.append("Route")
    if column != "Direction" and value.upper() in HEALTHY_DIRECTIONS and value.upper() != "UNKNOWN":
        fields.append("Direction")
    return sorted(set(fields))


def _active_incident_fragment_groups(values: set[str]) -> dict[str, set[str]]:
    active: dict[str, set[str]] = {}
    for group_name, variants in INCIDENT_FRAGMENT_GROUPS.items():
        present = variants & values
        if len(present) >= 2:
            active[group_name] = present
    return active


def _value_counts(series: pd.Series) -> dict[str, int]:
    cleaned = series.dropna().map(_clean_value)
    cleaned = cleaned[~cleaned.map(_is_null_like)]
    return cleaned.value_counts().to_dict()


def _clean_value(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def _is_null_like(value: object) -> bool:
    return _clean_value(value).lower() in NULL_LIKE_TEXT


def _contains_any(value: str, words: Iterable[str]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", value) for word in words)


def _percent(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(100.0 * count / total, 4)


def _audit_to_row(audit: ValueAudit) -> dict[str, object]:
    return {
        "column": audit.column,
        "value": audit.value,
        "count": audit.count,
        "percent": audit.percent,
        "suspicious": audit.suspicious,
        "reasons": "; ".join(audit.reasons),
        "looks_like_fields": "; ".join(audit.looks_like_fields),
    }


def main() -> None:
    args = parse_args()
    run_audit(Path(args.input), Path(args.output_dir), top_n=args.top_n)


if __name__ == "__main__":
    main()
