"""Create small public-facing figures from existing local report outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs" / "images"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def create_model_performance_comparison() -> Path:
    baseline = _read_json(ROOT / "reports" / "baselines" / "baseline_metrics.json")
    fixed = _read_json(ROOT / "reports" / "models" / "model_metrics.json")
    calibrated = _read_json(
        ROOT / "reports" / "calibration" / "calibrated_two_output_summary.json"
    )

    baseline_mae = next(
        item["mae"]
        for item in baseline["metrics"]
        if item["split"] == "test"
        and item["evaluation_name"] == "prior_route_mean_delay_filled"
    )
    fixed_mae = fixed["test_metrics"]["mae"]
    final_mae = next(
        item["mae"]
        for item in calibrated["selection"]["regression_metrics"]
        if item["split"] == "test"
    )

    labels = ["Route history\nbaseline", "Fixed XGBoost", "Final log-target\nXGBoost"]
    values = [baseline_mae, fixed_mae, final_mae]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#6b7280", "#2563eb", "#059669"])
    ax.set_ylabel("2024 holdout MAE (minutes)")
    ax.set_title("Expected-delay model comparison")
    ax.set_ylim(0, max(values) + 1.5)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    path = OUTPUT_DIR / "model_performance_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def create_severe_delay_metrics() -> Path:
    summary = _read_json(ROOT / "reports" / "risk_models" / "two_output_summary.json")
    metrics = pd.DataFrame(summary["classification_metrics"])
    selected = metrics[
        (metrics["split"] == "test") & (metrics["threshold_minutes"].isin([30, 60]))
    ].copy()
    selected = selected.sort_values("threshold_minutes")

    labels = [f"{int(row.threshold_minutes)}+ min" for row in selected.itertuples()]
    metric_names = ["roc_auc", "pr_auc", "recall"]
    x_positions = range(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2563eb", "#059669", "#f59e0b"]
    for offset, metric, color in zip([-width, 0, width], metric_names, colors):
        values = selected[metric].tolist()
        positions = [idx + offset for idx in x_positions]
        bars = ax.bar(
            positions,
            values,
            width=width,
            label=metric.upper().replace("_", "-"),
            color=color,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("2024 holdout score")
    ax.set_title("Calibrated severe-delay risk metrics")
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / "severe_delay_metrics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    created = [
        create_model_performance_comparison(),
        create_severe_delay_metrics(),
    ]
    for path in created:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
