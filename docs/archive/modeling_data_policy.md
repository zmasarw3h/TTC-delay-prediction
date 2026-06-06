# Modeling Data Policy

This project keeps the cleaned combined TTC delay dataset as an audit dataset. The cleaned file preserves source-level fields that are useful for inspection, traceability, and later policy decisions, including fields that may be unsuitable for the main model.

Modeling will use a derived modeling-ready dataset rather than modifying the cleaned audit dataset destructively. Any row filtering, target thresholding, feature exclusion, or modeling-specific transformation should be implemented as a reproducible step after cleaning.

## Target Quality

The current cleaned dataset contains extreme `Min Delay` values. These records must be diagnosed and handled before model training because very large delay durations can dominate regression loss, distort target summaries, and make final evaluation less representative of the incident-time prediction task.

Outlier handling is a modeling-data choice, not a claim that every excluded record is invalid. Some extreme records may reflect real events, recording conventions, unresolved incidents, or data-entry issues. They should remain available in the cleaned audit dataset for inspection.

## Provisional Target Policy

For the main modeling dataset:

- Keep records with `Min Delay >= 0`.
- Exclude records with `Min Delay > 240` unless target diagnostics support a different threshold.
- Retain excluded outlier records in the cleaned audit dataset.

The 240-minute threshold is provisional. It should be revisited after reviewing `reports/target_diagnostics/target_summary.json`, `reports/target_diagnostics/target_threshold_counts.csv`, and `reports/target_diagnostics/top_delay_outliers.csv`.

Final README and resume metrics must be based only on a clean, reproducible scripted run using the documented modeling dataset and documented target threshold. Metrics from the cleaned audit dataset or from undocumented threshold choices should not be reported.
