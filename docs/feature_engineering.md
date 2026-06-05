# Feature Engineering

Phase 5B creates a derived modeling-ready dataset from the cleaned audit dataset. It does not overwrite `data/processed/ttc_delays_cleaned.csv` and does not train models.

## Modeling Dataset Filter

The default modeling dataset keeps:

- rows with valid `ts`
- rows with `Min Delay >= 0`
- rows with `Min Delay <= 240`

The `Min Delay <= 240` threshold follows the Phase 5A target diagnostics policy. It removes the extreme right tail for the main modeling dataset while preserving those records in the cleaned audit dataset. This threshold is a modeling choice, not a claim that excluded records are invalid.

`Min Gap` is retained in the saved modeling dataset for audit/reference only. It is leakage-sensitive and is not included in the main feature list.

## Feature List

Basic time features:

- `hour`
- `day_of_week`
- `month`
- `is_weekend`
- `is_holiday`
- `hour_sin`
- `hour_cos`
- `day_of_year`
- `day_sin`
- `day_cos`

Categorical features:

- `mode`
- `Route`
- `Direction`
- `Incident`
- `Location`

These categorical modeling columns are normalized before feature creation. Raw values for `Route`, `Direction`, `Incident`, and `Location` are preserved with `_raw` suffixes when available. The normalization policy is documented in `docs/categorical_normalization.md`.

Historical features:

- `prior_route_mean_delay`
- `prior_route_hour_mean_delay`
- `prior_incident_mean_delay`
- `prior_mode_mean_delay`
- `prior_global_mean_delay`
- `prior_route_hour_7d_mean_delay`

Targets:

- `Min Delay`
- `severe_delay_15`, an optional secondary target equal to `Min Delay >= 15`

## Historical Features

Historical features are target-derived and must be prior-only. The feature-building script uses only rows with `ts < current ts`; rows with the same timestamp as the current row are not allowed to influence each other.

Definitions:

- `prior_route_mean_delay`: mean `Min Delay` from rows with `ts < current ts` and the same normalized `Route`
- `prior_route_hour_mean_delay`: mean `Min Delay` from rows with `ts < current ts` and the same normalized `Route` and `hour`
- `prior_incident_mean_delay`: mean `Min Delay` from rows with `ts < current ts` and the same normalized `Incident`
- `prior_mode_mean_delay`: mean `Min Delay` from rows with `ts < current ts` and the same normalized `mode`
- `prior_global_mean_delay`: mean `Min Delay` from all rows with `ts < current ts`
- `prior_route_hour_7d_mean_delay`: mean `Min Delay` from prior incidents with the same `Route` and `hour` whose timestamps are within the previous 7 calendar days and strictly before the current row

Fallbacks for `prior_route_hour_7d_mean_delay` are also prior-only: prior route mean, then prior mode mean, then prior global mean.

## Excluded and Leakage-Sensitive Features

Excluded from the main feature list:

- `Min Delay`
- `severe_delay_15`
- `Min Gap`
- `Date`
- `Vehicle`
- `source_file`
- `source_sheet`
- `ts`

Leakage-sensitive columns:

- `Min Delay`
- `Min Gap`

`Min Delay` is the regression target. `Min Gap` may reflect information not appropriate for the incident-time prediction model and must not be used in the main model unless a later documented design explicitly justifies it.

Weather enrichment is not included yet. It should only be added later through a reproducible process that uses information available at incident report time.

## Chronological Splits

Default split policy:

- Train: rows with `ts <= 2022-12-31`
- Validation: rows where `year(ts) == 2023`
- Test: rows where `year(ts) == 2024`

Do not use random train/test splits for final evaluation.
