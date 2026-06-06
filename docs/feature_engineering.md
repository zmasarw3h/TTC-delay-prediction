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
- `prior_route_incident_mean_delay`
- `prior_mode_incident_mean_delay`
- `prior_route_direction_mean_delay`
- `prior_route_incident_count`
- `prior_route_30d_mean_delay`
- `prior_incident_30d_mean_delay`
- `prior_route_30d_severe_rate_30`
- `prior_incident_30d_severe_rate_30`
- `prior_route_30d_severe_rate_60`
- `prior_incident_30d_severe_rate_60`
- `prior_location_mean_delay`
- `prior_location_count`

Targets:

- `Min Delay`
- `severe_delay_15`, an optional secondary target equal to `Min Delay >= 15`

## Historical Features

Historical features are target-derived and must be prior-only. The feature-building script uses only rows with `ts < current ts`; rows with the same timestamp as the current row are not allowed to influence each other.

For same-timestamp safety, expanding and rolling features aggregate history by timestamp within each grouping before cumulative or rolling calculations are shifted/excluded. This prevents records sharing the exact same `ts` from using each other's `Min Delay`.

## Phase 5 Historical Features

Definitions:

- `prior_route_mean_delay`: mean `Min Delay` from rows with `ts < current ts` and the same normalized `Route`
- `prior_route_hour_mean_delay`: mean `Min Delay` from rows with `ts < current ts` and the same normalized `Route` and `hour`
- `prior_incident_mean_delay`: mean `Min Delay` from rows with `ts < current ts` and the same normalized `Incident`
- `prior_mode_mean_delay`: mean `Min Delay` from rows with `ts < current ts` and the same normalized `mode`
- `prior_global_mean_delay`: mean `Min Delay` from all rows with `ts < current ts`
- `prior_route_hour_7d_mean_delay`: mean `Min Delay` from prior incidents with the same `Route` and `hour` whose timestamps are within the previous 7 calendar days and strictly before the current row

Fallbacks for `prior_route_hour_7d_mean_delay` are also prior-only: prior route mean, then prior mode mean, then prior global mean.

## Phase 11B Historical Features V2

Phase 11A model-improvement EDA found strong validation/test support for more specific historical groupings and recent-history windows. Phase 11B therefore adds these transparent prior-only features while preserving the existing v1 feature set.

Expanding prior features:

- `prior_route_incident_mean_delay`: mean `Min Delay` from rows with `ts < current ts` and the same normalized `Route` and `Incident`
- `prior_mode_incident_mean_delay`: mean `Min Delay` from rows with `ts < current ts` and the same normalized `mode` and `Incident`
- `prior_route_direction_mean_delay`: mean `Min Delay` from rows with `ts < current ts` and the same normalized `Route` and `Direction`
- `prior_route_incident_count`: count of rows with `ts < current ts` and the same normalized `Route` and `Incident`

Rolling 30-calendar-day features use this exact window:

```text
current ts - 30 days <= historical ts < current ts
```

Rolling mean features:

- `prior_route_30d_mean_delay`: mean `Min Delay` in the prior 30 days for the same normalized `Route`
- `prior_incident_30d_mean_delay`: mean `Min Delay` in the prior 30 days for the same normalized `Incident`

Rolling severe-rate features:

- `prior_route_30d_severe_rate_30`: mean of prior indicator `Min Delay >= 30` in the prior 30 days for the same normalized `Route`
- `prior_incident_30d_severe_rate_30`: mean of prior indicator `Min Delay >= 30` in the prior 30 days for the same normalized `Incident`
- `prior_route_30d_severe_rate_60`: mean of prior indicator `Min Delay >= 60` in the prior 30 days for the same normalized `Route`
- `prior_incident_30d_severe_rate_60`: mean of prior indicator `Min Delay >= 60` in the prior 30 days for the same normalized `Incident`

Location safeguards:

- `prior_location_mean_delay`: mean `Min Delay` from rows with `ts < current ts` and the same normalized `Location`
- `prior_location_count`: count of rows with `ts < current ts` and the same normalized `Location`

Location is high-cardinality and support-sensitive. `prior_location_mean_delay` is intentionally left missing when there is no prior location history instead of being filled with a target-derived fallback that would hide missingness. `prior_location_count` gives downstream models a support/confidence signal, and normal model-pipeline imputation can handle missing values.

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
