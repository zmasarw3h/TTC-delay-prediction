# API Input Contract

## Scope

The FastAPI service validates incident-time payloads, derives time-based model features from a caller-provided `timestamp`, and computes historical prior-delay features from local prior incident records. It does not implement raw TTC feed ingestion, weather enrichment, deployment, or live data updates.

The calibrated two-output model expects the approved feature columns from `feature_metadata.json`. For normal local-demo predictions, callers provide:

- `mode`
- `Route`
- `Direction`
- `Incident`
- `Location`
- `timestamp`

When `timestamp` is provided, the API derives the model's time fields and Ontario/Toronto holiday flag automatically. If historical prior-delay features are missing, the API computes them from `data/processed/modeling/modeling_dataset.csv` or `TTC_HISTORICAL_FEATURE_DATA_PATH`, using only rows where `ts < timestamp`.

Advanced callers may provide historical feature values manually for debugging. Supplied historical values are treated as overrides and produce warnings. If `timestamp` is missing, the request is rejected unless all required time and historical model features are supplied manually.

## Categorical Inputs

Categorical model fields are normalized before scoring:

- Leading and trailing whitespace is stripped.
- Empty strings, `nan`, `None`, `null`, `n/a`, and similar null-like strings are treated as missing.
- Missing categorical fields become `Unknown`.
- Route values are normalized through the same deterministic rule used in training. Values such as `29`, `29.0`, `501`, `32A`, `504B`, and `RAD` remain valid string categories. Obvious non-route text becomes `Unknown`.
- Direction values are normalized to `N`, `E`, `S`, `W`, `B`, or `Unknown`. Variants such as `N/B`, `north`, and `EB` use the normalized code; unsupported direction text becomes `Unknown` with a warning.
- Incident values are mapped to the curated operational categories documented in `docs/categorical_normalization.md`. Unrecognized non-missing labels become `Other`.
- Location values are normalized with safe uppercase text cleanup, separator normalization, and common abbreviation expansion. API location matching may compare normalized text to known normalized locations, but prediction input normalization does not fuzzy-snap or geocode locations.
- Mode is validated case-insensitively and must be `bus` or `streetcar` for API prediction requests.

The model preprocessing pipeline should use encoder unknown handling, so unseen categorical values can pass through validation. Those cases may still produce warnings because they are outside the categories observed or expected by the caller.

## Numeric Inputs

Missing numeric fields remain `None` after validation. This is intentional so the trained model pipeline can apply its numeric imputation strategy.

When `timestamp` is provided, missing time-derived fields are derived automatically:

- `hour`
- `day_of_week`
- `month`
- `is_weekend`
- `day_of_year`
- `hour_sin`
- `hour_cos`
- `day_sin`
- `day_cos`
- `is_holiday`

`is_holiday` is derived from Canadian/Ontario-relevant holidays. If a caller provides `is_holiday`, the API keeps the caller-provided value and returns a warning that it was not overwritten. If no `timestamp` is provided and `is_holiday` is missing, the API sets `is_holiday` to `0` and returns a warning.

Historical numeric features, such as prior route or prior route-hour delay means, are computed from records strictly before the prediction moment. Same-timestamp rows and future rows are excluded, the current incident target is never used, and `Min Gap` is never used.

If no prior support exists for v2 means or severe-rate features, the API leaves those values as `None` so the model pipeline can impute them. Historical count features use `0` when there is no support. The v1 `prior_route_hour_7d_mean_delay` feature can fall back to route, mode, and global prior means.

## Rejected Fields

The validator rejects leakage-sensitive or non-feature fields if they appear in an API payload:

- `Min Delay`
- `Min Gap`
- `Vehicle`
- `source_file`
- `source_sheet`
- `severe_delay_15`

These fields are excluded because they are targets, leakage-sensitive fields, audit metadata, or otherwise outside the approved prediction-time model input contract.
