# API Input Contract

## Scope

The FastAPI service validates engineered model feature payloads and can derive time-based model features from a caller-provided `timestamp`. It does not implement live historical feature lookup, raw incident-to-feature transformation, weather enrichment, or a frontend.

The calibrated two-output model expects the approved feature columns from `feature_metadata.json`. Callers usually need to provide:

- `mode`
- `Route`
- `Direction`
- `Incident`
- `Location`
- `timestamp`
- historical prior-delay features

When `timestamp` is provided, the API derives the model's time fields and Ontario/Toronto holiday flag automatically. Historical prior-delay features still need to be supplied by the caller or left missing for model-pipeline imputation.

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

Historical numeric features, such as prior route or prior route-hour delay means, must be computed from records strictly before the prediction moment. If those values are unavailable and left missing, the model may still score the row after imputation, but the prediction may be less reliable because less historical context was available.

## Rejected Fields

The validator rejects leakage-sensitive or non-feature fields if they appear in an API payload:

- `Min Delay`
- `Min Gap`
- `Vehicle`
- `source_file`
- `source_sheet`
- `severe_delay_15`

These fields are excluded because they are targets, leakage-sensitive fields, audit metadata, or otherwise outside the approved prediction-time model input contract.
