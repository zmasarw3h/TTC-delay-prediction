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
- Empty strings, `nan`, `None`, and `null` are treated as missing.
- Missing categorical fields become `Unknown`.
- Route values are kept as string categories, so values such as `29`, `501`, and `RAD` remain valid categories.
- Direction values are normalized to uppercase. Common TTC direction codes such as `N`, `S`, `E`, `W`, and `B` are accepted. Unusual direction values are not rejected, but they return warnings.
- Mode is validated case-insensitively and must be `bus` or `streetcar`.

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
