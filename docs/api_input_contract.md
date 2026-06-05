# API Input Contract

## Scope

The current API utilities validate engineered model feature payloads for the future FastAPI service. They do not implement a FastAPI app, endpoint routing, request schemas, live feature lookup, or raw incident-to-feature transformation.

The calibrated two-output model expects the approved feature columns from `feature_metadata.json`. A caller must supply those engineered features directly, including time features and prior-only historical features. Raw incident fields such as a report timestamp, route, incident type, and location are not yet enough to produce a prediction unless a separate feature lookup layer has already converted them into the model feature contract.

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
