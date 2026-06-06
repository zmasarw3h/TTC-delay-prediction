# Model Design

## Prediction Moment

The prediction moment is when a TTC bus or streetcar incident is reported.

The model should only use information that would reasonably be available at that time. It should not use information that is only known after the incident evolves or clears.

## Target Variable

Primary target:

- `Min Delay`

This represents expected delay duration in minutes.

## Main Modeling Task

The main task is regression:

- Input: incident-time features.
- Output: predicted delay duration in minutes.

The recommended primary model is XGBoost regression after a baseline is established.

## Severe-Delay Risk Task

The current two-output direction adds severe-delay risk classification alongside the expected-delay regression output.

Risk targets:

- `Min Delay >= 30`
- `Min Delay >= 60`

These classifiers support product-style risk outputs while keeping regression as the expected-delay estimate. Operating probability thresholds must be selected on validation data only, then applied once to test.

## Safe Feature Categories

Safe features are features that can be known or computed at incident report time.

Recommended categories:

- Mode: bus or streetcar.
- Route.
- Direction, if present and reliable at report time.
- Incident type.
- Location or location cluster, if available at incident report time.
- Hour of day.
- Day of week.
- Month.
- Weekend flag.
- Holiday flag.
- Cyclical time features, such as `hour_sin`, `hour_cos`, `day_sin`, and `day_cos`.
- Weather features available at or before the incident timestamp, such as temperature, precipitation, snow, and snowfall flag.
- Prior historical route delay features.
- Prior historical route-hour delay features.
- Prior historical incident-type delay features.

Historical delay features must be calculated using only incidents before the current row.

## Risky Or Excluded Features

Exclude or quarantine:

- `Min Delay` as an input feature.
- `Min Gap`, unless it is clearly confirmed to be known at incident report time.
- Rolling averages that include the current row.
- Same-day aggregate features that include later incidents from that day.
- Future aggregate features.
- Target-derived encodings fit on the full dataset.
- Preprocessing fit on the full dataset before train, validation, and test splitting.
- Any post-incident fields, resolution fields, or manually updated fields that would not exist at initial report time.

## Leakage Prevention Rules

Use these rules in the implementation phase:

- Use a chronological split, not a random split.
- Fit preprocessing on training data only.
- Fit target encoders on training folds only.
- Use `shift(1)` before rolling target-derived features.
- Compute validation and test historical features using only past data available before each row.
- Keep the test set untouched until final evaluation.
- Do not tune hyperparameters on the test set.
- Do not manually adjust the final model after viewing test results.
- Generate final README/resume metrics from a clean scripted run.

## Recommended Split

Recommended fixed chronological split:

- Train: 2014-2022.
- Validation: 2023.
- Test: 2024.

The current notebook output indicates both bus and streetcar data run through the end of 2024, so this split should be feasible.

If future data inspection shows that either mode has incomplete or unreliable 2024 data, use the latest complete year as the test set and the prior complete year as validation. Document the change explicitly.

## Recommended Baseline

The main baseline should be a historical route-hour average with fallback levels.

The preferred portfolio baseline is a true 7-day time-window average: prior incidents from the same route and hour during the previous 7 calendar days. This is easier to explain than the current notebook's row-count approach.

The current notebook attempts to create a historical route-hour delay feature using `shift(1).rolling(7*24)`, but because it is grouped by `Route` and `hour`, this means the previous 168 records within that route-hour group, not necessarily the previous 7 calendar days.

If a true time-window feature is difficult because of sparse route-hour groups, use a fixed prior-observation average instead, such as the prior N incidents for the same route-hour. Document the chosen definition clearly.

Recommended fallback order:

1. Prior 7-day route-hour time-window average, or documented prior-N route-hour average.
2. Prior route average.
3. Prior mode average.
4. Prior global average.

All fallback values must use only historical rows before the prediction timestamp.

This baseline is strong enough to be meaningful and simple enough to explain in the README.

## Recommended Model Features

Initial regression feature set:

- `mode`
- `Route`
- `Direction`
- `Incident`
- `Location` or a normalized location cluster
- `hour`
- `day_of_week`
- `month`
- `is_weekend`
- `is_holiday`
- `hour_sin`
- `hour_cos`
- `day_sin`
- `day_cos`
- `temp`
- `TOTAL_PRECIP`
- `TOTAL_SNOW`
- `snowfall_flag`
- prior route delay mean
- prior route-hour delay mean
- prior incident-type delay mean

Do not include `Min Gap` in the first production-style model unless the data dictionary or domain reasoning confirms it is available at report time.

## Recommended Metrics

Primary metric:

- MAE.

Secondary metrics:

- RMSE.
- R2.
- Improvement over baseline.

For the optional severe-delay classifier:

- Precision.
- Recall.
- F1.
- ROC AUC or PR AUC, depending on class balance.

MAE should be the headline metric because it is directly interpretable as average absolute error in minutes.

## XGBoost And Tuning Plan

Recommended implementation order:

1. Build and evaluate the baseline.
2. Train a default or lightly configured XGBoost regressor.
3. Compare against the baseline on validation data.
4. Add Optuna only if the baseline and first XGBoost result are stable.
5. Lock model choices before final test evaluation.

Optuna should optimize validation or time-series CV MAE, not test MAE. Use `suggest_float(..., log=True)` instead of deprecated Optuna APIs.

## Explainability Plan

Use SHAP after the final feature pipeline is stable.

Recommended outputs:

- Global feature importance summary.
- Beeswarm or bar plot for top drivers.
- Local explanation for example incidents.
- Short written interpretation of route, incident type, weather, and time effects.

Explainability should be framed as model behavior, not causal proof.

## API Design Preview

Planned FastAPI endpoints:

Weather caveat:

For historical evaluation, weather features are joined from recorded weather observations. For live API use, weather inputs should be supplied by the user, retrieved from a weather API, or approximated using the most recent available observation or forecast available at report time.

### `GET /health`

Purpose:

- Confirm that the API is running.

Example response:

```json
{
  "status": "ok"
}
```

### `GET /model-info`

Purpose:

- Return model metadata.

Example response:

```json
{
  "model_name": "calibrated_two_output_delay_and_risk_model",
  "target_column": "Min Delay",
  "risk_thresholds": [30, 60],
  "model_phase": "Phase 7C"
}
```

### `POST /predict-delay`

Purpose:

- Predict expected delay duration for one reported incident.

Example request:

```json
{
  "mode": "bus",
  "Route": "29",
  "Direction": "N",
  "Incident": "Mechanical",
  "Location": "Dufferin Station",
  "timestamp": "2024-02-15T08:30:00"
}
```

The current API derives time features and computes missing historical features from local prior records when `timestamp` is provided. Weather enrichment is not implemented.

Example response:

```json
{
  "predicted_delay_minutes": 12.4,
  "calibrated_severe_delay_probability_30": 0.18,
  "risk_band_30": "medium",
  "calibrated_severe_delay_probability_60": 0.04,
  "risk_band_60": "low",
  "model_phase": "Phase 7C"
}
```

## Implementation Status

The notebook has been converted into a reproducible project structure:

- `src/data/` for ingestion, cleaning, normalization, and category audits.
- `src/features/` for leakage-safe feature generation.
- `src/models/` for baseline, training, experiments, calibration, error analysis, and explainability.
- `src/api/` for the FastAPI local service, frontend, validation, and historical lookup.
- `reports/` or `artifacts/` for generated metrics, plots, and model files.

The notebook remains an exploratory reference, not the final implementation.
