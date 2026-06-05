# FastAPI Prediction Service

## Scope

Phase 9 adds a local FastAPI service for the existing calibrated two-output model artifact. Phase 10 adds a thin static frontend served by the same FastAPI app. It does not train models, run Optuna, compute SHAP values, add weather enrichment, or implement deployment.

The default model artifact path is:

```text
artifacts/calibration/calibrated_two_output_model.joblib
```

Set `TTC_MODEL_ARTIFACT_PATH` to load a different local artifact:

```bash
TTC_MODEL_ARTIFACT_PATH=/path/to/calibrated_two_output_model.joblib \
uvicorn src.api.app:app --reload
```

## Local Run

From the repository root:

```bash
uvicorn src.api.app:app --reload
```

Open the local demo UI at:

```text
http://127.0.0.1:8000/
```

The app is import-safe and loads the artifact lazily on the first endpoint that needs model metadata or predictions.

## Local Demo UI

The FastAPI app serves a lightweight static frontend from `src/api/static/`.

Frontend URL:

```text
http://127.0.0.1:8000/
```

Static assets:

```text
/static/styles.css
/static/app.js
```

The demo fetches `/health` and `/model-info` on page load, then submits engineered incident-time payloads to `/predict-delay`. It displays expected delay minutes, calibrated `30+` and `60+` minute probabilities, risk bands, binary severe-delay flags, and API warnings.

The UI includes exactly two presets:

- Bus incident: a bus example with reasonable prior-delay feature values and a timestamp so calendar fields are derived by the API.
- Streetcar incident: a streetcar example with reasonable prior-delay feature values and a timestamp so calendar fields are derived by the API.

This is a local demo UI only. It still expects engineered incident-time features; raw TTC incident-to-feature lookup and weather enrichment are not implemented.

## Endpoints

### `GET /`

Serves the local static demo frontend.

### `GET /static/styles.css` and `GET /static/app.js`

Serve the frontend CSS and JavaScript assets.

### `GET /health`

Returns service status, whether the model artifact is currently loaded in memory, whether the configured artifact file exists, and the configured artifact path.

### `GET /model-info`

Returns model metadata:

- model name
- model phase
- feature columns
- target column
- risk thresholds
- selected calibration methods
- selected operating cutoffs
- risk band definitions
- notes and limitations

### `POST /predict-delay`

Returns one calibrated two-output prediction for an engineered incident-time feature payload:

- expected delay in minutes
- calibrated severe-delay probabilities for `30+` and `60+` minute thresholds
- risk bands
- selected binary severe-delay decisions using validation-selected probability cutoffs
- non-fatal validation warnings

## Engineered Feature Input Contract

The current model requires engineered model features in the exact artifact feature order. Required model features are:

```text
mode
Route
Direction
Incident
Location
hour
day_of_week
month
is_weekend
is_holiday
hour_sin
hour_cos
day_of_year
day_sin
day_cos
prior_route_mean_delay
prior_route_hour_mean_delay
prior_incident_mean_delay
prior_mode_mean_delay
prior_global_mean_delay
prior_route_hour_7d_mean_delay
```

Callers usually only need to provide the categorical incident fields, `timestamp`, and the historical prior-delay features. If `timestamp` is provided and time-derived fields are missing, the API derives:

```text
hour
day_of_week
month
is_weekend
is_holiday
day_of_year
hour_sin
hour_cos
day_sin
day_cos
```

`is_holiday` is derived from Canadian/Ontario-relevant holidays using the timestamp date. If `is_holiday` is provided by the caller, the API keeps that value and returns a warning that it was not overwritten. If no timestamp is provided and `is_holiday` is missing, it is set to `0` and the response includes a warning.

Historical features are required because the trained model uses prior-only delay context, such as route-level and route-hour delay means. These values must be computed from incidents strictly before the prediction moment. The API does not currently include a feature store or raw incident-to-feature lookup layer, so callers must provide these engineered historical values. Missing historical numeric features may be passed as `null` so the model pipeline can impute, but the response warns that prediction reliability may be reduced.

## Example Request

```bash
curl -X POST http://127.0.0.1:8000/predict-delay \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "bus",
    "Route": "29",
    "Direction": "N",
    "Incident": "Mechanical",
    "Location": "Dufferin Station",
    "timestamp": "2024-02-03T08:30:00",
    "prior_route_mean_delay": 10.0,
    "prior_route_hour_mean_delay": 12.0,
    "prior_incident_mean_delay": 9.0,
    "prior_mode_mean_delay": 8.0,
    "prior_global_mean_delay": 7.0,
    "prior_route_hour_7d_mean_delay": 11.0
  }'
```

## Example Response

```json
{
  "predicted_delay_minutes": 12.5,
  "calibrated_severe_delay_probability_30": 0.35,
  "risk_band_30": "high",
  "severe_delay_prediction_30": 1,
  "selected_probability_cutoff_30": 0.2,
  "calibrated_severe_delay_probability_60": 0.08,
  "risk_band_60": "low",
  "severe_delay_prediction_60": 0,
  "selected_probability_cutoff_60": 0.3,
  "warnings": [
    "Derived missing time fields from timestamp.",
    "Derived is_holiday from timestamp."
  ],
  "model_name": "calibrated_two_output_delay_and_risk_model",
  "model_phase": "Phase 7C"
}
```

## Validation

The API rejects leakage-sensitive fields:

```text
Min Delay
Min Gap
Vehicle
source_file
source_sheet
severe_delay_15
```

`mode` must be `bus` or `streetcar`. Unknown route, incident, location, or unusual direction values are normalized and may return warnings instead of crashing.

## Limitations

- Raw incident-to-feature lookup is not implemented yet.
- Weather enrichment is not included yet.
- Predictions depend on the quality of provided historical prior-delay features.
- The API is local/demo-ready, not production-deployed.
