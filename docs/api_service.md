# FastAPI Prediction Service

## Scope

Phase 9 adds a local FastAPI service for the existing calibrated two-output model artifact. Phase 10 adds a thin static frontend served by the same FastAPI app, Phase 10B adds planner-oriented controls plus model-category and location-matching assistance, Phase 10C repairs the demo around the normalized categorical contract, and Phase 11C adds local historical feature lookup for inference. It does not train models, run Optuna, compute SHAP values, add weather enrichment, or implement deployment.

The default model artifact path is:

```text
artifacts/calibration/calibrated_two_output_model.joblib
```

Set `TTC_MODEL_ARTIFACT_PATH` to load a different local artifact:

```bash
TTC_MODEL_ARTIFACT_PATH=/path/to/calibrated_two_output_model.joblib \
uvicorn src.api.app:app --reload
```

The default local historical feature source is:

```text
data/processed/modeling/modeling_dataset.csv
```

Set `TTC_HISTORICAL_FEATURE_DATA_PATH` to use a different local modeling CSV for historical lookup.

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

The demo loads `/health`, `/model-info`, `/model-options`, `/route-options`, and `/historical-lookup-info` on page load, then submits basic incident-time payloads to `/predict-delay`. It displays planner-friendly cards for expected delay minutes, calibrated `30+` and `60+` minute probabilities, risk bands, binary severe-delay flags, and input notes. Selected probability cutoffs are kept under a collapsed model-details section.

Mode is selected with Bus and Streetcar buttons. Incident controls use controlled normalized options rather than raw artifact categories. Route is a dependency-free searchable picker filtered by the selected mode. After route selection, the Direction dropdown is limited to GTFS-derived directions for that route, using the normalized `N`, `E`, `S`, `W`, and `B` model contract. The selected mode is submitted as the same `bus` / `streetcar` model feature as before.

Location is route-scoped in the main UI. The user must choose a location from stops served by the selected route, using local TTC GTFS route-stop data. The frontend calls `/validate-route-location` before prediction and blocks submission when the selected location is not a stop on the selected route. This prevents invalid combinations such as route `29` with a Yonge/Dundas location.

GTFS is used only for route-stop validity and route-derived mode. The model still receives normalized text in the existing `Location` field, not GTFS `stop_id`. After route-stop validation, the UI can call `/match-location` to compare the selected stop name against known model locations:

- exact normalized matches are accepted automatically
- fuzzy matches with score `>= 90` are accepted automatically
- fuzzy matches with score from `75` to `< 90` are shown as suggestions that the user can accept
- lower-confidence model-location matches are not accepted, so the route-validated normalized stop text is submitted instead

This is a local demo UI only. It accepts basic incident details and timestamp, then computes model time features and historical features locally. Weather enrichment and live TTC data updates are not implemented.

Route-stop validation requires a local TTC GTFS zip file. Set `TTC_GTFS_ZIP_PATH` or place the zip at:

```text
data/raw/ttc_gtfs.zip
```

The TTC routes and schedules GTFS package is available from the City of Toronto Open Data portal: <https://open.toronto.ca/dataset/ttc-routes-and-schedules/>.

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

### `GET /model-options`

Returns category options for guided frontend controls:

- modes as `{value, label}` objects:
  - `bus`: Bus
  - `streetcar`: Streetcar
- routes as route-like strings only, such as `29`, `501`, `32A`, `504B`, or `RAD`
- directions as `{value, label}` objects:
  - `N`: North
  - `E`: East
  - `S`: South
  - `W`: West
  - `B`: Both / bidirectional
  - `Unknown`: Unknown
- incidents as curated normalized `{value, label}` objects
- locations as an empty list; known normalized locations are used server-side only for `/match-location`
- warnings for incomplete category extraction
- counts for each category list

Direction and incident options are not populated from raw artifact categories, which prevents polluted category values from leaking into the UI. Route options are extracted from the calibrated artifact's known categories where feasible, then filtered through the normalized route contract. If route extraction is incomplete, route suggestions are returned empty and callers can still type a route.

Curated incident values:

```text
Mechanical
Utilized Off Route
General Delay
Late Leaving Garage
Investigation
Operations - Operator
Operations
Diversion
Emergency Services
Security
Collision - TTC
Collision - TTC Involved
Road Blocked - NON-TTC Collision
Held By
Cleaning
Cleaning - Unsanitary
Vision
Overhead
Overhead - Pantograph
Rail/Switches
Other
Unknown
```

### `GET /route-options`

Returns route picker options. When GTFS route-stop data is configured, routes come from `routes.txt` and include derived mode:

```json
{
  "routes": [
    {"value": "29", "label": "29 - Dufferin", "mode": "bus"},
    {"value": "501", "label": "501 - Queen", "mode": "streetcar"}
  ],
  "gtfs_available": true,
  "source_path": "data/raw/ttc_gtfs.zip",
  "warning": null
}
```

When GTFS is not configured, the endpoint falls back to route-like model artifact categories for display, but route-stop validation remains unavailable and the frontend blocks prediction.

### `GET /route-locations`

Returns stop/location options scoped to a selected route:

```text
/route-locations?route=29
```

Response:

```json
{
  "route": "29",
  "normalized_route": "29",
  "mode": "bus",
  "directions": [
    {"value": "N", "label": "North"},
    {"value": "S", "label": "South"},
    {"value": "B", "label": "Both / bidirectional"}
  ],
  "locations": [
    {
      "value": "DUFFERIN STATION",
      "label": "Dufferin Station",
      "normalized_location": "DUFFERIN STATION"
    }
  ],
  "count": 1,
  "gtfs_available": true,
  "source_path": "data/raw/ttc_gtfs.zip",
  "warning": null
}
```

Branch routes such as `29A` may fall back to the base route stop list, for example route `29`, with a warning.

### `POST /validate-route-location`

Validates that a location belongs to the selected route before prediction.

Request:

```json
{
  "route": "29",
  "location": "Dufferin Station"
}
```

Accepted response:

```json
{
  "route": "29",
  "normalized_route": "29",
  "original_location": "Dufferin Station",
  "normalized_location": "DUFFERIN STATION",
  "route_location": "DUFFERIN STATION",
  "route_location_label": "Dufferin Station",
  "accepted_for_prediction": true,
  "warning": null
}
```

Rejected route/location combinations return `accepted_for_prediction: false` with a warning. They are not submitted by the frontend.

### `POST /match-location`

Matches route-validated normalized stop/location text to known model locations without changing `/predict-delay` behavior.

Request:

```json
{
  "location": "queen and spadina"
}
```

Response:

```json
{
  "original_location": "queen and spadina",
  "normalized_location": "QUEEN AND SPADINA",
  "matched_location": "Queen Street West and Spadina Avenue",
  "score": 85.0,
  "match_type": "fuzzy",
  "warning": "Possible location match found; review and accept it before prediction.",
  "accepted_for_prediction": false
}
```

The frontend submits the matched model location when high-confidence matching succeeds. Otherwise it submits the route-validated normalized stop text. The prediction endpoint itself does not force matching.

Invalid or malformed requests return standard FastAPI `422` JSON responses. Normal no-match cases return a successful JSON response with `match_type` set to `none`, `accepted_for_prediction` set to `false`, and a readable warning instead of raising a server error.

### `GET /historical-lookup-info`

Returns historical lookup status and coverage:

```json
{
  "historical_data_path": "data/processed/modeling/modeling_dataset.csv",
  "loaded": true,
  "loadable": true,
  "row_count": 12345,
  "min_timestamp": "2014-01-01T00:00:00",
  "max_timestamp": "2024-12-31T23:59:00",
  "available_historical_feature_names": ["prior_route_mean_delay"],
  "notes_limitations": ["Only rows with ts strictly before the prediction timestamp are used."],
  "warnings": []
}
```

### `POST /compute-historical-features`

Computes historical features without scoring the model. This is useful for debugging and frontend transparency.

Request:

```json
{
  "mode": "bus",
  "Route": "29",
  "Direction": "N",
  "Incident": "Mechanical",
  "Location": "Dufferin Station",
  "timestamp": "2024-02-03T08:30:00"
}
```

Response includes computed feature values, warnings, normalized input values, support counts, and metadata.

### `POST /predict-delay`

Returns one calibrated two-output prediction for a basic incident-time payload:

- expected delay in minutes
- calibrated severe-delay probabilities for `30+` and `60+` minute thresholds
- risk bands
- selected binary severe-delay decisions using validation-selected probability cutoffs
- non-fatal validation warnings

## Engineered Feature Input Contract

The current model still scores a DataFrame in the exact artifact feature order. The API builds that DataFrame from basic incident fields, timestamp-derived time features, and local historical lookup values.

Required model features are:

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
prior_route_incident_mean_delay
prior_mode_incident_mean_delay
prior_route_direction_mean_delay
prior_route_incident_count
prior_route_30d_mean_delay
prior_incident_30d_mean_delay
prior_route_30d_severe_rate_30
prior_incident_30d_severe_rate_30
prior_route_30d_severe_rate_60
prior_incident_30d_severe_rate_60
prior_location_mean_delay
prior_location_count
```

Callers usually only need to provide:

```text
mode
Route
Direction
Incident
Location
timestamp
```

If `timestamp` is provided and time-derived fields are missing, the API derives:

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

`is_holiday` is derived from Canadian/Ontario-relevant holidays using the timestamp date. If `is_holiday` is provided by the caller, the API keeps that value and returns a warning that it was not overwritten.

Historical features are computed from local prior incidents where `ts < timestamp`. Same-timestamp and future rows are excluded. If a caller supplies historical values manually, the API uses those override values and returns warnings naming the overridden features. If `timestamp` is missing, the request is rejected unless all required time and historical model features are provided manually.

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
    "timestamp": "2024-02-03T08:30:00"
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
    "Historical features were computed from prior local records with ts before the prediction timestamp.",
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

- Weather enrichment is not included yet.
- Historical lookup is a local CSV lookup, not a production feature store.
- Historical lookup is only as current as `modeling_dataset.csv`.
- The API is local/demo-ready, not production-deployed.
