# FastAPI Prediction Service

## Scope

Phase 9 adds a local FastAPI service for the existing calibrated two-output model artifact. Phase 10 adds a thin static frontend served by the same FastAPI app, Phase 10B adds planner-oriented controls plus model-category and location-matching assistance, and Phase 10C repairs the demo around the normalized categorical contract. It does not train models, run Optuna, compute SHAP values, add weather enrichment, or implement deployment.

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

The demo loads `/health`, `/model-info`, `/model-options`, and `/route-options` on page load, then submits engineered incident-time payloads to `/predict-delay`. It displays planner-friendly cards for expected delay minutes, calibrated `30+` and `60+` minute probabilities, risk bands, binary severe-delay flags, and input notes. Selected probability cutoffs are kept under a collapsed model-details section.

Mode is selected with Bus and Streetcar buttons. Incident controls use controlled normalized options rather than raw artifact categories. Route is a dependency-free searchable picker filtered by the selected mode. After route selection, the Direction dropdown is limited to GTFS-derived directions for that route, using the normalized `N`, `E`, `S`, `W`, and `B` model contract. The selected mode is submitted as the same `bus` / `streetcar` model feature as before.

Location is route-scoped in the main UI. The user must choose a location from stops served by the selected route, using local TTC GTFS route-stop data. The frontend calls `/validate-route-location` before prediction and blocks submission when the selected location is not a stop on the selected route. This prevents invalid combinations such as route `29` with a Yonge/Dundas location.

GTFS is used only for route-stop validity and route-derived mode. The model still receives normalized text in the existing `Location` field, not GTFS `stop_id`. After route-stop validation, the UI can call `/match-location` to compare the selected stop name against known model locations:

- exact normalized matches are accepted automatically
- fuzzy matches with score `>= 90` are accepted automatically
- fuzzy matches with score from `75` to `< 90` are shown as suggestions that the user can accept
- lower-confidence model-location matches are not accepted, so the route-validated normalized stop text is submitted instead

This is a local demo UI only. It still expects engineered incident-time features; raw TTC incident-to-feature lookup and weather enrichment are not implemented.

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
