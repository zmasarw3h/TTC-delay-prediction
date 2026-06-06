# Historical Feature Lookup

## Purpose

Phase 11C adds a local API inference layer that computes model-required historical features from prior incidents. This lets the local demo accept basic incident details instead of asking users to manually enter prior-delay features.

The lookup is for incident-time prediction only. For every request, it filters historical data to:

```text
historical_row.ts < prediction_timestamp
```

Same-timestamp rows and future rows are excluded. The lookup never uses the current incident target and never uses `Min Gap`.

## Data Source

Default local CSV:

```text
data/processed/modeling/modeling_dataset.csv
```

Override path:

```bash
TTC_HISTORICAL_FEATURE_DATA_PATH=/path/to/modeling_dataset.csv
```

The CSV is loaded lazily and cached by the API process. It must contain `ts`, `Min Delay`, `mode`, `Route`, `Direction`, `Incident`, and `Location`.

## Computed Features

The lookup computes all current model historical features:

```text
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

Definitions:

- Expanding means and counts use all prior rows before `prediction_timestamp`.
- The 7-day route-hour feature uses same normalized `Route` and prediction hour where `prediction_timestamp - 7 days <= ts < prediction_timestamp`.
- 30-day features use `prediction_timestamp - 30 days <= ts < prediction_timestamp`.
- Severe-rate features use prior indicators for `Min Delay >= 30` and `Min Delay >= 60`.
- Location features use the same normalized `Location` and prior rows only.

## Fallbacks

The v1 `prior_route_hour_7d_mean_delay` feature preserves the established fallback chain:

```text
7-day route-hour mean
prior_route_mean_delay
prior_mode_mean_delay
prior_global_mean_delay
```

For v2 means and rates, no-support values remain `None` so the trained model pipeline can impute them. Count features return `0` when there is no prior support.

The API response includes warnings when historical support is missing or low, when manual overrides are used, and when the prediction timestamp is outside or near the edge of local historical coverage.

## API Endpoints

`POST /predict-delay` accepts basic incident details:

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

If historical fields are omitted, the API computes them automatically. If a caller supplies any historical field, that value is treated as a manual override and the response includes a warning.

`GET /historical-lookup-info` returns the configured data path, load status, row count, min/max historical timestamps, historical feature names, and notes.

`POST /compute-historical-features` accepts the same basic incident fields plus timestamp and returns computed feature values, normalized inputs, support counts, metadata, and warnings. This endpoint is intended for debugging and frontend transparency.

## Limitations

- This is a local CSV lookup, not a production feature store.
- Historical features are only as current as the local `modeling_dataset.csv`.
- It is not suitable for live real-time TTC prediction without live data updates.
- Location matching remains approximate assistance based on normalized text.
- Request-time DataFrame filtering is acceptable for the local demo; future production work would need indexed or precomputed feature storage.
