# Model Card

## Intended Use

This model is for local/demo incident-time TTC bus and streetcar delay prediction. Given basic incident details and a timestamp, it estimates expected delay minutes and calibrated severe-delay probabilities for `30+` and `60+` minute thresholds.

## Non-Intended Use

The model is not a live dispatch tool, production deployment, real-time TTC feed, or operational decision system. It should not be used for service-control decisions without validation on live data, monitoring, and a production data pipeline.

## Data Source and Coverage

The project uses local TTC bus and streetcar delay records from 2014 through 2024. Raw files are not committed. The local API can optionally use a TTC GTFS zip for frontend route-stop validation, but GTFS is not a model target source.

## Target

Primary regression target: `Min Delay`.

Target filtering policy for the main modeling dataset:

```text
0 <= Min Delay <= 240
```

This filters extreme values for the main model while preserving raw cleaned data separately for auditability.

## Split Policy

Chronological split:

| Split | Years |
|---|---|
| Train | 2014-2022 |
| Validation | 2023 |
| Test | 2024 |

Validation data is used for experiment selection, probability-calibration method selection, and operating cutoff selection. Test data is evaluated after those choices are fixed.

## Leakage Controls

- Historical features use prior records only.
- Feature-building logic uses shifted historical targets before rolling or expanding aggregation.
- API historical lookup filters records with `ts < prediction timestamp`.
- Same-timestamp rows are excluded from each other at inference lookup time.
- `Min Delay` is the target and is never used as an input feature.
- `Min Gap` is excluded from the main model.
- Same-day aggregates that include the current or future incidents are not used.
- Target encodings fit on the full dataset are not used.

## Categorical Normalization

Deterministic normalization is applied before feature engineering and API validation for:

- `mode`
- `Route`
- `Direction`
- `Incident`
- `Location`

Normalization does not use target values, geocoding, fuzzy snapping, or model outputs.

## Feature Summary

- Time features: hour, day of week, month, weekend flag, holiday flag, cyclic hour/day encodings.
- Categorical features: mode, route, direction, incident, location.
- v1 historical features: prior route, route-hour, incident, mode, global, and route-hour seven-day means.
- v2 historical features: route-incident, mode-incident, route-direction, 30-day route/incident means, location history, and support counts.
- Severe-rate rolling features: 30-day prior severe-delay rates for `30+` and `60+` minute thresholds.

## Model Choices

- Expected-delay regression: validation-selected fixed XGBoost strategy with log-transformed target.
- Severe-delay risk: classifiers for `Min Delay >= 30` and `Min Delay >= 60`.
- Probability calibration: validation-selected isotonic calibration for both thresholds in the latest local run.
- API serving: local FastAPI service loads a calibrated two-output artifact and computes missing historical features from local prior records.

## Final Metrics

Chronological 2024 holdout metrics:

| Output | Metric | Value |
|---|---:|---:|
| Route-history baseline | MAE | 8.98 min |
| Final expected-delay regressor | MAE | 7.76 min |
| Improvement vs. baseline | MAE reduction | about 13.6% |
| `30+` severe-delay risk | ROC-AUC | 0.905 |
| `30+` severe-delay risk | PR-AUC | 0.563 |
| `30+` severe-delay risk | Recall | 0.761 |
| `60+` severe-delay risk | ROC-AUC | 0.952 |
| `60+` severe-delay risk | PR-AUC | 0.437 |
| `60+` severe-delay risk | Recall | 0.822 |

![Model performance comparison](images/model_performance_comparison.png)

![Severe-delay metrics](images/severe_delay_metrics.png)

## Calibration Summary

The latest calibration workflow compares uncalibrated, sigmoid, and isotonic probabilities. Calibration methods and operating cutoffs are selected using validation data only. The latest local run selected isotonic calibration for both `30+` and `60+` severe-delay probabilities.

Risk bands:

| Band | Probability |
|---|---|
| Low | `< 0.10` |
| Medium | `0.10` to `< 0.30` |
| High | `>= 0.30` |

## Explainability Summary

Permutation-importance reports describe fitted-model behavior on sampled validation or test rows. In the latest local explainability run, prior historical delay features were among the strongest contributors to the expected-delay and severe-risk outputs. These reports are associative model diagnostics, not causal explanations.

## Limitations

- Local demo only; not production deployed.
- Historical lookup uses local `modeling_dataset.csv`, not live TTC feeds.
- Predictions are only as current as the local historical CSV and model artifact.
- No weather enrichment is included.
- Location matching is approximate text assistance and optional route-stop validation, not geocoding.
- Performance may change under live operating conditions or future service patterns.

## Reproducibility Notes

Raw data, generated processed data, reports, and model artifacts are gitignored. Reproducing metrics requires local TTC data files and the documented pipeline commands in the README or Makefile.
