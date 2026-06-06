# Model Error Analysis

Phase 6C analyzes residuals from the first fixed XGBoost delay model. It does not train a new model, tune hyperparameters, run SHAP, or modify the saved model artifact.

The goal is to understand where the fixed model is accurate or weak across validation and test data, with special attention to mode, route, incident type, time patterns, delay-size buckets, high-delay incidents, and the largest individual errors.

## Command

Run from the repository root after the Phase 6B model artifact and modeling splits exist:

```bash
python3 -m src.models.analyze_errors \
  --modeling-dir data/processed/modeling \
  --model-path artifacts/models/xgb_delay_model.joblib \
  --baseline-report reports/baselines/baseline_metrics.json \
  --output-dir reports/error_analysis
```

The default paths match the command above. Route and incident breakdowns include groups with at least 100 evaluated rows by default. To change that threshold:

```bash
python3 -m src.models.analyze_errors --min-group-size 50
```

## Inputs

- `artifacts/models/xgb_delay_model.joblib`: trained Phase 6B fixed XGBoost pipeline.
- `data/processed/modeling/train.csv`: used only for optional filled-baseline fallback means.
- `data/processed/modeling/validation.csv`: scored for error analysis.
- `data/processed/modeling/test.csv`: scored for error analysis.
- `data/processed/modeling/feature_metadata.json`: source of approved feature columns and target column.
- `reports/baselines/baseline_metrics.json`: optional selected-baseline configuration.

## Outputs

Generated files are written under `reports/error_analysis/`:

- `error_summary.json`
- `error_summary.csv`
- `error_by_mode.csv`
- `error_by_route.csv`
- `error_by_incident.csv`
- `error_by_hour.csv`
- `error_by_month.csv`
- `error_by_delay_bucket.csv`
- `high_delay_performance.csv`
- `worst_predictions_validation.csv`
- `worst_predictions_test.csv`

If matplotlib is installed, simple figures are also written under `reports/error_analysis/figures/`.

## What To Look For

- Overall residual summary: compare validation vs test MAE, RMSE, R2, mean error, median absolute error, and tail absolute-error percentiles.
- Mode breakdown: check whether bus and streetcar errors diverge materially.
- Route and incident breakdowns: inspect high-MAE groups with enough rows to avoid overreacting to tiny samples.
- Hour and month breakdowns: look for systematic time-of-day or seasonal weakness.
- Delay bucket breakdown: confirm whether large actual delays are disproportionately underpredicted.
- High-delay performance: review `Min Delay >= 15`, `>= 30`, and `>= 60` rows for underprediction rate and average underprediction amount.
- Worst predictions: inspect the largest absolute errors for data quality issues, unusual routes/incidents, or model blind spots.

Do not use this analysis to make final README or resume performance claims unless the results come from a clean, reproducible scripted run.
