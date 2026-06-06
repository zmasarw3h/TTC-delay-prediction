# Two-Output Delay And Risk Model

## Purpose

Phase 7A found that the selected regression model is best for overall expected-delay MAE, but severe delays remain difficult because a single point estimate can understate tail risk.

Phase 7B keeps the expected-delay regression output and adds separate severe-delay risk probabilities. The intended prediction shape is:

```json
{
  "predicted_delay_minutes": 7.9,
  "severe_delay_probability_30": 0.28,
  "severe_delay_probability_60": 0.07,
  "risk_band_30": "medium",
  "risk_band_60": "low"
}
```

This preserves the regression task while making high-impact delay risk explicit.

## Regression Model Choice

The expected-delay output uses the Phase 7A selected regressor from:

```text
artifacts/experiments/selected_experiment.joblib
```

If that artifact is unavailable, `src/models/train_risk_models.py` trains the Phase 7A default fallback, `combined_xgb_log_target`, on the training split only.

The regression output column is:

- `predicted_delay_minutes`

## Risk Classification Targets

The script trains binary classifiers for:

- `Min Delay >= 30`
- `Min Delay >= 60`

The probability outputs are:

- `severe_delay_probability_30`
- `severe_delay_probability_60`

Each classifier uses the same approved feature columns as the regression model. Leakage-sensitive and non-feature columns such as `Min Gap`, `Min Delay`, `ts`, and generated severe-delay target columns are excluded through the feature metadata policy.

## Threshold Selection Policy

Operating probability thresholds are selected using validation data only.

For each severe-delay target, the script evaluates these probability cutoffs:

```text
0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90
```

Selection rule:

1. Prefer cutoffs with validation recall >= 0.70.
2. Among those cutoffs, choose the highest validation F1.
3. If no cutoff reaches validation recall >= 0.70, choose the highest validation F1.

Test data is evaluated only after the operating thresholds are fixed.

## Risk Bands

Risk bands are assigned directly from predicted probabilities:

```text
low: probability < 0.20
medium: 0.20 <= probability < 0.50
high: probability >= 0.50
```

These bands are descriptive service outputs, not tuned thresholds.

## Outputs

Run:

```bash
python3 -m src.models.train_risk_models \
  --modeling-dir data/processed/modeling \
  --selected-regressor-path artifacts/experiments/selected_experiment.joblib \
  --reports-dir reports/risk_models \
  --artifacts-dir artifacts/risk_models \
  --thresholds 30,60
```

Reports:

```text
reports/risk_models/regression_metrics.csv
reports/risk_models/classification_metrics.csv
reports/risk_models/classification_threshold_table.csv
reports/risk_models/selected_classification_thresholds.json
reports/risk_models/risk_band_summary.csv
reports/risk_models/two_output_predictions_validation.csv
reports/risk_models/two_output_predictions_test.csv
reports/risk_models/two_output_summary.json
```

Artifact:

```text
artifacts/risk_models/two_output_model.joblib
```

The artifact includes the expected-delay regressor, risk classifiers, approved feature columns, target column, selected probability thresholds, risk-band definitions, and metadata for later prediction-service integration.

## Limitations

- This phase does not implement Optuna.
- This phase does not implement SHAP.
- This phase does not implement FastAPI or frontend code.
- Classification thresholds are selected on validation data, so they should not be changed after reviewing test results.
- Risk probabilities should be checked for calibration before being presented as precise operational probabilities.
