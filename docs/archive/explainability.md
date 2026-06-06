# Model Explainability

Phase 8 adds reproducible explanation reports for the calibrated two-output model. The goal is to describe how the existing artifact behaves on validation or test rows without retraining it, changing model parameters, or making new performance claims.

The explained outputs are:

- expected delay regression
- calibrated severe-delay probability for `Min Delay >= 30`
- calibrated severe-delay probability for `Min Delay >= 60`

## Default Method

Permutation importance is the default because it works directly against the fitted sklearn-compatible pipelines and calibrated classifier wrappers. It measures how much a model score changes when one approved input column is shuffled across the sampled split.

The script computes permutation importance on original approved feature columns before preprocessing. Leakage-sensitive and non-feature columns such as `Min Delay`, `Min Gap`, `severe_delay_15`, `ts`, source fields, and metadata exclusions are not used as model inputs.

Regression uses negative mean absolute error. Risk classifiers use average precision when available, then fall back to ROC-AUC, then negative log loss.

## Optional SHAP

`--include-shap` requests an optional SHAP smoke test. SHAP is not required for the Phase 8 reports. If SHAP is missing or cannot explain the wrapped pipelines/calibrated models in the local environment, the script still writes the permutation-importance reports and records the SHAP status in `explainability_summary.json`.

## How To Run

Default validation explanations:

```bash
python3 -m src.models.explain_models
```

Use test rows only after validation-oriented interpretation is complete:

```bash
python3 -m src.models.explain_models \
  --split test \
  --max-rows 5000
```

Optional SHAP request:

```bash
python3 -m src.models.explain_models --include-shap
```

## Output Files

Generated files are written under `reports/explainability/`:

```text
permutation_importance_regression.csv
permutation_importance_risk_30.csv
permutation_importance_risk_60.csv
global_feature_importance.csv
representative_prediction_examples.csv
explainability_summary.json
figures/top_features_regression.png
figures/top_features_risk_30.png
figures/top_features_risk_60.png
```

Generated report files and figures are local artifacts and are gitignored.

## Interpreting Global Importance

Each permutation-importance row contains the model output, original feature column, mean importance, standard deviation, rank, scoring method, split, and row count. Larger positive values mean the model score worsened more when that feature was shuffled, so the fitted model relied more on that feature for the sampled rows.

Categorical variables are evaluated at the original column level before one-hot encoding. For example, shuffling `Incident` measures the impact of disturbing the whole incident category input, not a single encoded category.

## Interpreting Local Examples

`representative_prediction_examples.csv` contains representative prediction examples, not causal explanations. It selects low predicted delay, high predicted delay, high threshold-30 risk, high threshold-60 risk, and large regression-error rows where available.

Each row includes the incident context, actual delay, expected-delay prediction, calibrated probabilities, risk bands, absolute error, and a JSON field with values for important global features. Without SHAP output, these records should be read as local prediction context only.

## Limitations

- Feature importance is associative, not causal.
- One-hot encoded categorical variables are grouped only at the original feature level.
- Importance can vary by split, sample size, random seed, and correlated features.
- Calibrated risk classifier explanations describe risk ranking and probability outputs, not guaranteed causal mechanisms.
- Representative examples are selected from sampled validation or test rows and may not cover every operational scenario.
