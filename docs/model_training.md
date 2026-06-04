# Model Training

## Purpose

Phase 6B trains the first reproducible machine learning model for incident-time TTC delay prediction.

This phase uses the leakage-safe train, validation, and test splits created by `src/features/build_features.py`. Model selection is based on validation metrics only. Test metrics are generated once after the fixed model configuration is chosen.

This is not an Optuna-tuned model. SHAP, FastAPI, and frontend work are not part of this phase.

## Inputs

Default input directory:

```text
data/processed/modeling/
```

Required files:

- `train.csv`
- `validation.csv`
- `test.csv`
- `feature_metadata.json`

The training script reads `feature_columns`, `target_column`, `categorical_columns`, `numeric_columns`, and `excluded_columns` from `feature_metadata.json`.

## Features

The script uses the metadata-approved `feature_columns` after excluding known target, leakage-sensitive, timestamp, source, and audit columns:

- `Min Gap`
- `Min Delay`
- `severe_delay_15`
- `ts`
- `Date`
- `Vehicle`
- `source_file`
- `source_sheet`

Categorical features are imputed with `"Unknown"` and one-hot encoded with `handle_unknown="ignore"`.

Numeric features are median-imputed and passed through to the model.

## Model

The first model is a fixed-configuration `XGBRegressor`:

- `objective="reg:squarederror"`
- `n_estimators=400`
- `learning_rate=0.05`
- `max_depth=6`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `random_state=42`
- `tree_method="hist"`
- `n_jobs=-1`

These settings are intentionally conservative and reproducible. They are not the result of hyperparameter tuning.

## Metrics

For validation and test splits, the script reports:

- MAE
- RMSE
- R2
- rows evaluated

It also reports metrics by `mode` for validation and test rows.

If the Phase 6A baseline report exists at `reports/baselines/baseline_metrics.json`, the script compares model MAE against the selected best baseline for each split. This comparison is included for local project tracking only and should not be turned into README or resume claims unless regenerated from a clean scripted run.

## Command

Run from the repository root after feature building and baseline evaluation:

```bash
python3 -m src.models.train_model \
  --modeling-dir data/processed/modeling \
  --reports-dir reports/models \
  --artifacts-dir artifacts/models \
  --baseline-report reports/baselines/baseline_metrics.json
```

Expected local outputs:

- `reports/models/model_metrics.json`
- `reports/models/model_metrics.csv`
- `reports/models/model_metrics_by_mode.csv`
- `artifacts/models/xgb_delay_model.joblib`

Generated reports and model artifacts are ignored by git.
