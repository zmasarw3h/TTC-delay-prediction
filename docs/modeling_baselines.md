# Modeling Baselines

## Purpose

Phase 6A evaluates simple historical baseline predictors before training any machine learning model.

The baselines provide a reproducible reference point for the incident-time regression task:

- Target: `Min Delay`
- Validation split: 2023
- Test split: 2024
- Inputs: leakage-safe historical features already generated under `data/processed/modeling/`

No XGBoost, Optuna, SHAP, FastAPI, or frontend code is part of this phase.

## Evaluated Predictors

The baseline evaluation script evaluates these prior-only historical predictors:

- `prior_global_mean_delay`
- `prior_mode_mean_delay`
- `prior_route_mean_delay`
- `prior_route_hour_mean_delay`
- `prior_incident_mean_delay`
- `prior_route_hour_7d_mean_delay`

These columns are created by `src/features/build_features.py` using records with timestamps strictly before the current incident timestamp.

## Metrics

For each baseline and split, the script reports:

- MAE
- RMSE
- R2
- number of rows evaluated
- number of missing predictions before fallback
- percent of missing predictions before fallback

The best baseline is selected on validation MAE, with RMSE and R2 used only as tie-breakers. The selected baseline is then summarized by `mode` for validation and test rows.

## Fallback Policy

If a baseline has missing predictions, the script also evaluates a filled version using this order:

1. Baseline prediction
2. `prior_route_mean_delay`
3. `prior_mode_mean_delay`
4. `prior_global_mean_delay`
5. Training target mean

The fallback columns are historical features generated from prior rows only. The final training target mean fallback is computed from the training split only.

## Command

Run from the repository root after building modeling features:

```bash
python3 -m src.models.evaluate_baselines \
  --modeling-dir data/processed/modeling \
  --output-dir reports/baselines
```

Expected local outputs:

- `reports/baselines/baseline_metrics.json`
- `reports/baselines/baseline_metrics.csv`
- `reports/baselines/best_baseline_by_mode.csv`

These outputs are generated reports. They should not be treated as README or resume metrics unless produced by a clean, reproducible scripted run.
