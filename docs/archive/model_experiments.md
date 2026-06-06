# Model Improvement Experiments

Phase 7A compares a small set of fixed model strategies after Phase 6C error analysis showed three weaknesses in the first fixed XGBoost model:

- short delays are often overpredicted;
- severe delays are often underpredicted;
- streetcar validation/test performance is materially weaker than bus performance.

These experiments are intended to test targeted modeling changes while preserving the project's leakage rules. This phase is not Optuna tuning, does not run SHAP, and does not implement API or frontend code.

## Command

Run from the repository root after modeling splits exist:

```bash
python3 -m src.models.run_experiments \
  --modeling-dir data/processed/modeling \
  --baseline-report reports/baselines/baseline_metrics.json \
  --fixed-model-report reports/models/model_metrics.json \
  --reports-dir reports/experiments \
  --artifacts-dir artifacts/experiments \
  --selection-metric validation_mae
```

The script loads `train.csv`, `validation.csv`, `test.csv`, and `feature_metadata.json` from `data/processed/modeling/`. Feature columns come from metadata after excluding leakage-sensitive and non-feature columns such as `Min Delay`, `Min Gap`, `ts`, and derived severe-delay labels.

## Experiments

### `combined_xgb_fixed`

Reference experiment using the same fixed XGBoost configuration as the Phase 6B model. It is included so every Phase 7A comparison appears in one experiment table.

### `combined_xgb_weighted`

Uses the same combined XGBoost pipeline, but applies fixed training sample weights based on the target delay:

| Delay range | Weight |
|---|---:|
| `0 <= Min Delay <= 15` | 1.0 |
| `16 <= Min Delay <= 30` | 1.5 |
| `31 <= Min Delay <= 60` | 2.0 |
| `61 <= Min Delay <= 120` | 3.0 |
| `121 <= Min Delay <= 240` | 4.0 |

The purpose is to reduce severe-delay underprediction without changing hyperparameters or looking at the test set for model selection.

### `combined_xgb_log_target`

Uses the same combined XGBoost pipeline, but trains on `log1p(Min Delay)`. Predictions are converted back to minutes with `expm1` and clipped to `[0, 240]`.

This tests whether compressing the target scale reduces regression-to-the-mean behavior while keeping outputs on the original delay-minute scale.

### `mode_specific_xgb`

Trains one fixed XGBoost pipeline for bus rows and one for streetcar rows. At prediction time, rows are routed by `mode` to the matching fitted pipeline. The experiment fails clearly if either required mode is missing from training data or if a prediction row has no matching mode-specific model.

This tests whether streetcar performance improves when streetcar patterns are not forced into a single combined model with bus incidents.

## Evaluation

Each experiment is evaluated on validation and test splits. Selection uses validation data only.

Overall metrics:

- MAE
- RMSE
- R2
- mean error
- median absolute error
- p90 absolute error
- p95 absolute error
- row count

Additional reports include by-mode validation/test metrics and high-delay metrics for:

- `Min Delay >= 15`
- `Min Delay >= 30`
- `Min Delay >= 60`

High-delay reporting includes underprediction percent and average underprediction amount.

## Selection Rule

Default selection metric: `validation_mae`.

The default selection rule uses validation metrics only:

1. Find the best validation MAE.
2. Keep experiments within 1% of that best validation MAE.
3. Prefer lower validation MAE for `Min Delay >= 30`.
4. Prefer lower streetcar validation MAE.
5. Prefer the simpler predefined model order:
   - `combined_xgb_fixed`
   - `combined_xgb_weighted`
   - `combined_xgb_log_target`
   - `mode_specific_xgb`

Test metrics are reported for transparency, but they are not used to choose the experiment.

## Outputs

Reports are written under `reports/experiments/`:

- `experiment_metrics.csv`
- `experiment_metrics_by_mode.csv`
- `experiment_high_delay_metrics.csv`
- `experiment_selection.json`
- `experiment_summary.json`

Only the selected experiment artifact is saved:

- `artifacts/experiments/selected_experiment.joblib`

The existing Phase 6B artifact at `artifacts/models/xgb_delay_model.joblib` is not overwritten.
