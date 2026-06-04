# Probability Calibration

## Purpose

Phase 7B severe-delay classifiers rank high-risk incidents well, but raw classifier scores can be overconfident. Phase 7C calibrates the severe-delay outputs so later API responses can present them as probabilities rather than only as relative risk scores.

The expected-delay regression output is preserved. This phase changes only the severe-delay probability modeling layer.

## Split Logic

The calibration workflow keeps the existing chronological modeling splits:

- Train: 2014-2022
- Validation: 2023
- Test: 2024

Within the training split, Phase 7C creates two time-based subsets:

- Base classifier training: rows with `ts < 2022-01-01`
- Calibration fitting: rows from calendar year 2022

If `ts` is missing, unparseable, or either subset is empty, the script fails clearly. Validation data is used only for calibration-method choice and operating-threshold selection. Test data is evaluated only after those choices are fixed.

## Methods Compared

For each severe-delay threshold:

- `Min Delay >= 30`
- `Min Delay >= 60`

The script compares:

- Uncalibrated base classifier
- Sigmoid/Platt calibration
- Isotonic calibration

Base classifiers are trained on pre-2022 training rows. Sigmoid and isotonic calibrators are fit only on the 2022 calibration subset.

## Metrics

For each threshold, method, and evaluation split, the script reports:

- Positive rate
- ROC-AUC
- PR-AUC / average precision
- Log loss
- Brier score
- Expected calibration error, using 10 equal-width probability bins
- Precision
- Recall
- F1
- Accuracy
- Confusion matrix counts at the selected operating cutoff

It also writes 10-bin calibration tables with row count, mean predicted probability, actual severe-delay rate, and absolute calibration error.

## Operating Threshold Selection

Operating probability cutoffs are selected on validation data only. Candidate cutoffs are:

```text
0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90
```

Selection rule:

1. Prefer recall >= 0.70.
2. Among qualifying cutoffs, choose the highest F1.
3. If no cutoff qualifies, choose the highest F1.

## Calibration Method Selection

The final calibration method is selected independently for each severe-delay threshold using validation data only:

1. Prefer the method with the lowest validation Brier score.
2. If Brier scores are within 2%, prefer lower validation ECE.
3. If still close, prefer higher validation PR-AUC.
4. If still close, prefer the simpler method order: sigmoid, isotonic, uncalibrated.

## Risk Bands

Calibrated risk bands use fixed probability ranges:

```text
low: p < 0.10
medium: 0.10 <= p < 0.30
high: p >= 0.30
```

These bands are intended for later prediction-service outputs after calibration is complete.

## Outputs

Run:

```bash
python3 -m src.models.calibrate_risk_models \
  --modeling-dir data/processed/modeling \
  --phase-7b-artifact-path artifacts/risk_models/two_output_model.joblib \
  --selected-regressor-path artifacts/experiments/selected_experiment.joblib \
  --reports-dir reports/calibration \
  --artifacts-dir artifacts/calibration \
  --thresholds 30,60
```

Reports:

```text
reports/calibration/calibration_metrics.csv
reports/calibration/calibration_threshold_table.csv
reports/calibration/calibration_bin_table.csv
reports/calibration/calibrated_risk_band_summary.csv
reports/calibration/calibration_selection.json
reports/calibration/calibrated_two_output_summary.json
reports/calibration/calibrated_two_output_predictions_validation.csv
reports/calibration/calibrated_two_output_predictions_test.csv
reports/calibration/figures/calibration_curve_threshold_30_validation.png
reports/calibration/figures/calibration_curve_threshold_60_validation.png
reports/calibration/figures/calibration_curve_threshold_30_test.png
reports/calibration/figures/calibration_curve_threshold_60_test.png
```

Artifact:

```text
artifacts/calibration/calibrated_two_output_model.joblib
```

The artifact includes the expected-delay regressor, selected calibrated risk classifiers, all compared risk classifiers by method, selected calibration method by threshold, selected operating cutoff by threshold, approved feature columns, target column, calibrated risk-band definitions, and metadata for later API integration.

Phase 7C writes only under `reports/calibration/` and `artifacts/calibration/`. It does not overwrite Phase 7B reports or artifacts.

## Out Of Scope

This phase does not implement Optuna, SHAP, FastAPI, or frontend code.
