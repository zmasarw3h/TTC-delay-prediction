# Model Improvement EDA

Phase 11A adds a focused reporting layer for deciding which historical features should be built in Phase 11B. It investigates where the current expected-delay regressor is still wrong, how severe-delay errors concentrate, and whether candidate prior-only historical groupings have enough support to be useful.

This phase does not train models, tune hyperparameters, modify feature engineering, or write model artifacts.

## Required Inputs

The script expects normalized modeling splits and existing prediction/error reports:

```text
data/processed/modeling/train.csv
data/processed/modeling/validation.csv
data/processed/modeling/test.csv
reports/error_analysis/error_summary.json
reports/calibration/calibrated_two_output_predictions_validation.csv
reports/calibration/calibrated_two_output_predictions_test.csv
```

If any required prediction/error file is missing, rerun the existing model reporting pipeline before running this EDA:

```bash
python3 -m src.models.analyze_errors
python3 -m src.models.calibrate_risk_models
```

Use the full documented commands from the README when custom paths are needed.

## Command

Run from the repository root:

```bash
python3 -m src.analysis.model_improvement_eda \
  --modeling-dir data/processed/modeling \
  --error-analysis-dir reports/error_analysis \
  --calibration-dir reports/calibration \
  --output-dir reports/model_improvement_eda \
  --min-group-size 100 \
  --top-n 50
```

## Outputs

The script writes:

```text
reports/model_improvement_eda/error_by_group.csv
reports/model_improvement_eda/error_contribution_by_group.csv
reports/model_improvement_eda/severe_delay_by_group.csv
reports/model_improvement_eda/candidate_group_support.csv
reports/model_improvement_eda/candidate_prior_mean_scores.csv
reports/model_improvement_eda/rolling_window_opportunity.csv
reports/model_improvement_eda/feature_recommendations.csv
reports/model_improvement_eda/model_improvement_eda_summary.json
reports/model_improvement_eda/model_improvement_eda.md
```

Optional matplotlib figures are written under:

```text
reports/model_improvement_eda/figures/
```

## Interpreting Candidate Support

`candidate_group_support.csv` reports strict prior-only counts for validation and test rows. Counts use records with `ts < current ts`; rows at the same timestamp are not treated as prior history.

Use these columns as a practical support check:

- `pct_with_prior_1`: feature can usually be populated without fallback.
- `pct_with_prior_20`: grouping has enough repeated history to be more stable.
- `pct_with_prior_50`: grouping is likely robust enough for direct means or rates.
- `median_prior_count` and `p25_prior_count`: low values indicate sparse groups that need fallback.

High-cardinality groupings such as `Location` and `Location + Incident` should only move into Phase 11B if coverage is healthy and a clear fallback strategy is documented.

## Using Recommendations For Phase 11B

Start with `feature_recommendations.csv`. Prioritize rows where `recommended_for_phase_11b` is true and review the `risks` column before implementation.

Expected high-value candidates include route-incident prior means, mode-incident fallback means, route-direction means, route-incident counts, and recent 30-day route or incident history. Location-level features should use minimum support thresholds and fallbacks because location text remains high-cardinality even after normalization.

Any Phase 11B implementation must preserve the project rules:

- Fit preprocessing only on training data.
- Use chronological splits.
- Use `shift(1)` or equivalent strict prior-only logic before target-derived historical features.
- Ensure validation/test historical values use only prior records.
