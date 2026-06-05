# TTC Bus & Streetcar Delay Prediction

Portfolio-grade machine learning project for incident-time TTC bus and streetcar delay prediction.

The final system will estimate expected delay duration in minutes after an incident has been reported, using information reasonably available at report time. This is not a future delay forecasting project.

## Current Status

The project currently has reproducible data-cleaning, target-diagnostics, leakage-safe feature-building, baseline evaluation, first fixed-configuration model-training, fixed-model error-analysis, fixed model-improvement experiment scripts, a Phase 7B two-output delay/risk modeling script, Phase 7C severe-delay probability calibration, Phase 8 model explainability reports, API-ready input validation utilities, a Phase 9 local FastAPI prediction service, and a Phase 10 FastAPI-served local demo frontend with planner-oriented controls. SHAP is optional and not required for the default explainability workflow.

## Project Structure

```text
data/
  raw/          # local raw TTC files, gitignored
  processed/    # cleaned and modeling CSV outputs, gitignored
docs/           # project planning and modeling assumptions
notebooks/      # clean explanatory notebooks for later phases
src/data/       # reproducible loading and cleaning scripts
src/features/   # leakage-safe feature engineering
src/models/     # reproducible modeling scripts
src/api/        # FastAPI service and input validation
reports/        # generated reporting outputs, gitignored except placeholders
artifacts/      # model/pipeline artifacts, gitignored except placeholders
tests/          # lightweight validation tests
```

## Setup

Create and activate a virtual environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Cleaning

Place raw TTC files under:

```text
data/raw/
  TTC Bus Delays Data/
    *.xlsx
  TTC Streetcar Delays Data/
    *.xlsx
```

Run the cleaner from the repository root. By default it looks for bus and streetcar folders under `data/raw`:

```bash
python -m src.data.clean_data
```

You can also provide explicit paths:

```bash
python -m src.data.clean_data \
  --bus-raw-dir "data/raw/TTC Bus Delays Data" \
  --streetcar-raw-dir "data/raw/TTC Streetcar Delays Data" \
  --processed-dir data/processed
```

The script supports bus-only or streetcar-only runs if only one raw directory is provided. It writes cleaned CSV files to `data/processed/`.

`Min Gap` is retained in the cleaned data for auditability, but it is leakage-sensitive for later modeling and should not be used in the main model unless explicitly justified.

## Target Diagnostics

Before feature engineering or model training, inspect the `Min Delay` target distribution and largest delay records:

```bash
python3 -m src.features.diagnose_target \
  --input data/processed/ttc_delays_cleaned.csv \
  --output-dir reports/target_diagnostics
```

This writes target diagnostics under `reports/target_diagnostics/`. The command does not train models.

## Feature Building

Build the Phase 5 modeling-ready dataset and chronological train/validation/test splits:

```bash
python3 -m src.features.build_features \
  --input data/processed/ttc_delays_cleaned.csv \
  --output-dir data/processed/modeling \
  --max-delay-minutes 240
```

This creates generated files under `data/processed/modeling/`:

```text
modeling_dataset.csv
train.csv
validation.csv
test.csv
feature_metadata.json
```

The feature-building step applies the documented `0 <= Min Delay <= 240` modeling target policy and creates leakage-safe historical features using prior rows only. It does not train models.

## Baseline Evaluation

Evaluate Phase 6A historical baseline predictors on the existing validation and test splits:

```bash
python3 -m src.models.evaluate_baselines \
  --modeling-dir data/processed/modeling \
  --output-dir reports/baselines
```

This writes baseline metrics under `reports/baselines/`. The baseline step uses existing leakage-safe historical features and does not train XGBoost or any other ML model.

## First Model Training

Train the Phase 6B fixed-configuration XGBoost model using the existing modeling splits:

```bash
python3 -m src.models.train_model \
  --modeling-dir data/processed/modeling \
  --reports-dir reports/models \
  --artifacts-dir artifacts/models \
  --baseline-report reports/baselines/baseline_metrics.json
```

This writes local model reports under `reports/models/` and the local model artifact under `artifacts/models/`. This is the first fixed XGBoost model, not an Optuna-tuned model.

## Model Error Analysis

Analyze Phase 6C validation/test residuals for the existing fixed XGBoost model:

```bash
python3 -m src.models.analyze_errors \
  --modeling-dir data/processed/modeling \
  --model-path artifacts/models/xgb_delay_model.joblib \
  --baseline-report reports/baselines/baseline_metrics.json \
  --output-dir reports/error_analysis
```

This writes generated error-analysis reports under `reports/error_analysis/`. The command scores the existing model artifact only; it does not train or modify the model.

## Model Improvement Experiments

Run Phase 7A fixed model-improvement experiments:

```bash
python3 -m src.models.run_experiments \
  --modeling-dir data/processed/modeling \
  --baseline-report reports/baselines/baseline_metrics.json \
  --fixed-model-report reports/models/model_metrics.json \
  --reports-dir reports/experiments \
  --artifacts-dir artifacts/experiments \
  --selection-metric validation_mae
```

This compares predefined modeling strategies for severe-delay handling, streetcar performance, and regression-to-the-mean behavior. It does not perform broad hyperparameter search or Optuna tuning. Validation metrics select the experiment; test metrics are reported for the selected/fixed experiment comparison workflow.

Generated reports:

```text
reports/experiments/experiment_metrics.csv
reports/experiments/experiment_metrics_by_mode.csv
reports/experiments/experiment_high_delay_metrics.csv
reports/experiments/experiment_selection.json
reports/experiments/experiment_summary.json
```

The selected experiment artifact is written to `artifacts/experiments/selected_experiment.joblib`. The existing Phase 6B fixed model artifact is not overwritten.

## Two-Output Delay And Risk Model

Train the Phase 7B two-output model:

```bash
python3 -m src.models.train_risk_models \
  --modeling-dir data/processed/modeling \
  --selected-regressor-path artifacts/experiments/selected_experiment.joblib \
  --reports-dir reports/risk_models \
  --artifacts-dir artifacts/risk_models \
  --thresholds 30,60
```

This combines the selected Phase 7A expected-delay regressor with severe-delay risk classifiers for `Min Delay >= 30` and `Min Delay >= 60`. Operating probability thresholds are selected on validation data only, then applied to test.

Generated reports:

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

The two-output artifact is written to `artifacts/risk_models/two_output_model.joblib`. API and frontend code are still not implemented.

## Calibrated Severe-Delay Probabilities

Calibrate the Phase 7B severe-delay classifiers:

```bash
python3 -m src.models.calibrate_risk_models \
  --modeling-dir data/processed/modeling \
  --phase-7b-artifact-path artifacts/risk_models/two_output_model.joblib \
  --selected-regressor-path artifacts/experiments/selected_experiment.joblib \
  --reports-dir reports/calibration \
  --artifacts-dir artifacts/calibration \
  --thresholds 30,60
```

This Phase 7C workflow trains base severe-delay classifiers on pre-2022 training rows, fits sigmoid and isotonic calibrators on 2022 training rows only, selects calibration methods and operating cutoffs on validation data only, and evaluates test data after those choices are fixed. The Phase 7B uncalibrated outputs are preserved as reference diagnostics, and Phase 7C writes only to new calibration paths.

Generated reports:

```text
reports/calibration/calibration_metrics.csv
reports/calibration/calibration_threshold_table.csv
reports/calibration/calibration_bin_table.csv
reports/calibration/calibrated_risk_band_summary.csv
reports/calibration/calibration_selection.json
reports/calibration/calibrated_two_output_summary.json
reports/calibration/calibrated_two_output_predictions_validation.csv
reports/calibration/calibrated_two_output_predictions_test.csv
```

The calibrated artifact is written to `artifacts/calibration/calibrated_two_output_model.joblib`. These calibrated probabilities are served by the local FastAPI prediction service. Frontend code is still not implemented.

## Model Explainability

Generate Phase 8 explainability reports for the existing calibrated two-output artifact:

```bash
python3 -m src.models.explain_models \
  --modeling-dir data/processed/modeling \
  --artifact-path artifacts/calibration/calibrated_two_output_model.joblib \
  --output-dir reports/explainability
```

The default split is validation and the default maximum explanation sample is 5,000 rows. The script does not retrain models. It uses model-agnostic permutation importance by default and can optionally attempt a SHAP smoke test with `--include-shap`.

Generated reports:

```text
reports/explainability/permutation_importance_regression.csv
reports/explainability/permutation_importance_risk_30.csv
reports/explainability/permutation_importance_risk_60.csv
reports/explainability/global_feature_importance.csv
reports/explainability/representative_prediction_examples.csv
reports/explainability/explainability_summary.json
reports/explainability/figures/top_features_regression.png
reports/explainability/figures/top_features_risk_30.png
reports/explainability/figures/top_features_risk_60.png
```

These are local generated reports. The API uses the existing calibrated artifact and does not generate explainability output at request time.

## FastAPI Prediction Service

Run the local Phase 9 API service and Phase 10 demo frontend from the repository root:

```bash
uvicorn src.api.app:app --reload
```

Open the local demo UI:

```text
http://127.0.0.1:8000/
```

The demo includes exactly two presets:

- Bus incident
- Streetcar incident

The local demo UI includes searchable route and incident controls, mode and direction dropdowns, and location matching against known model locations where available. A high-confidence location match can be submitted automatically; medium-confidence matches are shown as suggestions. The app still expects engineered prior-delay features and does not implement raw TTC incident-to-feature lookup.

By default, the API loads:

```text
artifacts/calibration/calibrated_two_output_model.joblib
```

You can override the artifact path with:

```bash
TTC_MODEL_ARTIFACT_PATH=/path/to/calibrated_two_output_model.joblib \
uvicorn src.api.app:app --reload
```

Endpoints:

- `GET /health`
- `GET /model-info`
- `GET /model-options`
- `POST /match-location`
- `POST /predict-delay`
- `GET /`
- `GET /static/styles.css`
- `GET /static/app.js`

Example prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict-delay \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "bus",
    "Route": "29",
    "Direction": "N",
    "Incident": "Mechanical",
    "Location": "Dufferin Station",
    "timestamp": "2024-02-03T08:30:00",
    "prior_route_mean_delay": 10.0,
    "prior_route_hour_mean_delay": 12.0,
    "prior_incident_mean_delay": 9.0,
    "prior_mode_mean_delay": 8.0,
    "prior_global_mean_delay": 7.0,
    "prior_route_hour_7d_mean_delay": 11.0
  }'
```

The prediction endpoint can derive time fields and the Ontario holiday flag from `timestamp`. It still expects prior-only historical delay features, or it will rely on model-pipeline imputation with a warning. It does not yet implement raw incident-to-feature lookup or weather enrichment. The frontend is a local demo UI served by FastAPI, not a deployed application.

## API Input Validation

API-ready validation helpers live in `src/api/input_validation.py`. They normalize engineered model feature payloads, preserve route categories such as `29`, `501`, and `RAD`, convert missing categorical inputs to `Unknown`, keep missing numeric historical features as `None` for model-pipeline imputation, and reject leakage-sensitive fields. See [API input contract](docs/api_input_contract.md) and [FastAPI service docs](docs/api_service.md) for the current contract.

## Planning Docs

- [Project definition](docs/project_definition.md)
- [Current state audit](docs/current_state_audit.md)
- [Model design](docs/model_design.md)
- [Modeling data policy](docs/modeling_data_policy.md)
- [Feature engineering](docs/feature_engineering.md)
- [Modeling baselines](docs/modeling_baselines.md)
- [Model training](docs/model_training.md)
- [Model error analysis](docs/error_analysis.md)
- [Model improvement experiments](docs/model_experiments.md)
- [Two-output delay and risk model](docs/two_output_model.md)
- [Probability calibration](docs/probability_calibration.md)
- [Model explainability](docs/explainability.md)
- [API input contract](docs/api_input_contract.md)
- [FastAPI service](docs/api_service.md)
