# Project Definition

## Recommended Title

**TTC Bus & Streetcar Delay Prediction and Explainability System**

## Project Goal

Build an incident-time prediction system for TTC bus and streetcar delays. The system should estimate the expected delay duration in minutes after an incident has been reported.

This is not a future delay forecasting project. The goal is not to predict whether delays will occur later in the day, tomorrow, or on a route before any incident has happened. The prediction moment is after an operational delay incident is known, using only information that would reasonably be available at report time.

## Prediction Target

The primary target is:

- `Min Delay`: expected delay duration in minutes.

The main modeling task is regression.

## Intended Use Case

After a TTC bus or streetcar incident is reported, the system should estimate the likely delay duration and help identify high-risk patterns across:

- route
- mode
- incident type
- location
- time of day, day of week, month, weekend, and holiday timing
- weather conditions
- historical route, route-hour, and incident-type delay behavior

The final result should support portfolio-grade analysis and demonstration, not operational deployment claims. It should communicate model performance clearly, avoid data leakage, and provide explainability that is understandable to non-technical readers.

## Planned Deliverables

The final project should include:

- Reproducible data pipeline for bus and streetcar delay data.
- Leakage-safe feature engineering.
- Chronological train, validation, and test split.
- Baseline comparison.
- XGBoost regression model.
- Optuna hyperparameter tuning if the baseline model and data size justify it.
- SHAP explainability for global and local model interpretation.
- FastAPI prediction endpoint.
- Thin frontend or demo interface.
- Polished `README.md` with resume-safe metrics and clear caveats.

## Scope for Current Phase

The current phase is documentation and planning only:

- Define the project direction.
- Audit the current notebook.
- Design the leakage-safe modeling setup.

Do not convert the notebook into scripts yet. Do not train new models yet. Do not create the API or frontend yet.
