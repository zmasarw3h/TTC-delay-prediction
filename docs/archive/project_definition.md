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

After a TTC bus or streetcar incident is reported, the system estimates the likely delay duration and calibrated severe-delay risk from:

- route
- mode
- incident type
- location
- time of day, day of week, month, weekend, and holiday timing
- historical route, route-hour, incident-type, location, and recent severe-delay behavior

The final result supports portfolio-grade analysis and local demonstration, not operational deployment claims. It communicates model performance clearly, avoids data leakage, and provides explainability that is understandable to non-technical readers. Weather enrichment remains future work.

## Implemented Deliverables

The current reproducible implementation includes:

- Reproducible data pipeline for bus and streetcar delay data.
- Leakage-safe feature engineering.
- Chronological train, validation, and test split.
- Baseline comparison.
- XGBoost expected-delay regression.
- Calibrated severe-delay risk classifiers for `30+` and `60+` minute thresholds.
- Model-agnostic permutation-importance explainability reports.
- FastAPI local prediction endpoint.
- Local static frontend demo.
- Historical feature lookup for API inference from local prior records.
- Polished public documentation with resume-safe metrics and clear caveats.

## Current Scope

The current scope is a local/demo-ready ML system:

- Keep raw data, processed data, generated reports, and model artifacts out of the public repo.
- Avoid deployment and production-readiness claims.
- Report only metrics from clean scripted runs.
- Treat the notebook as historical exploratory reference, not the final implementation.
