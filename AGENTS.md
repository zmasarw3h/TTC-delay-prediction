# AGENTS.md

## Project Context

This repository is a portfolio-grade machine learning project for TTC bus and streetcar delay prediction.

The project goal is incident-time prediction: after a TTC incident is reported, predict expected delay duration in minutes using information reasonably available at report time.

This is not a future delay forecasting project.

Primary target:
- `Min Delay`

Primary task:
- Regression

Optional secondary task:
- Severe-delay risk classification, e.g. `Min Delay >= 15`

## Core Rules

- Do not make unsupported performance claims.
- Do not use final README or resume metrics unless they come from a clean, reproducible scripted run.
- Do not tune on the test set.
- Do not use random train/test splits as the final evaluation setup.
- Use chronological splits:
  - Train: 2014-2022
  - Validation: 2023
  - Test: 2024
- Fit preprocessing only on training data.
- Use `shift(1)` before any target-derived rolling or historical feature.
- Historical features for validation/test rows must use only prior records.

## Leakage-Sensitive Features

Do not include these in the main model unless explicitly justified in documentation:

- `Min Delay` as an input feature
- `Min Gap`
- same-day aggregates that include the current or future incidents
- future rolling averages
- post-incident or resolution fields
- target encodings fit on the full dataset

## Repository Direction

The notebook `TTC_Delays_Cleaned.ipynb` is an exploratory reference, not the final implementation.

Future work should move logic into reproducible scripts:

- `src/data/`
- `src/features/`
- `src/models/`
- `src/api/`

Keep notebooks clean and explanatory only.

## Documentation

When changing modeling assumptions, update the relevant docs:

- `docs/project_definition.md`
- `docs/current_state_audit.md`
- `docs/model_design.md`

## Style

- Prefer clear, simple Python modules over large notebooks.
- Avoid hard-coded local paths.
- Keep generated artifacts out of the repo unless intentionally documented.
- Use Markdown docs that are practical and implementation-ready.