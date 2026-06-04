# TTC Bus & Streetcar Delay Prediction

Portfolio-grade machine learning project for incident-time TTC bus and streetcar delay prediction.

The final system will estimate expected delay duration in minutes after an incident has been reported, using information reasonably available at report time. This is not a future delay forecasting project.

## Current Status

The project is currently in the data-cleaning pipeline phase. The repository now has a production-style structure for later data, feature, modeling, API, and frontend work, but modeling, SHAP explainability, FastAPI, and frontend code are not implemented yet.

## Project Structure

```text
data/
  raw/          # local raw TTC files, gitignored
  processed/    # cleaned CSV outputs, gitignored
docs/           # project planning and modeling assumptions
notebooks/      # clean explanatory notebooks for later phases
src/data/       # reproducible loading and cleaning scripts
src/features/   # planned leakage-safe feature engineering
src/models/     # planned modeling code
src/api/        # planned API code
reports/        # planned generated reporting outputs
artifacts/      # planned model/pipeline artifacts, gitignored
tests/          # lightweight validation tests
```

## Setup

Create and activate a virtual environment, then install the Phase 4 dependencies:

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

## Planning Docs

- [Project definition](docs/project_definition.md)
- [Current state audit](docs/current_state_audit.md)
- [Model design](docs/model_design.md)
- [Modeling data policy](docs/modeling_data_policy.md)
