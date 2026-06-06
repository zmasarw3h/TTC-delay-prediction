You are working in the GitHub repo TTC-delay-prediction.

Current repo state:

The project is functionally complete and ready for final public/resume polish.
It includes:
data cleaning and categorical normalization
leakage-safe feature engineering with v1/v2 historical features
chronological train/validation/test splits
baseline evaluation
XGBoost regression
model experiments
calibrated severe-delay risk classifiers
explainability reports
FastAPI prediction API
historical feature lookup for inference
local frontend demo
Final current metrics:
expected-delay regression test MAE: about 7.76 minutes
route-history baseline test MAE: about 8.98 minutes
MAE improvement: about 13.6%
30+ min severe-delay test ROC-AUC: about 0.905
30+ min severe-delay test PR-AUC: about 0.563
30+ min severe-delay test recall: about 0.761
60+ min severe-delay test ROC-AUC: about 0.952
60+ min severe-delay test PR-AUC: about 0.437
60+ min severe-delay test recall: about 0.822
AGENTS.md defines project rules.

Goal:
Make the repository ready for a public GitHub portfolio and resume review.

Important constraints:

Do not train models unless running very small smoke checks.
Do not modify model logic unless fixing obvious bugs/docs mismatches.
Do not commit raw data, processed data, generated reports, or large model artifacts.
Do not fabricate screenshots, metrics, or claims.
Do not make deployment claims.
Do not say the system is production-ready.
Keep language accurate: this is a local/demo-ready ML system.
Follow AGENTS.md.

Main deliverable:
Polish the repo so a recruiter, hiring manager, or technical reviewer can quickly understand:

what the project does,
why it is technically credible,
how to run it locally,
what the final metrics are,
what the limitations are,
how the code is organized.

Tasks:

1. Final repository audit

Review the repo structure and identify stale, duplicate, misleading, or obsolete files/docs.

Check:

README
docs/
src/
tests/
requirements.txt or dependency files
.gitignore
AGENTS.md
API/static frontend files
any committed generated files

Make safe cleanup edits:

remove or update stale references to old notebook-only workflow
remove or update stale references to old pre-normalization model metrics
remove or update stale references to manual historical feature entry as the default API behavior
make sure docs say historical features are now computed automatically by API lookup when basic incident details + timestamp are provided
ensure generated data/artifact/report paths are gitignored
ensure public repo does not accidentally commit raw TTC files, processed CSVs, model artifacts, or generated report outputs

Do not delete important source code.

2. README overhaul

Rewrite or heavily polish README.md so it is public-GitHub ready.

README should include:

Project title:
“TTC Delay Prediction & Severe-Risk Forecasting”
Short summary:
A calibrated two-output ML system that predicts expected TTC bus/streetcar incident delay and calibrated severe-delay risk from incident-time features and prior historical records.
Badges if appropriate:
Python
tests
FastAPI
XGBoost
Do not add fake CI badge unless CI exists.
Demo screenshot section:
Reference screenshots under docs/images/ only if they already exist or are created from user-provided/local screenshots.
If screenshots are not available, include a clear placeholder comment or section saying screenshots can be added later.
Do not fabricate image files.
Problem statement:
Explain incident-time delay prediction and severe-delay risk.
Final results table:
Include:
baseline test MAE: 8.98 min
final regression test MAE: 7.76 min
improvement: about 13.6%
30+ severe-delay ROC-AUC/PR-AUC/recall
60+ severe-delay ROC-AUC/PR-AUC/recall
State these are chronological 2024 holdout metrics.
Technical approach:
categorical normalization
leakage-safe historical features
chronological splits
log-target XGBoost regression
calibrated severe-delay classifiers
historical feature lookup
FastAPI + local frontend
Architecture:
Include a Mermaid diagram if appropriate:
raw TTC data -> cleaning/normalization -> feature engineering -> models -> calibration -> historical lookup -> API/frontend
Repository structure:
Brief tree-style overview of key folders.
Quickstart:
Include commands to:
create virtual environment
install requirements
run tests
run FastAPI/frontend locally
open localhost URL
API endpoints:
GET /
GET /health
GET /model-info
GET /historical-lookup-info
POST /predict-delay
POST /compute-historical-features, if implemented
GET /model-options
POST /match-location
Example prediction request:
Basic fields only:
mode
Route
Direction
Incident
Location
timestamp
Explain historical features are computed automatically from local prior records.
Reproducibility commands:
Include the main pipeline commands in order, but clearly mark them as optional because full data/artifact regeneration requires local data files.
Limitations:
local demo, not production deployment
historical lookup uses local CSV, not live TTC feeds
predictions only as current as the local dataset
no weather enrichment
location matching is approximate assistance, not geocoding
model should not be used for operational decisions without validation on live data
Resume-ready project highlights:
Include 2–3 concise bullets in README or docs, but keep them factual and not exaggerated.

3. Add or polish model card

Create or update:

docs/model_card.md

Include:

intended use
non-intended use
data source/coverage summary
target definition
target filtering policy: 0 <= Min Delay <= 240
chronological split:
train: 2014–2022
validation: 2023
test: 2024
leakage controls:
prior-only historical features
ts < prediction timestamp
same-timestamp rows excluded from each other
Min Gap excluded
categorical normalization:
mode
Route
Direction
Incident
Location
feature summary:
time features
categorical features
v1 historical features
v2 historical features
severe-rate rolling features
model choices:
expected-delay regression with log-target XGBoost
calibrated severe-delay classifiers for 30+ and 60+
final metrics table
calibration summary
explainability summary
limitations
ethical/operational caution:
local demo only
not for live dispatch decisions without further validation
reproducibility notes

4. Add or polish technical report

Create or update:

docs/technical_report.md

This should be a more detailed narrative than the README.

Suggested sections:

Overview
Dataset and target policy
Cleaning and categorical normalization
Feature engineering
Leakage prevention
Baseline modeling
Regression modeling
Severe-delay risk classification
Probability calibration
Error analysis
Explainability
API and frontend demo
Final metrics
Limitations
Future work

Keep it concise but professional.

5. Add docs index

Create or update:

docs/README.md

It should link to all important docs:

model card
technical report
API service docs
API input contract
historical feature lookup docs
feature engineering docs
categorical normalization docs
category quality audit docs
model improvement EDA docs
explainability docs
probability calibration docs
any other relevant docs

Add one-line descriptions for each.

6. Add architecture diagram

Create:

docs/architecture.md

Include:

Mermaid architecture diagram
description of each component:
data cleaning
categorical normalization
feature engineering
model training
calibration
historical lookup
API
frontend
note which outputs are generated locally and not committed

Optionally include the same Mermaid diagram in README.

7. Add final QA checklist

Create:

docs/final_qa_checklist.md

Include checklist items:

pytest passes
API starts locally
frontend loads
/predict-delay works with basic fields only
historical lookup info works
no generated data/artifacts committed
README metrics match latest reports
docs do not mention stale metrics
screenshots updated if available
git status clean before publishing
public repo privacy check
Add run commands / Makefile

If no Makefile exists, create a simple Makefile with safe commands:

make test
make api
make build-features
make baselines
make train
make experiments
make calibrate
make explain
make eda

Do not make commands destructive.

Add comments that data/artifact generation requires local files not committed to Git.

If a Makefile already exists, update it carefully.

8. Add environment/config example

Create or update:

.env.example

Include:

TTC_MODEL_ARTIFACT_PATH=artifacts/calibration/calibrated_two_output_model.joblib
TTC_HISTORICAL_FEATURE_DATA_PATH=data/processed/modeling/modeling_dataset.csv

Do not commit real secrets.

9. Gitignore/public safety

Review .gitignore.

Ensure it excludes:

raw data
processed data
model artifacts
generated reports
caches
notebook checkpoints
local env files
Python cache files

Do not ignore source code/docs/tests.

10. Code polish

Do light cleanup only:

improve docstrings where missing
remove obviously unused imports
ensure scripts have import-safe structure
ensure CLIs use if __name__ == "__main__"
ensure user-facing errors are clear
ensure API docs/comments align with current historical lookup behavior

Do not refactor heavily.

11. Tests

Run or preserve tests.

Add lightweight tests only if needed for public-readiness issues:

README examples do not include leakage fields
.env.example exists
/predict-delay can work with basic fields + timestamp in existing fake-artifact tests
docs links or expected files exist if there is already a docs test pattern

Do not add brittle tests that require real raw data/model artifacts.

12. Screenshots

- If local frontend screenshots already exist or user-provided screenshots are available in the repo, organize them under docs/images/.
- If not, leave clear instructions for adding screenshots later.
- Do not fabricate frontend screenshots.

13. Figures and graphs for public-facing reports

Where useful and supported by existing generated outputs, add lightweight figures/graphs to the README and docs.

Use existing generated plots if available. Do not fabricate plots or metrics.

Recommended figures:
- Model performance comparison:
  - baseline MAE vs fixed XGBoost MAE vs final log-target XGBoost MAE
- Severe-delay classifier summary:
  - 30+ and 60+ ROC-AUC / PR-AUC / recall comparison
- Pipeline/architecture diagram:
  - raw data -> cleaning -> normalization -> feature engineering -> modeling -> calibration -> API/frontend
- Error-analysis visual:
  - top error contribution by route or incident, if already generated
- Explainability visual:
  - top feature importance chart, if already generated
- Calibration visual:
  - calibration curve or risk-band calibration summary, if already generated

If figures are generated or copied, place them under:

docs/images/

Use stable filenames, for example:
- docs/images/model_performance_comparison.png
- docs/images/severe_delay_metrics.png
- docs/images/top_error_contribution_by_incident.png
- docs/images/top_feature_importance.png
- docs/images/calibration_summary.png
- docs/images/frontend_demo.png

If creating new figures from existing CSV/JSON reports:
- create a small script, e.g. `scripts/create_public_figures.py` or `src/analysis/create_public_figures.py`
- use matplotlib only
- do not require seaborn
- do not require raw data if existing summary reports are enough
- make the script import-safe
- document the command in README/docs
- do not commit large generated datasets

Include figures in:
- README, where useful
- docs/technical_report.md
- docs/model_card.md, if appropriate

Keep figures concise and readable. Do not overcrowd the README.

14. Final public-facing wording

Use accurate language:

“local demo”
“chronological 2024 holdout”
“calibrated severe-delay probabilities”
“historical features computed from prior records”
“not production deployed”
“not connected to live TTC feeds”

Avoid:

“production-ready”
“real-time TTC prediction”
“guaranteed probability”
“90% accurate”
any claim not supported by reports

15. Validation commands

At the end, run:

pytest

Also run lightweight smoke checks if possible:

import FastAPI app
ensure README exists
ensure docs/model_card.md exists
ensure docs/technical_report.md exists

Do not require full pipeline reruns unless necessary.

Acceptance criteria:

README is polished and public-ready.
Final metrics are accurate and current.
Model card exists.
Technical report exists.
Docs index exists.
Architecture doc exists.
Final QA checklist exists.
.env.example exists.
Makefile or equivalent command documentation exists.
.gitignore protects generated data/artifacts/reports.
API/frontend docs reflect historical lookup behavior.
No stale notebook-only language remains as the primary project description.
No generated data/model artifacts are committed.
Tests pass.
Repo is ready to be made public and linked on a resume.

Commit message:
Polish repository for public portfolio release