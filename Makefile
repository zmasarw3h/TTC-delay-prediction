.PHONY: test api build-features baselines train experiments calibrate explain eda

# Data and artifact generation requires local TTC files and writes gitignored outputs.

test:
	pytest

api:
	uvicorn src.api.app:app --reload

build-features:
	python3 -m src.features.build_features --input data/processed/ttc_delays_cleaned.csv --output-dir data/processed/modeling --max-delay-minutes 240

baselines:
	python3 -m src.models.evaluate_baselines --modeling-dir data/processed/modeling --output-dir reports/baselines

train:
	python3 -m src.models.train_model --modeling-dir data/processed/modeling --reports-dir reports/models --artifacts-dir artifacts/models --baseline-report reports/baselines/baseline_metrics.json

experiments:
	python3 -m src.models.run_experiments --modeling-dir data/processed/modeling --baseline-report reports/baselines/baseline_metrics.json --fixed-model-report reports/models/model_metrics.json --reports-dir reports/experiments --artifacts-dir artifacts/experiments --selection-metric validation_mae

calibrate:
	python3 -m src.models.calibrate_risk_models --modeling-dir data/processed/modeling --phase-7b-artifact-path artifacts/risk_models/two_output_model.joblib --selected-regressor-path artifacts/experiments/selected_experiment.joblib --reports-dir reports/calibration --artifacts-dir artifacts/calibration --thresholds 30,60

explain:
	python3 -m src.models.explain_models --modeling-dir data/processed/modeling --artifact-path artifacts/calibration/calibrated_two_output_model.joblib --output-dir reports/explainability

eda:
	python3 -m src.analysis.model_improvement_eda --modeling-dir data/processed/modeling --error-analysis-dir reports/error_analysis --calibration-dir reports/calibration --output-dir reports/model_improvement_eda --min-group-size 100 --top-n 50
