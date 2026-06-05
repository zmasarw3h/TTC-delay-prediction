"""FastAPI application for TTC incident-time delay predictions."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException

from src.api.prediction import CalibratedDelayPredictionService
from src.api.schemas import (
    DelayPredictionResponse,
    EngineeredIncidentFeatures,
    HealthResponse,
    ModelInfoResponse,
)


app = FastAPI(
    title="TTC Delay Prediction API",
    version="0.1.0",
    description="Local FastAPI service for calibrated TTC delay and severe-delay risk predictions.",
)
prediction_service = CalibratedDelayPredictionService()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_artifact_loaded=prediction_service.is_loaded,
        artifact_exists=prediction_service.artifact_path.exists(),
        artifact_path=str(prediction_service.artifact_path),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        return ModelInfoResponse(**prediction_service.model_info())
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict-delay", response_model=DelayPredictionResponse)
def predict_delay(payload: EngineeredIncidentFeatures) -> DelayPredictionResponse:
    try:
        result = prediction_service.predict(payload.model_dump())
        return DelayPredictionResponse(**asdict(result))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
