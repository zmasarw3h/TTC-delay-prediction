"""FastAPI application for TTC incident-time delay predictions."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.options import (
    location_options_from_categories,
    match_location,
    model_options_from_categories,
)
from src.api.prediction import CalibratedDelayPredictionService
from src.api.schemas import (
    DelayPredictionResponse,
    EngineeredIncidentFeatures,
    HealthResponse,
    LocationMatchRequest,
    LocationMatchResponse,
    ModelInfoResponse,
    ModelOptionsResponse,
)


app = FastAPI(
    title="TTC Delay Prediction API",
    version="0.1.0",
    description="Local FastAPI service for calibrated TTC delay and severe-delay risk predictions.",
)
prediction_service = CalibratedDelayPredictionService()
STATIC_DIR = Path(__file__).resolve().parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


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


@app.get("/model-options", response_model=ModelOptionsResponse)
def model_options() -> ModelOptionsResponse:
    try:
        options = model_options_from_categories(prediction_service.known_categories)
        return ModelOptionsResponse(**options)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/match-location", response_model=LocationMatchResponse)
def match_location_endpoint(payload: LocationMatchRequest) -> LocationMatchResponse:
    try:
        known_locations = location_options_from_categories(
            prediction_service.known_categories
        )
        match = match_location(payload.location, known_locations)
        return LocationMatchResponse(**match.__dict__)
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
