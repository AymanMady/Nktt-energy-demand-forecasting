"""FastAPI application: forecasts, model comparison and historical demand.

Run locally::

    uvicorn src.api.main:app --reload --port 8000

OpenAPI docs: http://localhost:8000/docs
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Annotated, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    MAX_HORIZON_H,
    BacktestResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    HistoricalResponse,
    MetricsResponse,
    ModelsResponse,
)
from src.api.service import (
    DataUnavailableError,
    ForecastService,
    InvalidRequestError,
    ModelNotReadyError,
)
from src.config import get_settings
from src.utils.logging import setup_logging

setup_logging(get_settings().log_level)

TAGS = [
    {"name": "system", "description": "Service health."},
    {
        "name": "forecast",
        "description": "Demand forecasts with uncertainty intervals and peak risk.",
    },
    {
        "name": "models",
        "description": "Model comparison, metrics and backtests from the last training run.",
    },
    {"name": "data", "description": "Observed historical demand."},
]

app = FastAPI(
    title="Nouakchott Energy Demand Forecasting API",
    version="0.1.0",
    description=(
        "Hourly electricity demand forecasts (1 h to 7 days) for a Nouakchott prototype.\n\n"
        "**Prototype data:** the models are trained on *synthetic* data. This is not SOMELEC data. "
        "Peak-risk levels and prediction intervals are experimental."
    ),
    openapi_tags=TAGS,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("API_CORS_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.state.service = ForecastService()


def get_service(request: Request) -> ForecastService:
    return request.app.state.service


Service = Annotated[ForecastService, Depends(get_service)]


@app.exception_handler(ModelNotReadyError)
async def _model_not_ready(_: Request, exc: ModelNotReadyError):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(DataUnavailableError)
async def _data_unavailable(_: Request, exc: DataUnavailableError):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(InvalidRequestError)
async def _invalid_request(_: Request, exc: InvalidRequestError):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(service: Service) -> dict:
    """Liveness + readiness: is a model loaded and is the database reachable?"""
    return service.health()


@app.get("/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast_get(
    service: Service,
    horizon: Annotated[int, Query(ge=1, le=MAX_HORIZON_H, description="Hours to forecast")] = 24,
    include_history_hours: Annotated[int, Query(ge=0, le=720)] = 0,
) -> dict:
    """Forecast the next `horizon` hours after the last observed hour."""
    return service.forecast(horizon, include_history_hours)


@app.post("/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast_post(request: ForecastRequest, service: Service) -> dict:
    """Same as `GET /forecast`, with a JSON body, e.g. `{"horizon": 24}`."""
    return service.forecast(request.horizon, request.include_history_hours)


@app.get("/models", response_model=ModelsResponse, tags=["models"])
def models(service: Service) -> dict:
    """All candidate models per horizon with CV and test metrics, and the selected model."""
    return service.models()


@app.get("/metrics", response_model=MetricsResponse, tags=["models"])
def metrics(service: Service) -> dict:
    """Metrics of the production (selected) model per horizon vs the best baseline."""
    return service.metrics()


@app.get("/backtest", response_model=BacktestResponse, tags=["models"])
def backtest(
    service: Service,
    horizon: Annotated[int, Query(ge=1, le=MAX_HORIZON_H)] = 24,
    days: Annotated[int, Query(ge=1, le=92)] = 14,
) -> dict:
    """Actual vs predicted demand on the held-out test period (selected model, with intervals)."""
    return service.backtest(horizon, days)


def _utc(value: datetime | None) -> pd.Timestamp | None:
    """Naive datetimes are interpreted as UTC (= Nouakchott local time)."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


@app.get("/historical-demand", response_model=HistoricalResponse, tags=["data"])
def historical_demand(
    service: Service,
    start: datetime | None = None,
    end: datetime | None = None,
    days: Annotated[int, Query(ge=1, le=1200, description="Used when start/end are omitted")] = 30,
    resolution: Literal["hourly", "daily"] = "hourly",
) -> dict:
    """Observed demand with summary statistics (mean, max, min, change vs previous period)."""
    if start and end and start > end:
        raise HTTPException(status_code=422, detail="start must be before end")
    return service.historical(_utc(start), _utc(end), resolution, days)
