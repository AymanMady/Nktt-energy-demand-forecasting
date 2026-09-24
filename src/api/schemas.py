"""Pydantic request/response models (they also drive the OpenAPI documentation)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_HORIZON_H = 168
DISCLAIMER = (
    "Prototype trained on synthetic data. Not SOMELEC data; forecasts and peak alerts are "
    "experimental and must not be used for operational decisions."
)

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class Metrics(BaseModel):
    mae: float = Field(description="Mean absolute error (MW)")
    rmse: float = Field(description="Root mean squared error (MW)")
    mape: float = Field(description="Mean absolute percentage error (%)")
    smape: float | None = Field(None, description="Symmetric MAPE (%)")
    bias: float | None = Field(None, description="Mean error, forecast - actual (MW)")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    database: Literal["ok", "unavailable"]
    data_end: datetime | None = None
    model_trained_at: str | None = None
    data_is_synthetic: bool = True


class ForecastRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"horizon": 24}]})

    horizon: int = Field(24, ge=1, le=MAX_HORIZON_H, description="Hours to forecast (1-168)")
    include_history_hours: int = Field(
        0,
        ge=0,
        le=24 * 30,
        description="Also return this many hours of observed demand before the origin",
    )


class ForecastPoint(BaseModel):
    timestamp: datetime
    lead_h: int = Field(description="Hours after the forecast origin")
    prediction_mw: float
    lower_mw: float = Field(description="Lower bound of the prediction interval")
    upper_mw: float = Field(description="Upper bound of the prediction interval")
    model: str = Field(description="Model that produced this lead time")
    bucket_h: int = Field(description="Horizon bucket of that model")
    peak_risk: RiskLevel


class DemandPoint(BaseModel):
    timestamp: datetime
    demand_mw: float | None
    temperature_c: float | None = None


class PeakPeriod(BaseModel):
    start: datetime
    end: datetime
    hours: int
    max_predicted_mw: float
    max_risk: RiskLevel


class ForecastResponse(BaseModel):
    origin: datetime = Field(description="Last observed hour; forecasts start one hour later")
    horizon: int
    generated_at: datetime
    interval_coverage: float = Field(description="Nominal coverage of the prediction intervals")
    peak_threshold_mw: float = Field(
        description="Experimental peak threshold (trailing percentile)"
    )
    forecast: list[ForecastPoint]
    history: list[DemandPoint] = []
    peak_periods: list[PeakPeriod] = []
    model_trained_at: str | None = None
    dataset_version: str | None = None
    disclaimer: str = DISCLAIMER


class CandidateModel(BaseModel):
    model: str
    cv: Metrics
    test: Metrics


class HorizonModels(BaseModel):
    horizon_h: int
    selected_model: str
    candidates: list[CandidateModel]
    interval_coverage_test: float | None = None
    interval_mean_width_mw: float | None = None


class ModelsResponse(BaseModel):
    trained_at: str
    dataset: dict
    splits: dict
    horizons: list[HorizonModels]
    mlflow: dict | None = None
    disclaimer: str = DISCLAIMER


class HorizonMetrics(BaseModel):
    horizon_h: int
    selected_model: str
    test: Metrics
    cv: Metrics
    best_baseline: str
    best_baseline_test: Metrics
    interval_coverage_test: float
    interval_mean_width_mw: float
    peak_alerts: dict


class MetricsResponse(BaseModel):
    trained_at: str
    dataset_version: str
    data_is_synthetic: bool
    horizons: list[HorizonMetrics]


class HistoricalSummary(BaseModel):
    n: int
    mean_mw: float | None
    max_mw: float | None
    max_timestamp: datetime | None
    min_mw: float | None
    min_timestamp: datetime | None
    change_vs_previous_period_pct: float | None = Field(
        None, description="Mean demand vs the same-length period just before"
    )


class HistoricalResponse(BaseModel):
    start: datetime
    end: datetime
    resolution: Literal["hourly", "daily"]
    points: list[DemandPoint]
    summary: HistoricalSummary
    data_is_synthetic: bool = True


class BacktestPoint(BaseModel):
    timestamp: datetime
    actual_mw: float
    prediction_mw: float
    lower_mw: float | None
    upper_mw: float | None


class BacktestResponse(BaseModel):
    horizon_h: int
    model: str
    metrics: Metrics
    points: list[BacktestPoint]
