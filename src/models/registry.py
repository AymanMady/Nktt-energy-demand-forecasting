"""Model registry: name -> factory. Used by training, tests and the CLI."""

from __future__ import annotations

from collections.abc import Callable

from src.models.base import ForecastModel
from src.models.baseline import NaiveModel, SeasonalNaiveModel, WeeklyNaiveModel
from src.models.lightgbm_model import LightGBMModel
from src.models.statistical.sarima import SARIMAModel
from src.models.xgboost_model import XGBoostModel

MODEL_FACTORIES: dict[str, Callable[[int], ForecastModel]] = {
    "naive": NaiveModel,
    "seasonal_naive": SeasonalNaiveModel,
    "weekly_naive": WeeklyNaiveModel,
    "sarima": SARIMAModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
}

DEFAULT_MODELS: tuple[str, ...] = tuple(MODEL_FACTORIES)


def create_model(key: str, horizon: int) -> ForecastModel:
    try:
        return MODEL_FACTORIES[key](horizon)
    except KeyError as exc:
        raise ValueError(f"Unknown model '{key}'. Available: {sorted(MODEL_FACTORIES)}") from exc
