"""Reference (naive) forecasts. Every other model must beat these.

For a horizon-``H`` forecast issued at ``t - H`` the most recent usable
observation of a season ``s`` is ``demand(t - s * ceil(H / s))``:

* Naive (s = 1):                  prediction(t) = demand(t - H)      [= demand(t-1) for H = 1]
* Seasonal naive (s = 24):        prediction(t) = demand(t - 24)     for H <= 24
* Weekly seasonal naive (s = 168): prediction(t) = demand(t - 168)   for H <= 168
"""

from __future__ import annotations

import math
from typing import ClassVar, Self

import numpy as np
import pandas as pd

from src.models.base import ForecastModel


class SeasonalNaiveModel(ForecastModel):
    """prediction(t) = demand(t - season * ceil(H / season))."""

    family: ClassVar[str] = "baseline"
    season: ClassVar[int] = 24
    name: ClassVar[str] = "Seasonal Naive"

    @property
    def lag(self) -> int:
        return self.season * math.ceil(self.horizon / self.season)

    @property
    def lag_column(self) -> str:
        return f"demand_lag_{self.lag}"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:  # noqa: ARG002 - nothing to learn
        if self.lag_column not in X.columns:
            raise ValueError(f"{self.name} needs feature '{self.lag_column}'")
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, history: pd.Series | None = None) -> np.ndarray:
        self._check_fitted()
        return X[self.lag_column].to_numpy(dtype=float)

    def get_params(self) -> dict:
        return {"horizon": self.horizon, "season": self.season, "lag": self.lag}


class NaiveModel(SeasonalNaiveModel):
    """Last observed value: prediction(t) = demand(t - H)."""

    season: ClassVar[int] = 1
    name: ClassVar[str] = "Naive"


class WeeklyNaiveModel(SeasonalNaiveModel):
    """Same hour, one week earlier: prediction(t) = demand(t - 168)."""

    season: ClassVar[int] = 168
    name: ClassVar[str] = "Weekly Seasonal Naive"
