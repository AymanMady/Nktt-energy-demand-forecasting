"""Shared logic for gradient-boosted tree forecasters (XGBoost, LightGBM).

Target normalisation
--------------------
Tree ensembles cannot extrapolate beyond the target values seen in training,
and demand grows year over year. With ``target_transform="ratio"`` (default),
the model learns ``demand(t) / rolling_mean_7d`` (the trailing weekly mean
known at the forecast origin) and the prediction is scaled back. The level is
carried by the leakage-safe rolling mean; the trees learn the shape.
"""

from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any, ClassVar, Literal, Self

import numpy as np
import pandas as pd

from src.models.base import ForecastModel

SCALE_COLUMN = "rolling_mean_7d"
# OpenMP threads per model. A moderate default avoids oversubscription on shared
# machines (busy OpenMP threads slow down sharply under CPU contention).
N_JOBS = int(os.getenv("MODEL_N_JOBS", "4"))


class GradientBoostingForecaster(ForecastModel):
    family: ClassVar[str] = "ml"
    default_params: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        horizon: int,
        params: dict[str, Any] | None = None,
        target_transform: Literal["ratio", "none"] = "ratio",
        random_state: int = 42,
    ) -> None:
        super().__init__(horizon)
        self.params = {**self.default_params, **(params or {})}
        self.target_transform = target_transform
        self.random_state = random_state
        self.feature_names_: list[str] = []
        self.estimator_: Any = None
        self._fallback_scale = 1.0

    @abstractmethod
    def _make_estimator(self) -> Any: ...

    def _scale(self, X: pd.DataFrame) -> np.ndarray:
        if self.target_transform == "none":
            return np.ones(len(X))
        if SCALE_COLUMN not in X.columns:
            raise ValueError(f"target_transform='ratio' requires feature '{SCALE_COLUMN}'")
        return X[SCALE_COLUMN].fillna(self._fallback_scale).to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        mask = y.notna().to_numpy()
        X, y = X.loc[mask], y.loc[mask]
        self.feature_names_ = list(X.columns)
        if self.target_transform == "ratio":
            self._fallback_scale = float(y.mean())
        target = y.to_numpy(dtype=float) / self._scale(X)
        self.estimator_ = self._make_estimator()
        self.estimator_.fit(X, target)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, history: pd.Series | None = None) -> np.ndarray:
        self._check_fitted()
        X = X[self.feature_names_]
        return np.asarray(self.estimator_.predict(X), dtype=float) * self._scale(X)

    def feature_importance(self) -> pd.Series:
        self._check_fitted()
        imp = np.asarray(self.estimator_.feature_importances_, dtype=float)
        total = imp.sum() or 1.0
        return pd.Series(imp / total, index=self.feature_names_).sort_values(ascending=False)

    def get_params(self) -> dict:
        return {
            "horizon": self.horizon,
            "target_transform": self.target_transform,
            "n_features": len(self.feature_names_),
            **self.params,
        }
