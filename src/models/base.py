"""Common interface for every forecasting model (baselines, statistical, ML).

All models are *direct* horizon-``H`` forecasters:

* ``fit(X, y)``: ``X`` is the feature frame from
  :func:`src.features.pipeline.build_features` (indexed by target time) and
  ``y`` the observed demand at those times.
* ``predict(X, history)``: predict demand at each target time of ``X.index``
  from information available ``H`` hours earlier. ``history`` is the observed
  demand series; models that need raw history (SARIMA) must only use values at
  or before ``t - H``. Feature-based models ignore it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Self

import numpy as np
import pandas as pd


class ForecastModel(ABC):
    """Abstract horizon-``H`` forecaster."""

    name: ClassVar[str] = "base"
    family: ClassVar[str] = "base"  # baseline | statistical | ml

    def __init__(self, horizon: int) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = horizon
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Fit on training rows (index = target timestamps)."""

    @abstractmethod
    def predict(self, X: pd.DataFrame, history: pd.Series | None = None) -> np.ndarray:
        """Predict demand (MW) for every row of ``X``."""

    def get_params(self) -> dict:
        """Hyper-parameters to log (MLflow)."""
        return {"horizon": self.horizon}

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} (h={self.horizon}) is not fitted")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(horizon={self.horizon})"
