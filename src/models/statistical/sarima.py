"""Seasonal ARIMA with weekly Fourier regressors (statsmodels SARIMAX).

Model
-----
``demand(t) = x(t)' beta + u(t)`` where ``u`` follows
SARIMA(p,d,q)(P,D,Q)_24 (daily seasonality) and ``x(t)`` are deterministic
sine/cosine terms of the hour-of-week (weekly seasonality). The regressors are
pure calendar functions, known in advance, so they introduce no leakage.

Fitting on the full multi-year hourly history is too slow with a 24-period
seasonal state, so the model is fitted on a trailing window (default 6 weeks)
before the forecast period; parameters are then kept fixed. The default
SARIMA(1,0,1)(0,1,1)_24 had the same AIC as heavier specifications
((2,0,1)(1,1,1)_24) on this data, at a fraction of the fitting cost.

Direct H-step predictions for every target time
-----------------------------------------------
For state-space models the H-step-ahead forecast from origin ``o`` is
``y_hat(o+H) = d(o+H) + Z T^H a(o|o)``, where ``a(o|o)`` is the Kalman
*filtered* state (uses observations up to ``o`` only), ``T`` the transition
matrix, ``Z`` the design vector and ``d`` the observation intercept (here the
Fourier regression term). Filtering the whole series once and applying ``T^H``
to every filtered state gives exact H-step forecasts for all origins at once,
causally, without refitting. A unit test checks this against statsmodels'
own ``forecast()``.
"""

from __future__ import annotations

import warnings
from typing import ClassVar, Self

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from threadpoolctl import threadpool_limits

from src.models.base import ForecastModel
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Fitted results are shared across horizons: the SARIMA fit does not depend on H.
_FIT_CACHE: dict[tuple, tuple[np.ndarray, float]] = {}


def weekly_fourier(index: pd.DatetimeIndex, order: int) -> pd.DataFrame:
    """Sine/cosine terms of the hour of week (period 168 h)."""
    how = (index.dayofweek * 24 + index.hour).to_numpy()
    cols = {}
    for k in range(1, order + 1):
        cols[f"wk_sin_{k}"] = np.sin(2 * np.pi * k * how / 168)
        cols[f"wk_cos_{k}"] = np.cos(2 * np.pi * k * how / 168)
    return pd.DataFrame(cols, index=index)


class SARIMAModel(ForecastModel):
    name: ClassVar[str] = "SARIMA"
    family: ClassVar[str] = "statistical"

    def __init__(
        self,
        horizon: int,
        order: tuple[int, int, int] = (1, 0, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 24),
        fourier_order: int = 3,
        fit_window_hours: int = 24 * 7 * 6,
        maxiter: int = 50,
    ) -> None:
        super().__init__(horizon)
        self.order = order
        self.seasonal_order = seasonal_order
        self.fourier_order = fourier_order
        self.fit_window_hours = fit_window_hours
        self.maxiter = maxiter
        self.params_: np.ndarray | None = None
        self.aic_: float | None = None
        self.fit_start_: pd.Timestamp | None = None
        self.fit_end_: pd.Timestamp | None = None

    # ------------------------------------------------------------------
    def _exog(self, index: pd.DatetimeIndex) -> pd.DataFrame | None:
        return weekly_fourier(index, self.fourier_order) if self.fourier_order > 0 else None

    def _make_model(self, endog: pd.Series, index: pd.DatetimeIndex) -> SARIMAX:
        return SARIMAX(
            endog,
            exog=self._exog(index),
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend="n",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:  # noqa: ARG002 - uses y only
        y = y.sort_index()
        end = y.index.max()
        start = end - pd.Timedelta(hours=self.fit_window_hours - 1)
        index = pd.date_range(start, end, freq="h")
        endog = y.reindex(index)
        key = (
            start,
            end,
            float(np.nansum(endog)),
            self.order,
            self.seasonal_order,
            self.fourier_order,
        )
        if key not in _FIT_CACHE:
            # Small state matrices: multi-threaded BLAS only adds contention.
            with warnings.catch_warnings(), threadpool_limits(1):
                warnings.simplefilter("ignore")
                result = self._make_model(endog, index).fit(disp=False, maxiter=self.maxiter)
            # Keep only the estimated parameters: the full results object stores every
            # Kalman state matrix and would make the persisted model hundreds of MB.
            _FIT_CACHE[key] = (result.params.to_numpy(), float(result.aic))
            logger.info("SARIMA fitted on %s -> %s (AIC %.0f)", start, end, result.aic)
        self.params_, self.aic_ = _FIT_CACHE[key]
        self.fit_start_, self.fit_end_ = start, end
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, history: pd.Series | None = None) -> np.ndarray:
        """Exact H-step-ahead forecasts for every target time in ``X.index``."""
        self._check_fitted()
        if history is None:
            raise ValueError("SARIMA needs the observed demand history to forecast")
        targets = pd.DatetimeIndex(X.index)
        h = pd.Timedelta(hours=self.horizon)
        if (targets - h).min() < self.fit_start_:
            raise ValueError("Forecast origins precede the SARIMA fitting window")

        segment = pd.date_range(self.fit_start_, targets.max(), freq="h")
        endog = history.reindex(segment)
        # Causality guard: nothing after the last origin is ever needed.
        endog[segment > targets.max() - h] = np.nan
        with warnings.catch_warnings(), threadpool_limits(1):
            warnings.simplefilter("ignore")
            applied = self._make_model(endog, segment).filter(self.params_)
        fr = applied.filter_results

        T = fr.transition[:, :, 0]
        Z = fr.design[:, :, 0]
        c = fr.state_intercept[:, 0]
        TH = np.linalg.matrix_power(T, self.horizon)
        # Accumulated state intercept over H steps (zero for trend="n", kept for generality).
        c_acc = sum(np.linalg.matrix_power(T, j) @ c for j in range(self.horizon))

        origin_pos = segment.get_indexer(targets - h)
        target_pos = segment.get_indexer(targets)
        states = fr.filtered_state[:, origin_pos]
        preds = (Z @ (TH @ states + c_acc[:, None])).ravel()
        d = fr.obs_intercept
        preds += d[0, target_pos] if d.shape[1] > 1 else d[0, 0]
        return preds

    def get_params(self) -> dict:
        return {
            "horizon": self.horizon,
            "order": str(self.order),
            "seasonal_order": str(self.seasonal_order),
            "fourier_order": self.fourier_order,
            "fit_window_hours": self.fit_window_hours,
        }
