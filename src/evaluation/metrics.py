"""Forecast accuracy metrics.

* MAE   - mean absolute error (MW). Primary selection metric: robust, in business units.
* RMSE  - root mean squared error (MW). Penalises large misses (peaks).
* MAPE  - mean absolute percentage error (%). Readable, but undefined at zero demand
          and asymmetric; rows with |actual| < ``eps`` are excluded.
* sMAPE - symmetric MAPE (%), bounded, less sensitive to small actuals.
* Bias  - mean error (forecast - actual, MW). Positive = over-forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_NAMES: tuple[str, ...] = ("mae", "rmse", "mape", "smape", "bias")


def _clean(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def mae(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp))) if yt.size else float("nan")


def rmse(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2))) if yt.size else float("nan")


def mape(y_true, y_pred, eps: float = 1e-6) -> float:
    yt, yp = _clean(y_true, y_pred)
    keep = np.abs(yt) > eps
    if not keep.any():
        return float("nan")
    return float(np.mean(np.abs((yt[keep] - yp[keep]) / yt[keep])) * 100)


def smape(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    denom = np.abs(yt) + np.abs(yp)
    keep = denom > 0
    if not keep.any():
        return float("nan")
    return float(np.mean(2 * np.abs(yp[keep] - yt[keep]) / denom[keep]) * 100)


def bias(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.mean(yp - yt)) if yt.size else float("nan")


def evaluate_model(y_true, y_pred) -> dict[str, float]:
    """Return every metric plus the number of evaluated points ``n``."""
    yt, _ = _clean(y_true, y_pred)
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "n": int(yt.size),
    }


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """``{model: metrics}`` -> table sorted by MAE (Model | MAE | RMSE | MAPE | sMAPE | Bias)."""
    df = pd.DataFrame(results).T[list(METRIC_NAMES)]
    df.index.name = "model"
    return df.sort_values("mae")
