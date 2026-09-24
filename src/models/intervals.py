"""Prediction intervals from empirical out-of-sample errors.

METHOD (split-conformal style, relative errors)
-----------------------------------------------
1. During cross-validation every model produces *out-of-sample* forecasts.
   For the selected model we compute relative errors ``r = (y - y_hat) / y_hat``.
2. For a nominal coverage ``1 - alpha`` (default 90 %), we keep the empirical
   quantiles ``q_lo = Q(alpha/2)`` and ``q_hi = Q(1 - alpha/2)`` of ``r``,
   separately for each block of hours of the day (errors are larger around the
   evening peak than at night).
3. A new forecast gets ``[y_hat * (1 + q_lo), y_hat * (1 + q_hi)]``.

Relative errors make the band scale with the demand level (demand grows
over time). Quantiles are asymmetric, so a skewed error distribution is kept.

LIMITATIONS
-----------
* Coverage is only approximately guaranteed, and only if future errors behave
  like the calibration errors (exchangeability). Regime changes (new plants,
  tariff changes, outages, extreme heat) break this assumption.
* Intervals are marginal (per hour), not joint over a whole forecast path.
* They do not include uncertainty on future weather beyond what the
  calibration errors already contain.
They are indicative ranges, **not guarantees**. Empirical coverage on the
held-out test period is reported so the reader can judge calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ResidualIntervals:
    coverage: float = 0.90
    hour_block_size: int = 6  # 4 blocks: 0-5h, 6-11h, 12-17h, 18-23h
    min_samples: int = 50
    quantiles_: dict[int, tuple[float, float]] = field(default_factory=dict)
    global_: tuple[float, float] = (0.0, 0.0)
    n_calibration_: int = 0

    @property
    def alpha(self) -> float:
        return 1 - self.coverage

    def _block(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray(index.hour) // self.hour_block_size

    def _q(self, r: np.ndarray) -> tuple[float, float]:
        return float(np.quantile(r, self.alpha / 2)), float(np.quantile(r, 1 - self.alpha / 2))

    def fit(self, y_true: pd.Series, y_pred: pd.Series) -> ResidualIntervals:
        """Calibrate on out-of-sample predictions (index = timestamps)."""
        df = pd.DataFrame({"y": y_true, "p": y_pred}).dropna()
        df = df[df["p"].abs() > 1e-9]
        if len(df) < self.min_samples:
            raise ValueError(f"Need >= {self.min_samples} calibration points, got {len(df)}")
        r = ((df["y"] - df["p"]) / df["p"]).to_numpy()
        self.global_ = self._q(r)
        blocks = self._block(pd.DatetimeIndex(df.index))
        self.quantiles_ = {
            int(b): self._q(r[blocks == b])
            for b in np.unique(blocks)
            if (blocks == b).sum() >= self.min_samples
        }
        self.n_calibration_ = len(df)
        return self

    def predict(self, y_pred: np.ndarray, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) bounds for point forecasts at ``index``."""
        y_pred = np.asarray(y_pred, dtype=float)
        blocks = self._block(pd.DatetimeIndex(index))
        lo = np.array([self.quantiles_.get(int(b), self.global_)[0] for b in blocks])
        hi = np.array([self.quantiles_.get(int(b), self.global_)[1] for b in blocks])
        return y_pred * (1 + lo), y_pred * (1 + hi)

    def to_dict(self) -> dict:
        return {
            "method": "empirical quantiles of relative out-of-sample errors, by 6-hour block",
            "coverage": self.coverage,
            "n_calibration": self.n_calibration_,
            "global_quantiles": self.global_,
            "block_quantiles": {str(k): v for k, v in self.quantiles_.items()},
        }


def interval_metrics(y_true, lower, upper) -> dict[str, float]:
    """Empirical coverage and mean width of prediction intervals."""
    y = np.asarray(y_true, dtype=float)
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    y, lo, hi = y[mask], lo[mask], hi[mask]
    if not y.size:
        return {"coverage": float("nan"), "mean_width_mw": float("nan"), "n": 0}
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width_mw": float(np.mean(hi - lo)),
        "n": int(y.size),
    }
