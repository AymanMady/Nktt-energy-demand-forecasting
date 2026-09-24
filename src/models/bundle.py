"""Production forecaster: one selected model per horizon bucket + intervals.

Multi-step forecasting with direct models
-----------------------------------------
To forecast lead times ``1..N`` from origin ``T``, lead ``k`` is predicted by
the model of the smallest trained bucket ``H >= k`` (e.g. buckets 1/6/24/168:
lead 1 -> H1, leads 2-6 -> H6, 7-24 -> H24, 25-168 -> H168). The bucket-``H``
model only uses observations ``<= t - H <= T``, so every lead is a genuine
out-of-sample forecast from ``T``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.pipeline import FeatureConfig, build_features, feature_columns
from src.ingestion.schema import TARGET_COL, TIMESTAMP_COL
from src.models.base import ForecastModel
from src.models.intervals import ResidualIntervals
from src.peaks.detector import PeakConfig, classify_risk, compute_threshold

BUNDLE_FILENAME = "forecaster.joblib"


@dataclass
class ForecastBundle:
    models: dict[int, ForecastModel]
    intervals: dict[int, ResidualIntervals]
    feature_config: FeatureConfig
    peak_config: PeakConfig = field(default_factory=PeakConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def horizons(self) -> list[int]:
        return sorted(self.models)

    @property
    def max_horizon(self) -> int:
        return max(self.models)

    def bucket_for(self, lead: int) -> int:
        """Smallest trained horizon bucket covering lead time ``lead``."""
        for h in self.horizons:
            if h >= lead:
                return h
        raise ValueError(f"Lead time {lead}h exceeds the maximum horizon {self.max_horizon}h")

    def forecast(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Forecast the ``horizon`` hours following the last observed demand.

        Args:
            history: canonical hourly frame (timestamp, demand_mw, weather...),
                containing at least ~3 weeks of data before the origin.
            horizon: number of hours to forecast (1 .. max_horizon).

        Returns:
            Frame with timestamp, lead_h, prediction_mw, lower_mw, upper_mw,
            model, bucket_h, peak_risk, threshold_mw.
        """
        if not 1 <= horizon <= self.max_horizon:
            raise ValueError(f"horizon must be in [1, {self.max_horizon}]")
        hist = history.copy()
        hist[TIMESTAMP_COL] = pd.to_datetime(hist[TIMESTAMP_COL], utc=True)
        hist = hist.sort_values(TIMESTAMP_COL)
        observed = hist.dropna(subset=[TARGET_COL])
        origin = observed[TIMESTAMP_COL].max()
        future_index = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon, freq="h")
        future = pd.DataFrame({TIMESTAMP_COL: future_index})
        frame_input = pd.concat([hist[hist[TIMESTAMP_COL] <= origin], future], ignore_index=True)
        demand_history = observed.set_index(TIMESTAMP_COL)[TARGET_COL]

        leads = np.arange(1, horizon + 1)
        buckets = np.array([self.bucket_for(k) for k in leads])
        parts = []
        for h in np.unique(buckets):
            idx = future_index[buckets == h]
            frame = build_features(frame_input, int(h), self.feature_config)
            X = frame.loc[idx, feature_columns(frame)]
            model = self.models[int(h)]
            pred = model.predict(X, history=demand_history)
            lower, upper = self.intervals[int(h)].predict(pred, idx)
            parts.append(
                pd.DataFrame(
                    {
                        "timestamp": idx,
                        "prediction_mw": pred,
                        "lower_mw": lower,
                        "upper_mw": upper,
                        "model": model.name,
                        "bucket_h": int(h),
                    }
                )
            )
        out = pd.concat(parts).sort_values("timestamp").reset_index(drop=True)
        out.insert(1, "lead_h", leads)
        threshold = compute_threshold(demand_history, origin, self.peak_config)
        out["threshold_mw"] = threshold
        out["peak_risk"] = classify_risk(out["prediction_mw"], out["upper_mw"], threshold)
        return out

    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / BUNDLE_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> ForecastBundle:
        p = Path(path)
        if p.is_dir():
            p = p / BUNDLE_FILENAME
        obj = joblib.load(p)
        if not isinstance(obj, ForecastBundle):
            raise TypeError(f"{p} does not contain a ForecastBundle")
        return obj
