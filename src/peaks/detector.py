"""Experimental high-demand (peak) detection on top of forecasts.

THRESHOLD
---------
The peak threshold is a percentile (default P95) of *observed* demand over a
trailing window (default 365 days) ending at the forecast origin. A trailing
window is used because demand grows over time: a percentile over the whole
history would flag almost every recent hour.

This threshold is a statistical convention chosen for the prototype. It is
**not** an official SOMELEC (or any operator) rule; a real deployment would use
operational limits (available capacity, reserve margins) defined by the operator.

RISK LEVELS
-----------
* HIGH   - the point forecast is at or above the threshold;
* MEDIUM - the point forecast is below, but the upper bound of the prediction
           interval reaches the threshold;
* LOW    - the whole interval is below the threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")


@dataclass(frozen=True)
class PeakConfig:
    percentile: float = 0.95
    window_days: int = 365

    def to_dict(self) -> dict:
        return asdict(self)


def compute_threshold(
    history: pd.Series, as_of: pd.Timestamp | None = None, config: PeakConfig | None = None
) -> float:
    """Percentile of observed demand over the trailing window ending at ``as_of``."""
    config = config or PeakConfig()
    s = history.dropna().sort_index()
    if s.empty:
        raise ValueError("Empty demand history")
    as_of = as_of or s.index.max()
    window = s.loc[(s.index > as_of - pd.Timedelta(days=config.window_days)) & (s.index <= as_of)]
    if window.empty:
        raise ValueError("No observations in the trailing window")
    return float(window.quantile(config.percentile))


def classify_risk(forecast, upper, threshold: float) -> np.ndarray:
    """Vectorised LOW / MEDIUM / HIGH classification."""
    f = np.asarray(forecast, dtype=float)
    u = np.asarray(upper, dtype=float)
    return np.where(f >= threshold, "HIGH", np.where(u >= threshold, "MEDIUM", "LOW"))


def detect_peaks(forecast: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Annotate a forecast frame (timestamp, prediction_mw, upper_mw) with peak risk.

    Returns columns: timestamp, date, hour, predicted_demand_mw, threshold_mw, peak_risk.
    """
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(forecast["timestamp"], utc=True),
            "predicted_demand_mw": forecast["prediction_mw"].to_numpy(dtype=float),
            "upper_mw": forecast["upper_mw"].to_numpy(dtype=float),
        }
    )
    out["date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    out["hour"] = out["timestamp"].dt.hour
    out["threshold_mw"] = threshold
    out["peak_risk"] = classify_risk(out["predicted_demand_mw"], out["upper_mw"], threshold)
    return out[["timestamp", "date", "hour", "predicted_demand_mw", "threshold_mw", "peak_risk"]]


def peak_periods(
    annotated: pd.DataFrame, levels: tuple[str, ...] = ("HIGH", "MEDIUM")
) -> pd.DataFrame:
    """Group consecutive hours at the given risk levels into periods."""
    df = annotated.sort_values("timestamp").reset_index(drop=True)
    flag = df["peak_risk"].isin(levels)
    if not flag.any():
        return pd.DataFrame(columns=["start", "end", "hours", "max_predicted_mw", "max_risk"])
    new_group = flag & ~(flag.shift(fill_value=False)) | (
        df["timestamp"].diff() != pd.Timedelta(hours=1)
    )
    group_id = new_group.cumsum()
    rank = {lvl: i for i, lvl in enumerate(RISK_LEVELS)}
    rows = []
    for _, g in df[flag].groupby(group_id[flag]):
        rows.append(
            {
                "start": g["timestamp"].iloc[0],
                "end": g["timestamp"].iloc[-1],
                "hours": len(g),
                "max_predicted_mw": float(g["predicted_demand_mw"].max()),
                "max_risk": max(g["peak_risk"], key=rank.__getitem__),
            }
        )
    return pd.DataFrame(rows)


def evaluate_peak_detection(actual, forecast, upper, threshold: float) -> dict[str, float]:
    """Precision / recall of HIGH (and HIGH+MEDIUM) alerts against actual exceedances."""
    y = np.asarray(actual, dtype=float)
    risk = classify_risk(forecast, upper, threshold)
    actual_peak = y >= threshold
    out: dict[str, float] = {"threshold_mw": threshold, "actual_peak_hours": int(actual_peak.sum())}
    for name, alert in {
        "high": risk == "HIGH",
        "high_or_medium": np.isin(risk, ["HIGH", "MEDIUM"]),
    }.items():
        tp = int((alert & actual_peak).sum())
        precision = tp / alert.sum() if alert.sum() else float("nan")
        recall = tp / actual_peak.sum() if actual_peak.sum() else float("nan")
        out[f"{name}_alerts"] = int(alert.sum())
        out[f"{name}_precision"] = float(precision)
        out[f"{name}_recall"] = float(recall)
    return out
