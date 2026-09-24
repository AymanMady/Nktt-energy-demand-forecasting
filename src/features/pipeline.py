"""Leakage-safe feature engineering for direct multi-horizon forecasting.

ANTI-LEAKAGE RULE
=================
A model with horizon ``H`` predicts demand at target time ``t`` from a forecast
issued at origin ``t - H``. Therefore, for every row with target time ``t``:

* every feature derived from *observed* data (demand, measured weather) must
  only use observations with timestamp ``<= t - H``;
* only deterministic calendar information about ``t`` itself may be used.

This is enforced mechanically: every observed series is ``shift(k)``-ed with
``k >= H`` *before* any rolling operation, so rolling windows end at ``t - H``.
``tests/test_features.py`` checks it by corrupting all values after an origin and
asserting that no feature at or before the corresponding target changes.

Direct strategy
---------------
Instead of feeding predictions back recursively, one model is trained per
horizon bucket (1h, 6h, 24h, 168h). The bucket-``H`` model only sees lags
``>= H``; to forecast lead time ``k`` we use the smallest bucket ``H >= k``.
Lags shorter than ``H`` are simply unavailable to that model, which is what
makes a 7-day forecast honest.

Weather
-------
By default (``use_future_weather=False``) weather enters only through values
observed at or before the origin (last observation, same hour on the most
recent available day, trailing means). ``use_future_weather=True`` adds weather
*at the target time*, i.e. assumes a perfect weather forecast. It is an upper
bound for experiments, not a deployable configuration.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.features.calendar import CALENDAR_FEATURES, calendar_features, local_calendar_flags
from src.ingestion.schema import TARGET_COL, TIMESTAMP_COL

DEMAND_LAGS: tuple[int, ...] = (1, 2, 3, 6, 12, 24, 48, 72, 168, 336)
ROLLING_MEAN_WINDOWS: dict[str, int] = {"6h": 6, "12h": 12, "24h": 24, "48h": 48, "7d": 168}
ROLLING_STATS_WINDOW = 24  # std / max / min over 24h
COMFORT_TEMP_C = 24.0  # cooling threshold assumption (same as dbt var comfort_temp_c)

TARGET = "target"


@dataclass(frozen=True)
class FeatureConfig:
    """Feature switches. The defaults are the configuration used for training."""

    demand_lags: tuple[int, ...] = DEMAND_LAGS
    rolling_mean_windows: dict[str, int] = field(default_factory=lambda: dict(ROLLING_MEAN_WINDOWS))
    rolling_stats_window: int = ROLLING_STATS_WINDOW
    use_weather: bool = True
    use_future_weather: bool = False
    local_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def available_lags(horizon: int, lags: tuple[int, ...] = DEMAND_LAGS) -> list[int]:
    """Demand lags a horizon-``H`` model may use: ``k >= H`` (plus ``H`` itself)."""
    return sorted({k for k in lags if k >= horizon} | {horizon})


def same_hour_lag(horizon: int) -> int:
    """Smallest multiple of 24 that is ``>= horizon``: same hour on the latest known day."""
    return 24 * math.ceil(horizon / 24)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Index by timestamp on a complete hourly grid (gaps become NaN rows)."""
    frame = df.copy()
    if TIMESTAMP_COL in frame.columns:
        frame[TIMESTAMP_COL] = pd.to_datetime(frame[TIMESTAMP_COL], utc=True)
        frame = frame.set_index(TIMESTAMP_COL)
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate timestamps: de-duplicate before building features")
    full = pd.date_range(frame.index.min(), frame.index.max(), freq="h", name=TIMESTAMP_COL)
    return frame.reindex(full)


def _demand_features(demand: pd.Series, horizon: int, config: FeatureConfig) -> pd.DataFrame:
    feats: dict[str, pd.Series] = {}
    for k in available_lags(horizon, config.demand_lags):
        feats[f"demand_lag_{k}"] = demand.shift(k)

    # Rolling windows end at t - H (shift BEFORE rolling -> no leakage).
    past = demand.shift(horizon)
    for name, w in config.rolling_mean_windows.items():
        feats[f"rolling_mean_{name}"] = past.rolling(w, min_periods=max(1, w // 2)).mean()
    w = config.rolling_stats_window
    roll = past.rolling(w, min_periods=w // 2)
    feats[f"rolling_std_{w}h"] = roll.std()
    feats[f"rolling_max_{w}h"] = roll.max()
    feats[f"rolling_min_{w}h"] = roll.min()

    # Same hour on the most recent fully-known day, relative to the weekly mean level.
    sh = same_hour_lag(horizon)
    feats["same_hour_ratio"] = demand.shift(sh) / feats["rolling_mean_7d"]
    return pd.DataFrame(feats)


def _weather_features(frame: pd.DataFrame, horizon: int, config: FeatureConfig) -> pd.DataFrame:
    feats: dict[str, pd.Series] = {}
    temp = frame["temperature_c"]
    cooling = (temp - COMFORT_TEMP_C).clip(lower=0)
    sh = same_hour_lag(horizon)

    # Observed weather, known at the forecast origin (t - H).
    feats["temp_last_obs"] = temp.shift(horizon)
    feats["temp_same_hour_last_day"] = temp.shift(sh)
    feats["temp_mean_24h"] = temp.shift(horizon).rolling(24, min_periods=12).mean()
    feats["temp_max_24h"] = temp.shift(horizon).rolling(24, min_periods=12).max()
    feats["cooling_degree_mean_24h"] = cooling.shift(horizon).rolling(24, min_periods=12).mean()
    feats["cooling_degree_mean_72h"] = cooling.shift(horizon).rolling(72, min_periods=36).mean()
    feats["humidity_same_hour_last_day"] = frame["humidity"].shift(sh)
    feats["humidity_mean_24h"] = frame["humidity"].shift(horizon).rolling(24, min_periods=12).mean()
    feats["wind_speed_same_hour_last_day"] = frame["wind_speed"].shift(sh)
    feats["cloud_cover_mean_24h"] = (
        frame["cloud_cover"].shift(horizon).rolling(24, min_periods=12).mean()
    )

    if config.use_future_weather:
        # ORACLE: weather at the target time (assumes a perfect weather forecast).
        feats["temp_at_target"] = temp
        feats["humidity_at_target"] = frame["humidity"]
        feats["cooling_degree_at_target"] = cooling
    return pd.DataFrame(feats)


def build_features(
    df: pd.DataFrame, horizon: int, config: FeatureConfig | None = None
) -> pd.DataFrame:
    """Build the supervised frame for a horizon-``H`` model.

    Args:
        df: hourly data with ``timestamp`` (column or index), ``demand_mw`` and
            the weather columns. Future rows to forecast may be appended with a
            NaN ``demand_mw``: their features only use observations ``<= t - H``.
        horizon: forecast horizon in hours (>= 1).
        config: feature switches.

    Returns:
        Frame indexed by target timestamp with the feature columns and a
        ``target`` column (= ``demand_mw`` at ``t``, NaN when unknown).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    config = config or FeatureConfig()
    frame = _prepare(df)
    parts = [
        calendar_features(frame.index),
        local_calendar_flags(frame.index, config.local_flags),
        _demand_features(frame[TARGET_COL], horizon, config),
    ]
    if config.use_weather:
        parts.append(_weather_features(frame, horizon, config))
    out = pd.concat(parts, axis=1)
    out[TARGET] = frame[TARGET_COL]
    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """All model input columns of a frame produced by :func:`build_features`."""
    return [c for c in frame.columns if c != TARGET]


def training_rows(
    frame: pd.DataFrame, horizon: int, config: FeatureConfig | None = None
) -> pd.DataFrame:
    """Drop the warm-up period (longest lag not yet available) and rows without a target."""
    config = config or FeatureConfig()
    warmup = max(available_lags(horizon, config.demand_lags)) + max(
        config.rolling_mean_windows.values()
    )
    warm = frame.iloc[warmup:] if len(frame) > warmup else frame.iloc[0:0]
    return warm[warm[TARGET].notna()]


__all__ = [
    "CALENDAR_FEATURES",
    "FeatureConfig",
    "TARGET",
    "available_lags",
    "build_features",
    "feature_columns",
    "same_hour_lag",
    "training_rows",
]
