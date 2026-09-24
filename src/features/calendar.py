"""Calendar features and local (Mauritanian) calendar flags.

Calendar features are deterministic functions of the *target* timestamp: they
are known in advance for any future hour, so they can never leak information.
Nouakchott is UTC+0 without daylight saving, so UTC hour = local hour.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

CALENDAR_FEATURES: tuple[str, ...] = (
    "hour",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
    # Local context
    "is_friday",
    "is_friday_prayer",
    # Cyclical encodings (help models see that 23h is next to 0h, Dec next to Jan)
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
)

# Flags that exist in the local calendar but are OFF by default: the synthetic data
# does not simulate these effects and the Islamic-calendar dates are approximate.
# Enable them (FeatureConfig.local_flags) once verified calendars and real data exist.
LOCAL_CALENDAR_FLAGS: tuple[str, ...] = ("is_holiday", "is_ramadan", "is_eid", "is_school_holiday")
LOCAL_CALENDAR_PATH = PROJECT_ROOT / "dbt" / "seeds" / "local_calendar.csv"

WEEKEND_DAYS = (5, 6)  # Saturday, Sunday (Mauritania)
FRIDAY = 4
FRIDAY_PRAYER_HOURS = (12, 13, 14)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return calendar features for each timestamp of ``index`` (UTC)."""
    idx = index.tz_convert("UTC") if index.tz is not None else index
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()
    out = pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": dow,
            "day_of_month": idx.day.to_numpy(),
            "week_of_year": idx.isocalendar().week.to_numpy().astype(int),
            "month": idx.month.to_numpy(),
            "quarter": idx.quarter.to_numpy(),
            "is_weekend": np.isin(dow, WEEKEND_DAYS).astype(int),
            "is_friday": (dow == FRIDAY).astype(int),
            "is_friday_prayer": ((dow == FRIDAY) & np.isin(hour, FRIDAY_PRAYER_HOURS)).astype(int),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=index,
    )
    return out


@lru_cache(maxsize=4)
def load_local_calendar(path: Path = LOCAL_CALENDAR_PATH) -> pd.DataFrame:
    """Load the local calendar (same CSV as the dbt seed: single source of truth)."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index(df["date"].dt.date)


def local_calendar_flags(
    index: pd.DatetimeIndex, flags: tuple[str, ...], path: Path = LOCAL_CALENDAR_PATH
) -> pd.DataFrame:
    """Return the requested local-calendar flags (0/1) for each timestamp."""
    unknown = set(flags) - set(LOCAL_CALENDAR_FLAGS)
    if unknown:
        raise ValueError(f"Unknown local calendar flags: {sorted(unknown)}")
    out = pd.DataFrame(index=index)
    if not flags:
        return out
    cal = load_local_calendar(path)
    dates = pd.Index(index.date)
    for flag in flags:
        out[flag] = dates.map(cal[flag].to_dict()).fillna(False).astype(int).to_numpy()
    return out
