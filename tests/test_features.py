"""Tests for feature engineering, including the anti-leakage guarantee."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.calendar import calendar_features, local_calendar_flags
from src.features.pipeline import (
    TARGET,
    FeatureConfig,
    available_lags,
    build_features,
    feature_columns,
    same_hour_lag,
    training_rows,
)

HORIZONS = [1, 6, 24, 168]


@pytest.fixture(scope="module")
def data(small_dataset) -> pd.DataFrame:
    return small_dataset.data.copy()


def _corrupt_after(df: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Replace every observation strictly after ``origin`` by garbage."""
    bad = df.copy()
    mask = bad["timestamp"] > origin
    rng = np.random.default_rng(0)
    for col in ["demand_mw", "temperature_c", "humidity", "wind_speed", "cloud_cover"]:
        bad.loc[mask, col] = rng.uniform(0, 1000, mask.sum())
    return bad


@pytest.mark.parametrize("horizon", HORIZONS)
def test_no_leakage_from_observations_after_origin(data, horizon):
    """Features for any target t <= origin + H must not depend on data after the origin."""
    origin = data["timestamp"].iloc[len(data) // 2]
    clean = build_features(data, horizon)
    corrupted = build_features(_corrupt_after(data, origin), horizon)
    cols = feature_columns(clean)
    visible = clean.index <= origin + pd.Timedelta(hours=horizon)
    pd.testing.assert_frame_equal(clean.loc[visible, cols], corrupted.loc[visible, cols])
    # Sanity: targets after origin did change, so the corruption was effective.
    assert not clean.loc[clean.index > origin, TARGET].equals(
        corrupted.loc[corrupted.index > origin, TARGET]
    )


def test_leakage_test_detects_oracle_weather(data):
    """The leakage test must be sensitive: oracle (future) weather features DO change."""
    origin = data["timestamp"].iloc[len(data) // 2]
    cfg = FeatureConfig(use_future_weather=True)
    clean = build_features(data, 24, cfg)
    corrupted = build_features(_corrupt_after(data, origin), 24, cfg)
    window = (clean.index > origin) & (clean.index <= origin + pd.Timedelta(hours=24))
    assert not np.allclose(
        clean.loc[window, "temp_at_target"], corrupted.loc[window, "temp_at_target"]
    )


@pytest.mark.parametrize("horizon", HORIZONS)
def test_lag_values_and_availability(data, horizon):
    frame = build_features(data, horizon)
    lags = available_lags(horizon)
    assert min(lags) == horizon
    for k in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        assert (f"demand_lag_{k}" in frame.columns) == (k >= horizon)
    s = data.set_index("timestamp")["demand_mw"]
    t = frame.index[400]
    for k in lags:
        assert frame.loc[t, f"demand_lag_{k}"] == s.loc[t - pd.Timedelta(hours=k)]


def test_rolling_windows_end_at_origin(data):
    frame = build_features(data, 6)
    s = data.set_index("timestamp")["demand_mw"]
    t = frame.index[500]
    window = s.loc[t - pd.Timedelta(hours=6 + 23) : t - pd.Timedelta(hours=6)]
    assert len(window) == 24
    assert frame.loc[t, "rolling_mean_24h"] == pytest.approx(window.mean())
    assert frame.loc[t, "rolling_max_24h"] == pytest.approx(window.max())
    assert frame.loc[t, "rolling_min_24h"] == pytest.approx(window.min())
    assert frame.loc[t, "rolling_std_24h"] == pytest.approx(window.std())


def test_required_feature_names_present(data):
    frame = build_features(data, 1)
    expected = {
        "hour", "day_of_week", "day_of_month", "week_of_year", "month", "quarter", "is_weekend",
        "demand_lag_1", "demand_lag_2", "demand_lag_3", "demand_lag_6", "demand_lag_12",
        "demand_lag_24", "demand_lag_48", "demand_lag_72", "demand_lag_168",
        "rolling_mean_6h", "rolling_mean_12h", "rolling_mean_24h", "rolling_mean_48h",
        "rolling_mean_7d", "rolling_std_24h", "rolling_max_24h", "rolling_min_24h",
    }  # fmt: skip
    assert expected <= set(frame.columns)
    assert "temp_at_target" not in frame.columns  # no oracle weather by default


def test_future_rows_get_features(data):
    """Appending future hours with unknown demand yields usable features (API use case)."""
    last = data["timestamp"].max()
    future = pd.DataFrame({"timestamp": pd.date_range(last, periods=25, freq="h")[1:]})
    frame = build_features(pd.concat([data, future], ignore_index=True), 24)
    fut = frame.loc[frame.index > last]
    assert len(fut) == 24
    assert fut[TARGET].isna().all()
    assert fut["demand_lag_24"].notna().all()
    assert fut["rolling_mean_7d"].notna().all()


def test_gaps_are_reindexed_not_shifted(data):
    """A missing hour must not make lag-k point to the wrong timestamp."""
    gappy = data.drop(index=[300, 301, 302])
    frame = build_features(gappy, 1)
    s = data.set_index("timestamp")["demand_mw"]
    t = data["timestamp"].iloc[330]
    assert frame.loc[t, "demand_lag_24"] == s.loc[t - pd.Timedelta(hours=24)]
    assert np.isnan(frame.loc[data["timestamp"].iloc[301], TARGET])


def test_training_rows_drop_warmup_and_missing_targets(data):
    frame = build_features(data, 1)
    rows = training_rows(frame, 1)
    assert rows[TARGET].notna().all()
    assert rows["demand_lag_336"].notna().all()
    assert len(rows) == len(frame) - (336 + 168)


def test_duplicate_timestamps_rejected(data):
    with pytest.raises(ValueError, match="Duplicate"):
        build_features(pd.concat([data, data.iloc[[3]]]), 1)


def test_calendar_features():
    idx = pd.DatetimeIndex(["2026-09-25 13:00", "2026-09-26 20:00"], tz="UTC")  # Fri, Sat
    cal = calendar_features(idx)
    assert cal["day_of_week"].tolist() == [4, 5]
    assert cal["is_weekend"].tolist() == [0, 1]
    assert cal["is_friday_prayer"].tolist() == [1, 0]
    assert cal["quarter"].tolist() == [3, 3]


def test_local_calendar_flags_are_opt_in():
    idx = pd.DatetimeIndex(["2026-11-28 10:00", "2026-11-29 10:00"], tz="UTC")
    assert local_calendar_flags(idx, ()).empty
    flags = local_calendar_flags(idx, ("is_holiday",))
    assert flags["is_holiday"].tolist() == [1, 0]  # Independence Day
    with pytest.raises(ValueError):
        local_calendar_flags(idx, ("is_carnival",))


def test_same_hour_lag():
    assert [same_hour_lag(h) for h in (1, 6, 24, 25, 168)] == [24, 24, 24, 48, 168]
